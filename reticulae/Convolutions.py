"""Sparse convolutions (regular, submanifold, transposed, inverse; dilation; any number of dimensions) in pure PyTorch.

The computation of a layer, given its rulebook grouped by kernel offset into padded [R, M] index tables (Rulebook.py),
is ONE gather of the input rows, ONE batched matmul over the R offsets (torch.bmm, each offset with its weight matrix)
and ONE scatter-add into the output rows (index_add_), whatever the kernel size; when the gathered rows would exceed
TEMP_BYTES the offsets are processed in a few chunks. Features are site-major, [sites, batch, channels]:
gathering/scattering a site moves a contiguous row. The backward pass (SparseConvFn) recomputes the gather instead of
storing the gathered input, and splits the long reduction of the weight gradient into chunks so that the GPU is not
left with only R x c_in x c_out outputs to compute.

Two ways to use it:
	SparseTensor + SparseConv / SubMConv / SparseConvTranspose / SparseInverseConv (and their 1d/2d/3d aliases):
	        every sample has its own active sites (as spconv); features [n_active, C], indices [n_active, 1 + D]
	        (sample, x_1..x_D); the rulebook is built on the device at every call (submanifold rulebooks are cached on
	        the tensor and reused by the following submanifold layers with the same kernel and dilation).
	FixedSparseConv / FixedSubMConv / FixedSparseConvTranspose / FixedSparseInverseConv: the same active sites in
	        every sample (e.g. all samples genotyped at the same loci); the rulebook is built once from the sites and
	        the batch is a dense dimension, features [N, B, C].
"""
import math
import numpy as np
import torch as t
import torch.nn as nn
from reticulae import Rulebook as RB

TEMP_BYTES = 2 ** 28   # budget of the temporary gathered rows of a layer: the kernel offsets are processed in chunks

def chunks(R, M, B, channels, elem):
	"""rows of the [R, M] tables per chunk so that M * B * channels values per row fit in TEMP_BYTES"""
	return max(1, min(R, TEMP_BYTES // max(1, M * B * channels * elem)))

def computeDtype(x):
	return t.get_autocast_dtype("cuda") if x.is_cuda and t.is_autocast_enabled("cuda") else x.dtype

class SparseConvFn(t.autograd.Function):
	"""x [N, B, cin], weight [K, cin, cout], tables inTab/outTab [R, M], offs [R] -> [nOut, B, cout] (no bias);
	centre >= 0 (submanifold, nOut = N): weight row of the identity offset, added as x @ weight[centre].
	Under torch.autocast the matmuls run in the autocast dtype; the gradients keep the dtypes of x and weight."""

	@staticmethod
	@t.amp.custom_fwd(device_type = "cuda")
	def forward(ctx, x, weight, inTab, outTab, offs, nOut, centre = -1):
		N, B, cin = x.shape
		R, M = inTab.shape
		cout = weight.shape[2]
		dt = computeDtype(x)
		out = x.new_zeros((nOut + 1, B, cout), dtype = dt)
		step = chunks(R, M, B, cin + cout, t.finfo(dt).bits // 8)
		for a in range(0, R, step):
			it = inTab[a:a + step]
			y = t.bmm(x[it.reshape(-1)].reshape(len(it), M * B, cin), weight[offs[a:a + step]])
			out.index_add_(0, outTab[a:a + step].reshape(-1), y.reshape(len(it) * M, B, cout).to(dt))
		out = out[:nOut]
		if centre >= 0:
			out += x @ weight[centre]
		ctx.save_for_backward(x, weight, inTab, outTab, offs)
		ctx.centre = centre
		return out

	@staticmethod
	@t.amp.custom_bwd(device_type = "cuda")
	def backward(ctx, go):
		x, weight, inTab, outTab, offs = ctx.saved_tensors
		N, B, cin = x.shape
		R, M = inTab.shape
		cout = weight.shape[2]
		gp = t.cat([go, go.new_zeros((1, B, cout))])
		gx = t.zeros_like(x) if ctx.needs_input_grad[0] else None
		gw = t.zeros((R, cin, cout), dtype = weight.dtype, device = weight.device) if ctx.needs_input_grad[1] else None
		step = chunks(R, M, B, 2 * cin + cout, go.element_size())
		for a in range(0, R, step):
			it, ot, r = inTab[a:a + step], outTab[a:a + step], min(step, R - a)
			# weight gradient = r reductions over M*B rows: split them into S chunks (padding the tables) for parallelism
			S = max(1, min(math.ceil(1024 / (r * math.ceil(cin / 64) * math.ceil(cout / 64))), (M * B) // 2048)) if gw is not None else 1
			Mp = math.ceil(M / S) * S
			if Mp > M:
				it = t.nn.functional.pad(it, (0, Mp - M), value = 0)
				ot = t.nn.functional.pad(ot, (0, Mp - M), value = go.shape[0])
			gg = gp[ot.reshape(-1)].reshape(r, Mp * B, cout)
			if gx is not None:
				gx.index_add_(0, it.reshape(-1), t.bmm(gg, weight[offs[a:a + step]].transpose(1, 2)).reshape(r * Mp, B, cin).to(x.dtype))
			if gw is not None:
				g = x[it.reshape(-1)].reshape(r * S, (Mp // S) * B, cin)
				gw[a:a + step] = t.bmm(g.transpose(1, 2), gg.reshape(r * S, (Mp // S) * B, cout)).reshape(r, S, cin, cout).sum(1).to(weight.dtype)
		if gw is not None:
			gw = t.zeros_like(weight).index_add_(0, offs, gw)
		if ctx.centre >= 0:
			if gx is not None:
				gx += (go @ weight[ctx.centre].t()).to(x.dtype)
			if gw is not None:
				gw[ctx.centre] += (x.reshape(-1, cin).t() @ go.reshape(-1, cout)).to(weight.dtype)
		return gx, gw, None, None, None, None, None

class SparseTensor():
	"""A batch of sparse images whose active sites may differ between samples.
		features     [n, C] float, one row per active site
		indices      [n, 1 + D] int64 rows (sample, x_1, ..., x_D); every site at most once
		spatialShape (S_1, ..., S_D), batchSize B
	cache (rulebooks of submanifold layers, valid for these indices) and history (rulebooks of regular convolutions
	with a key, read by SparseInverseConv) are carried along by the layers."""

	def __init__(self, features, indices, spatialShape, batchSize, cache = None, history = None):
		self.features = features
		self.indices = indices.long()
		self.spatialShape = tuple(int(a) for a in spatialShape)
		self.batchSize = int(batchSize)
		self.cache = {} if cache is None else cache
		self.history = {} if history is None else history

	@staticmethod
	def fromDense(img):
		"""[B, C, S_1, .., S_D] dense tensor -> SparseTensor of the sites where any channel is non-zero"""
		D = img.dim() - 2
		f = img.permute(0, *range(2, D + 2), 1)
		idx = t.nonzero((f != 0).any(-1))
		return SparseTensor(f[tuple(idx.t())], idx, img.shape[2:], img.shape[0])

	def replace(self, features):
		"""same sites, new features (e.g. after a normalisation or an activation)"""
		return SparseTensor(features, self.indices, self.spatialShape, self.batchSize, self.cache, self.history)

	def dense(self):
		"""[B, C, S_1, .., S_D] zero-filled"""
		C = self.features.shape[1]
		out = self.features.new_zeros((self.batchSize * math.prod(self.spatialShape), C))
		out[RB.linear(self.indices[:, 0], self.indices[:, 1:], self.spatialShape)] = self.features
		D = len(self.spatialShape)
		return out.reshape((self.batchSize,) + self.spatialShape + (C,)).permute(0, D + 1, *range(1, D + 1))

class _Conv(nn.Module):
	"""weights [prod(k), cin, cout] (kernel offsets row-major), bias [cout]; mode regular | subm | transposed | inverse"""

	def __init__(self, D, cin, cout, k, s, p, d, mode, bias):
		super().__init__()
		self.D, self.k, self.s, self.p, self.d, self.mode = D, RB.tup(k, D), RB.tup(s, D), RB.tup(p, D), RB.tup(d, D), mode
		if mode == "subm":
			assert self.s == (1,) * D and self.p == (0,) * D, "submanifold convolutions have stride 1 and no padding"
		K = math.prod(self.k)
		self.weight = nn.Parameter(t.empty(K, cin, cout))
		self.bias = nn.Parameter(t.zeros(cout)) if bias else None
		# xavier-uniform on the full kernel (fan_in = K*cin, fan_out = K*cout), as init_weights in SchizophreniaGI
		bound = np.sqrt(6.0 / (K * cin + K * cout))
		nn.init.uniform_(self.weight, -bound, bound)

	def rulebook(self, idx, shape, batch):
		"""-> output idx, output shape, pairs (ins, outs, offset), tables (inTab, outTab, offs, centre)"""
		K = self.weight.shape[0]
		if self.mode == "subm":
			ins, outs, off, centre = RB.submanifold(idx, shape, batch, self.k, self.d)
			return idx, shape, (ins, outs, off), RB.tables(ins, outs, off, K, len(idx)) + (centre,)
		build = RB.transposed if self.mode == "transposed" else RB.regular
		outIdx, outShape, ins, outs, off = build(idx, shape, batch, self.k, self.s, self.p, self.d)
		return outIdx, outShape, (ins, outs, off), RB.tables(ins, outs, off, K, len(outIdx)) + (-1,)

	def conv(self, x, tabs, nOut):
		y = SparseConvFn.apply(x, self.weight, tabs[0], tabs[1], tabs[2], nOut, tabs[3])
		return y + self.bias.to(y.dtype) if self.bias is not None else y

	def extra_repr(self):
		return "D=%d, %d -> %d, k=%s, s=%s, p=%s, d=%s, %s" % (self.D, self.weight.shape[1], self.weight.shape[2], self.k, self.s, self.p, self.d, self.mode)

# ---------------------------------------------------------------- every sample with its own active sites

class SparseConv(_Conv):
	"""regular sparse convolution: out[o] = b + sum_delta W[delta] in[s*o + d*delta - p] over the active inputs; an
	output site is active iff its receptive field contains an active input. key: name under which the rulebook is
	kept for a SparseInverseConv(key = ...) that maps back to the input sites"""

	def __init__(self, D, cin, cout, k, s = 1, p = 0, d = 1, bias = True, key = None):
		super().__init__(D, cin, cout, k, s, p, d, "regular", bias)
		self.key = key

	def forward(self, x):
		outIdx, outShape, pairs, tabs = self.rulebook(x.indices, x.spatialShape, x.batchSize)
		if self.key is not None:
			x.history[self.key] = (x.indices, x.spatialShape, pairs, self.k, outIdx)
		f = self.conv(x.features[:, None], tabs, len(outIdx))[:, 0]
		return SparseTensor(f, outIdx, outShape, x.batchSize, history = x.history)

class SubMConv(_Conv):
	"""submanifold sparse convolution (k odd): out[o] = b + sum_delta W[delta] in[o + d*(delta - (k-1)/2)]; output
	sites = input sites, so the sparsity pattern never grows"""

	def __init__(self, D, cin, cout, k, d = 1, bias = True):
		super().__init__(D, cin, cout, k, 1, 0, d, "subm", bias)

	def forward(self, x):
		key = ("subm", self.k, self.d)
		if key not in x.cache:
			x.cache[key] = self.rulebook(x.indices, x.spatialShape, x.batchSize)[3]
		return x.replace(self.conv(x.features[:, None], x.cache[key], len(x.indices))[:, 0])

class SparseConvTranspose(_Conv):
	"""transposed sparse convolution: out[s*x + d*delta - p] += W[delta] in[x] (+ b); an output site is active iff it
	receives from an active input; output shape (S - 1)*s - 2p + d*(k - 1) + 1 (= torch conv_transposeNd)"""

	def __init__(self, D, cin, cout, k, s = 1, p = 0, d = 1, bias = True):
		super().__init__(D, cin, cout, k, s, p, d, "transposed", bias)

	def forward(self, x):
		outIdx, outShape, pairs, tabs = self.rulebook(x.indices, x.spatialShape, x.batchSize)
		f = self.conv(x.features[:, None], tabs, len(outIdx))[:, 0]
		return SparseTensor(f, outIdx, outShape, x.batchSize, history = x.history)

class SparseInverseConv(_Conv):
	"""inverse of the SparseConv(key = key) applied earlier (same kernel size): back to its input sites and shape, with
	the pairs of its rulebook reversed (out[x] = b + sum W[delta] in[o] over the pairs x -> o); the input must have
	the output sites of that convolution (e.g. after submanifold layers). Used in sparse U-Nets."""

	def __init__(self, D, cin, cout, k, key, bias = True):
		super().__init__(D, cin, cout, k, 1, 0, 1, "inverse", bias)
		self.key = key

	def forward(self, x):
		inIdx, inShape, (ins, outs, off), k, outIdx = x.history[self.key]
		assert k == self.k, "the inverse convolution needs the kernel size of the convolution it inverts"
		assert t.equal(x.indices, outIdx), "the input of SparseInverseConv must have the output sites of SparseConv(key = %r)" % self.key
		tabs = RB.tables(outs, ins, off, self.weight.shape[0], len(inIdx)) + (-1,)
		f = self.conv(x.features[:, None], tabs, len(inIdx))[:, 0]
		return SparseTensor(f, inIdx, inShape, x.batchSize, history = x.history)

class SparseConv1d(SparseConv):
	def __init__(self, *a, **kw): super().__init__(1, *a, **kw)
class SparseConv2d(SparseConv):
	def __init__(self, *a, **kw): super().__init__(2, *a, **kw)
class SparseConv3d(SparseConv):
	def __init__(self, *a, **kw): super().__init__(3, *a, **kw)
class SubMConv1d(SubMConv):
	def __init__(self, *a, **kw): super().__init__(1, *a, **kw)
class SubMConv2d(SubMConv):
	def __init__(self, *a, **kw): super().__init__(2, *a, **kw)
class SubMConv3d(SubMConv):
	def __init__(self, *a, **kw): super().__init__(3, *a, **kw)
class SparseConvTranspose1d(SparseConvTranspose):
	def __init__(self, *a, **kw): super().__init__(1, *a, **kw)
class SparseConvTranspose2d(SparseConvTranspose):
	def __init__(self, *a, **kw): super().__init__(2, *a, **kw)
class SparseConvTranspose3d(SparseConvTranspose):
	def __init__(self, *a, **kw): super().__init__(3, *a, **kw)
class SparseInverseConv1d(SparseInverseConv):
	def __init__(self, *a, **kw): super().__init__(1, *a, **kw)
class SparseInverseConv2d(SparseInverseConv):
	def __init__(self, *a, **kw): super().__init__(2, *a, **kw)
class SparseInverseConv3d(SparseInverseConv):
	def __init__(self, *a, **kw): super().__init__(3, *a, **kw)

# ---------------------------------------------------------------- the same active sites in every sample

class _Fixed(_Conv):
	"""rulebook built once from the active sites coords [N, D] (shared by every sample); input [N, B, cin] ->
	output [nOut, B, cout] on the sites self.outCoords [nOut, D] of the grid self.outShape"""

	def __init__(self, coords, spatialShape, cin, cout, k, s, p, d, mode, bias):
		coords = t.as_tensor(np.asarray(coords), dtype = t.long)
		D = coords.shape[1]
		super().__init__(D, cin, cout, k, s, p, d, mode, bias)
		self.inShape = RB.tup(spatialShape, D)
		self.inCoords = coords.numpy()
		idx = t.cat([t.zeros((len(coords), 1), dtype = t.long), coords], 1)
		outIdx, self.outShape, self.pairs, tabs = self.rulebook(idx, self.inShape, 1)
		self.setTables(tabs, len(coords), outIdx[:, 1:].numpy())

	def setTables(self, tabs, nIn, outCoords):
		self.outCoords = outCoords
		self.nIn, self.nOut = nIn, len(outCoords)
		self.nPairs = int((tabs[1] < self.nOut).sum()) + (nIn if tabs[3] >= 0 else 0)
		self.register_buffer("inTab", tabs[0], persistent = False)
		self.register_buffer("outTab", tabs[1], persistent = False)
		self.register_buffer("offs", tabs[2], persistent = False)
		self.centre = tabs[3]

	def forward(self, x):
		return self.conv(x, (self.inTab, self.outTab, self.offs, self.centre), self.nOut)

class FixedSparseConv(_Fixed):
	"""regular sparse convolution (as SparseConv) on sites shared by every sample"""

	def __init__(self, coords, spatialShape, cin, cout, k, s = 1, p = 0, d = 1, bias = True):
		super().__init__(coords, spatialShape, cin, cout, k, s, p, d, "regular", bias)

class FixedSubMConv(_Fixed):
	"""submanifold sparse convolution (as SubMConv) on sites shared by every sample; outCoords = coords"""

	def __init__(self, coords, spatialShape, cin, cout, k, d = 1, bias = True):
		super().__init__(coords, spatialShape, cin, cout, k, 1, 0, d, "subm", bias)

class FixedSparseConvTranspose(_Fixed):
	"""transposed sparse convolution (as SparseConvTranspose) on sites shared by every sample"""

	def __init__(self, coords, spatialShape, cin, cout, k, s = 1, p = 0, d = 1, bias = True):
		super().__init__(coords, spatialShape, cin, cout, k, s, p, d, "transposed", bias)

class FixedSparseInverseConv(_Conv):
	"""inverse of a FixedSparseConv `conv`: from its output sites back to its input sites (conv.inCoords), with the
	pairs of its rulebook reversed (as SparseInverseConv)"""

	def __init__(self, conv, cin, cout, bias = True):
		super().__init__(conv.D, cin, cout, conv.k, 1, 0, 1, "inverse", bias)
		ins, outs, off = conv.pairs
		self.outShape = conv.inShape
		_Fixed.setTables(self, RB.tables(outs, ins, off, self.weight.shape[0], conv.nIn) + (-1,), conv.nOut, conv.inCoords)

	forward = _Fixed.forward
