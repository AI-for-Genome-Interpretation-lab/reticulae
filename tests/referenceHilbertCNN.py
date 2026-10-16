"""Reference implementation used by the tests: the Hilbert-curve sparse CNN written as a straightforward loop over the
kernel offsets (one gather, matmul and scatter-add per offset, features [batch, sites, channels]). It computes the same
function as reticulae.SparseHilbertCNN with the same parameters; tests/test_reticulae.py checks predictions and
gradients against it.
"""
import numpy as np
import torch as t
import torch.nn as nn

def rulebook(coords, size, k, s):
	"""coords [N, 2] int64 active input sites on a size x size grid. Returns output coords, output size and, for
	every kernel offset, the (input index, output index) pairs"""
	outSize = (size - k) // s + 1
	ins, outs, offs = [], [], []
	for dx in range(k):
		for dy in range(k):
			ox, oy = coords[:, 0] - dx, coords[:, 1] - dy
			ok = (ox >= 0) & (oy >= 0) & (ox % s == 0) & (oy % s == 0)
			ox, oy = ox // s, oy // s
			ok &= (ox < outSize) & (oy < outSize)
			idx = np.nonzero(ok)[0]
			ins.append(idx)
			outs.append(ox[idx] * outSize + oy[idx])
			offs.append(np.full(len(idx), dx * k + dy))
	ins, outs, offs = np.concatenate(ins), np.concatenate(outs), np.concatenate(offs)
	keys, outIdx = np.unique(outs, return_inverse = True)
	outCoords = np.stack([keys // outSize, keys % outSize], 1)
	order = np.argsort(offs, kind = "stable")
	ins, outIdx, offs = ins[order], outIdx[order], offs[order]
	bounds = np.searchsorted(offs, np.arange(k * k + 1))
	return outCoords, outSize, ins, outIdx, bounds

class SparseConv(nn.Module):
	"""regular sparse 2-D convolution with a fixed, precomputed rulebook; input/output [B, N, C]"""
	def __init__(self, coords, size, cin, cout, k, s):
		super().__init__()
		outCoords, outSize, ins, outIdx, bounds = rulebook(coords, size, k, s)
		self.outCoords, self.outSize, self.nOut = outCoords, outSize, len(outCoords)
		self.register_buffer("ins", t.as_tensor(ins, dtype = t.long))
		self.register_buffer("outIdx", t.as_tensor(outIdx, dtype = t.long))
		self.bounds = bounds.tolist()
		self.k2 = k * k
		self.weight = nn.Parameter(t.empty(k * k, cin, cout))
		self.bias = nn.Parameter(t.zeros(cout))
		# xavier-uniform on the full kernel (fan_in = k*k*cin, fan_out = k*k*cout), as init_weights in SchizophreniaGI
		bound = np.sqrt(6.0 / (k * k * cin + k * k * cout))
		nn.init.uniform_(self.weight, -bound, bound)
		self.nPairs = len(ins)

	def forward(self, x):
		B = x.shape[0]
		out = x.new_zeros((B, self.nOut, self.weight.shape[2]))
		for d in range(self.k2):
			a, b = self.bounds[d], self.bounds[d + 1]
			if a == b:
				continue
			out.index_add_(1, self.outIdx[a:b], x[:, self.ins[a:b]] @ self.weight[d])
		return out + self.bias

class SparseHilbertCNN(nn.Module):
	inputType = "snp"

	def __init__(self, nSnps, nTraits, xy, order, channels = (1, 4, 16, 32, 64, 128, 128, 128), dropout = 0.3, hidden = 50, name = "hilbert_locus"):
		super().__init__()
		self.name = name
		xy = np.asarray(xy, dtype = np.int64)
		keys, site = np.unique(xy[:, 0] * (1 << order) + xy[:, 1], return_inverse = True)
		self.register_buffer("site", t.as_tensor(site, dtype = t.long))          # SNP -> active input site
		self.register_buffer("count", t.as_tensor(np.bincount(site).astype(np.float32)))
		self.nSites = len(keys)
		coords = np.stack([keys >> order, keys & ((1 << order) - 1)], 1)
		size = 1 << order
		# depth adapted to the curve size: up to two 8x8 stride-8 layers (learned patch pooling) while the grid stays
		# >= 32, then 3x3 stride-2 layers until the grid is <= 20 (HCspconv1 of SchizophreniaGI for a 2^16 curve)
		specs, S, cin = [], size, 1
		for c in channels[:2]:
			if S // 8 >= 32:
				specs.append((cin, c, 8, 8)); cin = c; S = (S - 8) // 8 + 1
		chans = [c for c in channels[2:]] + [channels[-1]] * 8
		i = 0
		while S > 20:
			specs.append((cin, chans[i], 3, 2)); cin = chans[i]; S = (S - 3) // 2 + 1; i += 1
		layers, norms = [], []
		self.levels = []
		for li, (ci, co, k, s) in enumerate(specs):
			conv = SparseConv(coords, size, ci, co, k, s)
			self.levels.append((size, len(coords), conv.outSize, conv.nOut, conv.nPairs))
			coords, size = conv.outCoords, conv.outSize
			layers.append(conv)
			norms.append(nn.LayerNorm(co) if li > 0 else nn.Identity())
		self.convs = nn.ModuleList(layers)
		self.norms = nn.ModuleList(norms)
		self.act = nn.LeakyReLU()
		self.finalSize = size
		self.register_buffer("finalFlat", t.as_tensor(coords[:, 0] * size + coords[:, 1], dtype = t.long))
		cLast = specs[-1][1]
		self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(size * size * cLast, hidden), nn.LeakyReLU(), nn.Linear(hidden, nTraits))
		for m in self.head:
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				nn.init.zeros_(m.bias)
		self.register_buffer("mean", t.zeros(nSnps))
		self.inp = nn.Identity()   # hook point on the per-SNP dosages (saliency in SNP order, runCV interpretation)

	def prepare(self, X, Y = None):
		X = np.asarray(X)
		mu = np.zeros(X.shape[1])
		cnt = np.zeros(X.shape[1])
		for i in range(0, len(X), 2048):
			g = X[i:i + 2048].astype(np.float64)
			ok = X[i:i + 2048] != -127
			mu += np.where(ok, g, 0).sum(0)
			cnt += ok.sum(0)
		self.mean.copy_(t.as_tensor(mu / np.maximum(cnt, 1), dtype = t.float32))

	def describe(self):
		return "\n".join("level %d: %d x %d grid, %d active -> %d x %d grid, %d active (%d rulebook pairs)" % ((i,) + (lv[0], lv[0], lv[1], lv[2], lv[2], lv[3], lv[4])) for i, lv in enumerate(self.levels))

	def sparseInput(self, x):
		"""int8 dosages [B, m] -> features of the active sites [B, N, 1] (mean dosage if loci share a pixel)"""
		g = x.float()
		g = t.where(x == -127, self.mean.expand_as(g), g)
		g = self.inp(g)
		f = g.new_zeros((g.shape[0], self.nSites)).index_add_(1, self.site, g) / self.count
		return f.unsqueeze(2)

	def forward(self, x, idx = None):
		h = self.sparseInput(x)
		for li, (conv, norm) in enumerate(zip(self.convs, self.norms)):
			h = conv(h)
			h = self.act(norm(h)) if li > 0 else self.act(h)
		B, N, C = h.shape
		dense = h.new_zeros((B, self.finalSize * self.finalSize, C))
		dense[:, self.finalFlat] = h
		return {"pred": self.head(dense.reshape(B, -1))}
