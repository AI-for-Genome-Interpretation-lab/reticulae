"""Sparse pooling in pure PyTorch: max or mean over the ACTIVE inputs of every window (inactive sites are absent, not
zeros), on the windows of a regular convolution with kernel k, stride s (default k), padding p and dilation d: an
output site is active iff its window contains an active input. globalPool reduces every sample to one vector."""
import numpy as np
import torch as t
import torch.nn as nn
from reticulae import Rulebook as RB
from reticulae.Convolutions import SparseTensor, addRows, scatterAdd, gather

def pool(x, ins, outs, off, K, nOut, mode):
	"""x [N, B, C], pairs (ins -> outs) sorted by kernel offset off -> [nOut, B, C]: max or mean over the inputs of every
	output (the mean sums one offset at a time, as the convolutions: Convolutions.addRows)"""
	if mode == "max":
		y = gather(x, ins)
		# initialised to -inf, not left empty: the gradient of scatter_reduce compares the initial values with the result
		return x.new_full((nOut,) + x.shape[1:], -float("inf")).scatter_reduce_(0, outs[:, None, None].expand_as(y), y, "amax", include_self = False)
	inTab, outTab, _ = RB.tables(ins, outs, off, K, nOut)
	out = x.new_zeros((nOut + 1,) + x.shape[1:])
	addRows(out, outTab, gather(x, inTab.reshape(-1)))
	cnt = t.bincount(outs, minlength = nOut).to(x.dtype)
	return out[:nOut] / cnt[:, None, None]

class SparsePool(nn.Module):
	"""mode 'max' or 'avg' over the active inputs of every window (SparseTensor in, SparseTensor out)"""

	def __init__(self, D, k, s = None, p = 0, d = 1, mode = "max"):
		super().__init__()
		self.D, self.k, self.s, self.p, self.d, self.mode = D, RB.tup(k, D), RB.tup(k if s is None else s, D), RB.tup(p, D), RB.tup(d, D), mode

	def forward(self, x):
		outIdx, outShape, ins, outs, off = RB.regular(x.indices, x.spatialShape, x.batchSize, self.k, self.s, self.p, self.d)
		f = pool(x.features[:, None], ins, outs, off, len(RB.kernelOffsets(self.k, "cpu")), len(outIdx), self.mode)[:, 0]
		return SparseTensor(f, outIdx, outShape, x.batchSize, history = x.history)

	def extra_repr(self):
		return "D=%d, k=%s, s=%s, p=%s, d=%s, %s" % (self.D, self.k, self.s, self.p, self.d, self.mode)

class SparseMaxPool(SparsePool):
	def __init__(self, D, k, s = None, p = 0, d = 1): super().__init__(D, k, s, p, d, "max")
class SparseAvgPool(SparsePool):
	def __init__(self, D, k, s = None, p = 0, d = 1): super().__init__(D, k, s, p, d, "avg")
class SparseMaxPool1d(SparseMaxPool):
	def __init__(self, *a, **kw): super().__init__(1, *a, **kw)
class SparseMaxPool2d(SparseMaxPool):
	def __init__(self, *a, **kw): super().__init__(2, *a, **kw)
class SparseMaxPool3d(SparseMaxPool):
	def __init__(self, *a, **kw): super().__init__(3, *a, **kw)
class SparseAvgPool1d(SparseAvgPool):
	def __init__(self, *a, **kw): super().__init__(1, *a, **kw)
class SparseAvgPool2d(SparseAvgPool):
	def __init__(self, *a, **kw): super().__init__(2, *a, **kw)
class SparseAvgPool3d(SparseAvgPool):
	def __init__(self, *a, **kw): super().__init__(3, *a, **kw)

class FixedSparsePool(nn.Module):
	"""pooling (mode 'max' or 'avg') on active sites coords [N, D] shared by every sample: input [N, B, C] ->
	output [nOut, B, C] on the sites self.outCoords of the grid self.outShape"""

	def __init__(self, coords, spatialShape, k, s = None, p = 0, d = 1, mode = "max"):
		super().__init__()
		coords = t.as_tensor(np.asarray(coords), dtype = t.long)
		D = coords.shape[1]
		self.mode = mode
		idx = t.cat([t.zeros((len(coords), 1), dtype = t.long), coords], 1)
		outIdx, self.outShape, ins, outs, off = RB.regular(idx, RB.tup(spatialShape, D), 1, RB.tup(k, D), RB.tup(k if s is None else s, D), RB.tup(p, D), RB.tup(d, D))
		self.outCoords, self.nOut = outIdx[:, 1:].numpy(), len(outIdx)
		self.register_buffer("ins", ins, persistent = False)
		self.register_buffer("outs", outs, persistent = False)
		self.register_buffer("off", off, persistent = False)
		self.K = len(RB.kernelOffsets(RB.tup(k, D), "cpu"))

	def forward(self, x):
		return pool(x, self.ins, self.outs, self.off, self.K, self.nOut, self.mode)

def globalPool(x, mode = "max"):
	"""SparseTensor -> [B, C]: max, mean or sum of the features of every sample over its active sites (0 for a
	sample without active sites)"""
	B, C = x.batchSize, x.features.shape[1]
	b = x.indices[:, 0]
	if mode == "max":
		out = x.features.new_zeros((B, C)).scatter_reduce_(0, b[:, None].expand(-1, C), x.features, "amax", include_self = False)
		return out
	out = scatterAdd(x.features.new_zeros((B, C)), b, x.features)
	if mode == "mean":
		out = out / t.bincount(b, minlength = B).clamp(min = 1).to(out.dtype)[:, None]
	return out
