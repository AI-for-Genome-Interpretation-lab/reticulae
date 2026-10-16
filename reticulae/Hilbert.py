"""Sparse convolutional network on locus-based Hilbert-curve images of the genome.

The image of every sample is a 2^order x 2^order Hilbert curve of the genome (one base pair per pixel, hilbertXY) in
which only the genotyped loci are active; the pixel value is the allele dosage (0/1/2, missing = training mean).
Only the active sites are stored: the input of the network is the [samples, loci] dosage matrix.

Architecture = HCspconv1 of SchizophreniaGI (Verplaetse, Moreau & Raimondi): SparseConv(1,1,8,8), SparseConv(1,4,8,8),
then SparseConv(3x3, stride 2) with 16, 32, 64, 128, 128, 128 channels, LayerNorm + LeakyReLU after every layer
(except the first), densified final map -> dropout -> Linear(., 50) -> LeakyReLU -> Linear(50, traits).

SparseHilbertCNN computes the same function, with the same parameters, as the loop implementation in
tests/referenceHilbertCNN.py, on site-major features (see Convolutions.py).
"""
import numpy as np
import torch as t
import torch.nn as nn
from reticulae.Convolutions import FixedSparseConv

def d2xy(order, d):
	"""Hilbert curve index -> (x, y), vectorised (classic iterative algorithm)"""
	n = 1 << order
	x = np.zeros_like(d)
	y = np.zeros_like(d)
	tt = d.copy()
	s = 1
	while s < n:
		rx = 1 & (tt // 2)
		ry = 1 & (tt ^ rx)
		swap = ry == 0
		flip = swap & (rx == 1)
		x = np.where(flip, s - 1 - x, x)
		y = np.where(flip, s - 1 - y, y)
		x, y = np.where(swap, y, x), np.where(swap, x, y)
		x = x + s * rx
		y = y + s * ry
		tt = tt // 4
		s *= 2
	return x, y

def hilbertXY(chrom, pos, chromLengths = None):
	"""locus-based Hilbert curve: chromosomes concatenated, one base pair per curve position,
	order = ceil(log2(sqrt(genome length))). chrom, pos (1-based) per locus -> xy [m, 2], order.
	chromLengths {name: length}: the genome layout (chromosome order = order of the dict, lengths), the same for every
	call; required to map the mutations of different samples consistently. Without it, the chromosomes are taken in
	order of appearance with the last locus as length (a panel of loci mapped once)."""
	chrom = np.asarray(chrom).astype(str)
	pos = np.asarray(pos, dtype = np.int64)
	if chromLengths:
		chromLengths = {str(c): L for c, L in chromLengths.items()}
		assert set(chrom) <= set(chromLengths), "chromosomes missing from chromLengths"
		assert (pos >= 1).all() and (pos <= np.array([chromLengths[c] for c in chrom])).all(), "positions outside the chromosomes"
	off, offsets = 0, {}
	for c in (chromLengths if chromLengths else dict.fromkeys(chrom)):
		offsets[c] = off
		L = chromLengths.get(c) if chromLengths else None
		off += int(L if L else pos[chrom == c].max() + 1)
	order = int(np.ceil(np.log2(np.sqrt(off))))
	g = np.array([offsets[c] for c in chrom], dtype = np.int64) + pos - 1
	x, y = d2xy(order, g)
	return np.stack([x, y], 1), order

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
			conv = FixedSparseConv(coords, (size, size), ci, co, k, s)
			self.levels.append((size, len(coords), conv.outShape[0], conv.nOut, conv.nPairs))
			coords, size = conv.outCoords, conv.outShape[0]
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
		self.inp = nn.Identity()   # hook point on the per-SNP dosages (saliency in SNP order)

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
		"""int8 dosages [B, m] -> features of the active sites [N, B, 1] (mean dosage if loci share a pixel)"""
		g = x.float()
		g = t.where(x == -127, self.mean.expand_as(g), g)
		g = self.inp(g)
		f = g.new_zeros((self.nSites, g.shape[0])).index_add_(0, self.site, g.t()) / self.count[:, None]
		return f.unsqueeze(2)

	def forward(self, x, idx = None):
		h = self.sparseInput(x)
		for li, (conv, norm) in enumerate(zip(self.convs, self.norms)):
			h = conv(h)
			h = self.act(norm(h)) if li > 0 else self.act(h)
		N, B, C = h.shape
		dense = h.new_zeros((self.finalSize * self.finalSize, B, C))
		dense[self.finalFlat] = h
		return {"pred": self.head(dense.transpose(0, 1).reshape(B, -1))}   # flattened as (row, column, channel)
