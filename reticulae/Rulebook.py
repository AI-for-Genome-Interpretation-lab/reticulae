"""Rulebooks of sparse convolutions in D dimensions, built with torch on the device of the coordinates.

A rulebook is the list of (input site, output site, kernel offset) triples of a convolution. Coordinates are int64
[n, 1 + D] rows (sample, x_1, ..., x_D); a site is identified by its linear key (sample, x_1, ..., x_D) in row-major
order, so any grid with batch x prod(shape) < 2^62 is supported. Offsets delta are row-major over the kernel
(delta_1 slowest), which is also the order of the rows of the weight tensor [prod(k), c_in, c_out].

	regular       out[o] = sum_delta W[delta] in[s*o + d*delta - p]      (output active iff its receptive field
	              contains an active input; = torch convNd / spconv SparseConvNd)
	transposed    out[s*x + d*delta - p] += W[delta] in[x]               (output active iff it receives from an active
	              input; = torch conv_transposeNd / spconv SparseConvTransposeNd)
	submanifold   out[o] = sum_delta W[delta] in[o + d*(delta - (k-1)/2)] (output sites = input sites, k odd; = spconv
	              SubMConvNd); the centre offset (every site with itself) is not in the rulebook but computed as a
	              plain matmul of all the sites
	inverse       the pairs of a regular convolution with input and output swapped: back to its input sites

The pairs are then grouped by offset and every group is padded to the size of the largest (tables): [R, M] index
tables over the R offsets that have pairs, input padding = row 0 (multiplied by a zero output gradient, never used
in the output), output padding = an extra discarded row nOut.
"""
import math
import torch as t

def tup(v, D):
	return tuple(int(a) for a in v) if isinstance(v, (tuple, list)) else (int(v),) * D

def kernelOffsets(k, device):
	"""[prod(k), D] offsets, row-major"""
	g = t.meshgrid(*[t.arange(a, device = device) for a in k], indexing = "ij")
	return t.stack([a.reshape(-1) for a in g], 1)

def linear(b, c, shape):
	"""sample b [n], coordinates c [n, D] on grid shape -> int64 key"""
	key = b.clone()
	for i, S in enumerate(shape):
		key = key * S + c[:, i]
	return key

def unlinear(key, shape):
	"""key -> [n, 1 + D] (sample, x_1..x_D)"""
	cols = []
	for S in reversed(shape):
		cols.append(key % S)
		key = key // S
	return t.stack([key] + cols[::-1], 1)

def checkSize(batch, shape):
	n = batch
	for S in shape:
		n *= S
	assert n < 2 ** 62, "batch x grid size must be < 2^62"

def strides(shape):
	"""row-major strides of a grid, and its size"""
	st = [1] * len(shape)
	for i in range(len(shape) - 2, -1, -1):
		st[i] = st[i + 1] * shape[i + 1]
	return st, st[0] * shape[0]

def separable(idx, k, oks, vals, outShape, chunk):
	"""pairs from per-dimension validity oks[i] [n, k_i] and output coordinates vals[i] [n, k_i]: combined over the
	prod(k) offsets by broadcasting, offset-major; output keys made unique -> output idx, pairs (ins, outs, offset)"""
	dev = idx.device
	ost, ovol = strides(outShape)
	b, n, D = idx[:, 0], len(idx), len(k)
	offs = kernelOffsets(k, dev)
	ins, keys, off = [], [], []
	step = max(1, chunk // max(1, n))
	for a in range(0, len(offs), step):
		o = offs[a:a + step]                                                               # [K', D]
		ok = oks[0][:, o[:, 0]]
		for i in range(1, D):
			ok = ok & oks[i][:, o[:, i]]
		d, i = t.nonzero(ok.t(), as_tuple = True)                                          # offset-major
		key = b[i] * ovol
		for j in range(D):
			key = key + vals[j][i, o[d, j]] * ost[j]
		ins.append(i); keys.append(key); off.append(d + a)
	ins, keys, off = t.cat(ins), t.cat(keys), t.cat(off)
	uk, outs = t.unique(keys, return_inverse = True)
	return unlinear(uk, outShape), ins, outs, off

def regular(idx, shape, batch, k, s, p, d, chunk = 2 ** 26):
	"""idx [n, 1 + D] -> output idx, output shape, pairs (ins, outs, offset) sorted by offset.
	Input x reaches output o = (x + p - d*delta) / s when it is an integer inside the output grid: the validity of a
	pair and its output coordinate are separable over dimensions ([n, k_i] per dimension)"""
	D = len(shape)
	outShape = tuple((shape[i] + 2 * p[i] - d[i] * (k[i] - 1) - 1) // s[i] + 1 for i in range(D))
	assert min(outShape) > 0, "kernel larger than the padded input"
	checkSize(batch, shape); checkSize(batch, outShape)
	oks, vals = [], []
	for i in range(D):
		v = idx[:, 1 + i, None] + p[i] - d[i] * t.arange(k[i], device = idx.device)[None]   # [n, k_i]
		oks.append((v >= 0) & (v % s[i] == 0) & (v // s[i] < outShape[i]))
		vals.append(v // s[i])
	outIdx, ins, outs, off = separable(idx, k, oks, vals, outShape, chunk)
	return outIdx, outShape, ins, outs, off

def transposed(idx, shape, batch, k, s, p, d, chunk = 2 ** 26):
	"""idx [n, 1 + D] -> output idx, output shape, pairs (ins, outs, offset) sorted by offset.
	Input x reaches output o = s*x + d*delta - p when it is inside the output grid"""
	D = len(shape)
	outShape = tuple((shape[i] - 1) * s[i] - 2 * p[i] + d[i] * (k[i] - 1) + 1 for i in range(D))
	assert min(outShape) > 0, "padding larger than the output"
	checkSize(batch, outShape)
	oks, vals = [], []
	for i in range(D):
		v = idx[:, 1 + i, None] * s[i] - p[i] + d[i] * t.arange(k[i], device = idx.device)[None]   # [n, k_i]
		oks.append((v >= 0) & (v < outShape[i]))
		vals.append(v)
	outIdx, ins, outs, off = separable(idx, k, oks, vals, outShape, chunk)
	return outIdx, outShape, ins, outs, off

def submanifold(idx, shape, batch, k, d, chunk = 2 ** 26):
	"""idx [n, 1 + D] (unique sites) -> pairs (ins, outs, offset) sorted by offset, WITHOUT the centre offset (every
	site with itself, computed apart), and the weight row of the centre; output sites = input sites.
	Sites are keyed on the grid padded by d*(k-1)/2 on every side, so that the key of a neighbour is the key of the
	site plus a constant per offset and neighbours outside the grid can never match a site."""
	assert all(a % 2 == 1 for a in k), "submanifold convolutions need odd kernel sizes"
	D = len(shape)
	m = [d[i] * (k[i] // 2) for i in range(D)]
	padded = tuple(shape[i] + 2 * m[i] for i in range(D))
	checkSize(batch, padded)
	dev = idx.device
	st, vol = strides(padded)
	key = idx[:, 0] * vol
	for i in range(D):
		key = key + (idx[:, 1 + i] + m[i]) * st[i]
	sk, perm = t.sort(key)
	offs = (kernelOffsets(k, dev) - t.tensor([a // 2 for a in k], device = dev)) * t.tensor(d, device = dev)
	offKey = (offs * t.tensor(st, device = dev)).sum(1)
	K, centre = len(offs), len(offs) // 2
	rows = t.cat([t.arange(centre, device = dev), t.arange(centre + 1, K, device = dev)])
	e = t.zeros(0, dtype = t.long, device = dev)
	ins, outs, off = [e], [e], [e]
	step = max(1, chunk // max(1, len(idx)))
	for a in range(0, len(rows), step):
		r = rows[a:a + step]
		q = key[None] + offKey[r, None]                                                    # [K', n]
		pos = t.searchsorted(sk, q).clamp(max = len(sk) - 1)
		d, i = t.nonzero(sk[pos] == q, as_tuple = True)                                    # offset-major
		ins.append(perm[pos[d, i]]); outs.append(i); off.append(r[d])
	return t.cat(ins), t.cat(outs), t.cat(off), centre

def tables(ins, outs, off, K, nOut):
	"""pairs sorted by offset -> padded index tables inTab, outTab [R, M] and the weight row of each table row [R]"""
	cnt = t.bincount(off, minlength = K)
	used = t.nonzero(cnt)[:, 0]
	R, M = len(used), int(cnt.max()) if len(off) else 0
	row = t.full((K,), -1, dtype = t.long, device = off.device)
	row[used] = t.arange(R, device = off.device)
	pos = t.arange(len(off), device = off.device) - (t.cumsum(cnt, 0) - cnt)[off]
	inTab = t.zeros((R, M), dtype = t.long, device = off.device)
	outTab = t.full((R, M), nOut, dtype = t.long, device = off.device)
	inTab[row[off], pos] = ins
	outTab[row[off], pos] = outs
	return inTab, outTab, used
