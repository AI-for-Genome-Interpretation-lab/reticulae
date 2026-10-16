"""Compact storage of sparse images (CPU, numpy), converted to SparseTensor batches on demand.

PackedArray   values with the fewest bits: integers in [0, 2^b) with b = 1, 2 or 4 are bit-packed (8/b values per
              byte, e.g. dosages 1/2 -> 2 bits), other integers use the smallest numpy integer type, floats keep their
              dtype, and an array whose values are all equal stores only that value.
SparseImages  many samples, each with its own active sites: the coordinates of every dimension in the smallest
              unsigned type that holds the grid size (uint8 up to 256, uint16 up to 65,536, ...), the values as a
              PackedArray, and the sample boundaries as offsets (sample i = rows offsets[i]:offsets[i+1]).
An unobserved site (e.g. a reference base) is simply absent and costs nothing; a stored site with value 0 is an
active site whose value is 0.
"""
import numpy as np
import torch as t
from reticulae.Convolutions import SparseTensor

def smallestUint(maxValue):
	for dt in (np.uint8, np.uint16, np.uint32, np.uint64):
		if maxValue <= np.iinfo(dt).max:
			return dt

def smallestInt(lo, hi):
	if lo >= 0:
		return smallestUint(hi)
	for dt in (np.int8, np.int16, np.int32, np.int64):
		if np.iinfo(dt).min <= lo and hi <= np.iinfo(dt).max:
			return dt

class PackedArray():
	"""array [n] or [n, C] stored with the fewest bits; a[rows] decodes the given rows (numpy)"""

	def __init__(self, values):
		v = np.asarray(values)
		self.shape, self.dtype = v.shape, v.dtype
		self.bits, self.const, self.data = None, None, None
		if v.size and (v == v.flat[0]).all():
			self.const = v.flat[0]
		elif v.dtype.kind in "iub" and v.size:
			lo, hi = int(v.min()), int(v.max())
			self.bits = next((b for b in (1, 2, 4) if lo >= 0 and hi < 2 ** b), None)
			if self.bits is None:
				self.data = v.astype(smallestInt(lo, hi))
			else:
				per = 8 // self.bits
				flat = np.zeros(-(-v.size // per) * per, dtype = np.uint8)
				flat[:v.size] = v.reshape(-1)
				flat = flat.reshape(-1, per)
				self.data = np.zeros(len(flat), dtype = np.uint8)
				for j in range(per):
					self.data |= flat[:, j] << (j * self.bits)
		else:
			self.data = v

	def __len__(self):
		return self.shape[0]

	@property
	def nbytes(self):
		return 0 if self.data is None else self.data.nbytes

	def __getitem__(self, rows):
		rows = np.asarray(rows, dtype = np.int64)
		shape = (len(rows),) + self.shape[1:]
		if self.const is not None:
			return np.full(shape, self.const, dtype = self.dtype)
		if self.bits is None:
			return self.data[rows]
		C = int(np.prod(self.shape[1:]))
		i = (rows[:, None] * C + np.arange(C)[None]).reshape(-1) * self.bits
		return ((self.data[i >> 3] >> (i & 7).astype(np.uint8)) & ((1 << self.bits) - 1)).astype(smallestUint(2 ** self.bits - 1)).reshape(shape)

	def state(self, prefix):
		"""arrays for np.savez"""
		return {prefix + "shape": np.array(self.shape), prefix + "dtype": np.array(self.dtype.str), prefix + "bits": np.array(-1 if self.bits is None else self.bits),
			prefix + "const": np.array([] if self.const is None else [self.const], dtype = self.dtype), prefix + "data": np.zeros(0) if self.data is None else self.data}

	@staticmethod
	def fromState(z, prefix):
		a = PackedArray.__new__(PackedArray)
		a.shape, a.dtype = tuple(int(s) for s in z[prefix + "shape"]), np.dtype(str(z[prefix + "dtype"]))
		a.bits = None if int(z[prefix + "bits"]) < 0 else int(z[prefix + "bits"])
		a.const = z[prefix + "const"][0] if len(z[prefix + "const"]) else None
		a.data = None if a.const is not None else z[prefix + "data"]
		return a

class SparseImages():
	"""samples with their own active sites, stored compactly. Build with fromLists or fromDense; get a SparseTensor
	with batch(ids, device); save / load to .npz."""

	def __init__(self, coords, values, offsets, spatialShape):
		self.coords, self.values = coords, values          # list of D arrays [n], PackedArray [n] or [n, C]
		self.offsets, self.spatialShape = offsets, tuple(int(s) for s in spatialShape)

	@staticmethod
	def fromLists(coords, values, spatialShape):
		"""coords: one int array [n_i, D] per sample (unique sites); values: one array [n_i] or [n_i, C] per sample, or
		None (every active site has value 1, e.g. the presence of a mutation)"""
		D = len(spatialShape)
		n = np.array([len(c) for c in coords])
		offsets = np.concatenate([[0], np.cumsum(n)]).astype(np.int64)
		allc = np.concatenate([np.asarray(c, dtype = np.int64).reshape(-1, D) for c in coords]) if len(coords) else np.zeros((0, D), dtype = np.int64)
		assert ((allc >= 0) & (allc < np.array(spatialShape))).all(), "coordinates outside the grid"
		key = np.repeat(np.arange(len(coords)), n)
		for i in range(D):
			key = key * spatialShape[i] + allc[:, i]
		assert len(np.unique(key)) == len(key), "a site appears twice in a sample"
		cs = [allc[:, i].astype(smallestUint(spatialShape[i] - 1)) for i in range(D)]
		vals = np.ones(len(allc), dtype = np.uint8) if values is None else np.concatenate([np.asarray(v) for v in values])
		return SparseImages(cs, PackedArray(vals), offsets, spatialShape)

	@staticmethod
	def fromDense(images):
		"""numpy [N, S_1, .., S_D] -> the non-zero pixels of every image"""
		images = np.asarray(images)
		coords, values = [], []
		for img in images:
			c = np.argwhere(img != 0)
			coords.append(c); values.append(img[tuple(c.T)])
		return SparseImages.fromLists(coords, values, images.shape[1:])

	def __len__(self):
		return len(self.offsets) - 1

	@property
	def nbytes(self):
		return sum(c.nbytes for c in self.coords) + self.values.nbytes + self.offsets.nbytes

	def rows(self, ids):
		ids = np.asarray(ids, dtype = np.int64)
		lens = self.offsets[ids + 1] - self.offsets[ids]
		start = np.repeat(self.offsets[ids] - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
		return start + np.arange(lens.sum()), np.repeat(np.arange(len(ids)), lens)

	def sample(self, i):
		"""coordinates [n_i, D] int64 and values of sample i"""
		r = np.arange(self.offsets[i], self.offsets[i + 1])
		return np.stack([c[r].astype(np.int64) for c in self.coords], 1), self.values[r]

	def batch(self, ids, device = "cpu", dtype = t.float32):
		"""SparseTensor of the samples ids (sample b of the batch = ids[b]); features [n, C] (C = 1 for 1-D values)"""
		r, b = self.rows(ids)
		idx = np.stack([b] + [c[r].astype(np.int64) for c in self.coords], 1)
		v = self.values[r]
		f = t.as_tensor(v.reshape(len(r), -1), device = device).to(dtype)
		return SparseTensor(f, t.as_tensor(idx, device = device), self.spatialShape, len(ids))

	def save(self, path):
		z = {"offsets": self.offsets, "spatialShape": np.array(self.spatialShape)}
		z.update({"coords%d" % i: c for i, c in enumerate(self.coords)})
		z.update(self.values.state("values_"))
		np.savez(path, **z)

	@staticmethod
	def load(path):
		z = np.load(path)
		shape = tuple(int(s) for s in z["spatialShape"])
		return SparseImages([z["coords%d" % i] for i in range(len(shape))], PackedArray.fromState(z, "values_"), z["offsets"], shape)
