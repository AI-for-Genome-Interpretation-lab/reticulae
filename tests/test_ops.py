"""Tests of dilation, transposed and inverse convolutions, pooling and storage against dense PyTorch:
python -m pytest -q tests     (DEVICE=cuda:N for a GPU; the spconv tests are skipped if spconv is not importable)

	dilation      SparseConv / SubMConv with d > 1  == conv1d/2d/3d(dilation = d) of the zero-filled image
	transposed    SparseConvTranspose               == conv_transpose1d/2d/3d (stride, padding, dilation) of the
	              zero-filled image, at the output sites, which must be exactly the sites receiving from an active input
	inverse       SparseInverseConv                 == unpadded conv_transposeNd of the output of the forward
	              convolution, window [p, p + S) (the input grid), at the input sites of the forward convolution
	max pooling   SparseMaxPool                     == max_poolNd of the image filled with -inf at inactive sites
	avg pooling   SparseAvgPool                     == (sum of the window) / (number of active sites in the window)
	globalPool    per-sample max / mean / sum
	fixed sites   every Fixed* layer == the SparseTensor layer when all samples have the same sites
	storage       PackedArray / SparseImages: exact round trips, smallest dtypes, bit packing, save / load
with gradients (input, weight, bias) against the dense operations.
"""
import math, os, sys, zlib, subprocess
import numpy as np
import torch as t
import torch.nn.functional as F
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import reticulae as R
from test_reticulae import DEVICE, CONV, SHAPES, randomTensor, denseWeight, at, tensorFrom, checkRegular, checkSubm

CONVT = {1: F.conv_transpose1d, 2: F.conv_transpose2d, 3: F.conv_transpose3d}
MAXP = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}

def seed(*a):
	return zlib.crc32(str(a).encode())

def transWeight(w, k):
	"""[prod(k), cin, cout] -> conv_transposeNd weight [cin, cout, *k]"""
	D = len(k)
	return w.reshape(*k, w.shape[1], w.shape[2]).permute(D, D + 1, *range(D))

def indicator(x):
	return R.SparseTensor(t.ones(len(x.indices), 1, dtype = t.float64, device = DEVICE), x.indices, x.spatialShape, x.batchSize).dense()

# ---------------------------------------------------------------- dilation

@pytest.mark.parametrize("D,k,s,p,d", [(1, 3, 1, 0, 2), (1, 3, 2, 2, 3), (2, 3, 1, 2, 2), (2, 3, 2, 0, (1, 3)), (2, 2, 1, 1, 4), (3, 3, 1, 2, 2), (3, 2, 2, 1, 2)])
@pytest.mark.parametrize("density", [0.05, 0.5])
def test_dilated_regular_vs_dense(D, k, s, p, d, density):
	x = randomTensor(D, SHAPES[D], 3, density, 3, seed(D, k, s, p, d, density))
	conv = R.SparseConv(D, 3, 4, k, s, p, d).double().to(DEVICE)
	conv.bias.data.normal_()
	y = conv(x)
	ref = CONV[D](x.dense(), denseWeight(conv.weight, conv.k), conv.bias, stride = conv.s, padding = conv.p, dilation = conv.d)
	reach = CONV[D](indicator(x), t.ones((1, 1) + conv.k, dtype = t.float64, device = DEVICE), stride = conv.s, padding = conv.p, dilation = conv.d)[:, 0] > 0
	assert tuple(ref.shape[2:]) == y.spatialShape
	assert sorted(map(tuple, y.indices.tolist())) == sorted(map(tuple, t.nonzero(reach).tolist()))
	assert t.allclose(y.features, at(ref, y.indices), atol = 1e-10)

@pytest.mark.parametrize("D,k,d", [(1, 3, 2), (1, 5, 3), (2, 3, 2), (2, 3, (1, 3)), (3, 3, 2)])
@pytest.mark.parametrize("density", [0.05, 0.5])
def test_dilated_subm_vs_dense(D, k, d, density):
	x = randomTensor(D, SHAPES[D], 3, density, 3, seed(D, k, d, density))
	conv = R.SubMConv(D, 3, 4, k, d).double().to(DEVICE)
	conv.bias.data.normal_()
	y = conv(x)
	ref = CONV[D](x.dense(), denseWeight(conv.weight, conv.k), conv.bias, padding = tuple(conv.d[i] * (conv.k[i] // 2) for i in range(D)), dilation = conv.d)
	assert t.equal(y.indices, x.indices)
	assert t.allclose(y.features, at(ref, x.indices), atol = 1e-10)

# ---------------------------------------------------------------- transposed

TRANS = [(1, 3, 2, 0, 1), (1, 4, 3, 1, 2), (1, 2, 4, 0, 1), (2, 3, 2, 1, 1), (2, 2, 2, 0, 1), (2, 3, 1, 1, 2), (2, (3, 2), (2, 3), (1, 0), (1, 2)), (3, 3, 2, 1, 1), (3, 2, 2, 0, 2)]

@pytest.mark.parametrize("D,k,s,p,d", TRANS)
@pytest.mark.parametrize("density", [0.05, 0.5])
def test_transposed_vs_dense(D, k, s, p, d, density):
	shape = tuple(max(3, a // 2) for a in SHAPES[D])
	x = randomTensor(D, shape, 3, density, 3, seed(D, k, s, p, d, density))
	conv = R.SparseConvTranspose(D, 3, 4, k, s, p, d).double().to(DEVICE)
	conv.bias.data.normal_()
	x.features.requires_grad_()
	y = conv(x)
	img = x.dense().detach().requires_grad_()
	w = conv.weight.detach().clone().requires_grad_()
	b = conv.bias.detach().clone().requires_grad_()
	ref = CONVT[D](img, transWeight(w, conv.k), b, stride = conv.s, padding = conv.p, dilation = conv.d)
	reach = CONVT[D](indicator(x), t.ones((1, 1) + conv.k, dtype = t.float64, device = DEVICE), stride = conv.s, padding = conv.p, dilation = conv.d)[:, 0] > 0
	assert tuple(ref.shape[2:]) == y.spatialShape
	assert sorted(map(tuple, y.indices.tolist())) == sorted(map(tuple, t.nonzero(reach).tolist()))
	assert t.allclose(y.features, at(ref, y.indices), atol = 1e-10)
	gy = t.randn_like(y.features)
	y.features.backward(gy)
	at(ref, y.indices).backward(gy)
	assert t.allclose(x.features.grad, at(img.grad, x.indices), atol = 1e-10)
	assert t.allclose(conv.weight.grad, w.grad, atol = 1e-9) and t.allclose(conv.bias.grad, b.grad, atol = 1e-9)

# ---------------------------------------------------------------- inverse

@pytest.mark.parametrize("D,k,s,p,d", [(1, 3, 2, 0, 1), (1, 4, 3, 1, 2), (2, 3, 2, 0, 1), (2, 2, 2, 0, 1), (2, 3, 2, 1, 2), (2, 2, 3, 0, 1), (3, 3, 2, 1, 1), (3, 2, 2, 0, 1)])
def test_inverse_vs_dense(D, k, s, p, d):
	x = randomTensor(D, SHAPES[D], 3, 0.3, 3, seed(D, k, s, p, d))
	down = R.SparseConv(D, 3, 5, k, s, p, d, key = "down").double().to(DEVICE)
	mid = R.SubMConv(D, 5, 5, 3).double().to(DEVICE)
	up = R.SparseInverseConv(D, 5, 4, k, "down").double().to(DEVICE)
	up.bias.data.normal_()
	h = mid(down(x))
	h.features = h.features.detach().requires_grad_()
	y = up(h)
	assert t.equal(y.indices, x.indices) and y.spatialShape == x.spatialShape
	img = h.dense().detach().requires_grad_()
	w = up.weight.detach().clone().requires_grad_()
	b = up.bias.detach().clone().requires_grad_()
	ref = CONVT[D](img, transWeight(w, up.k), stride = down.s, dilation = down.d)                          # unpadded: input x + p
	ref = ref[(slice(None), slice(None)) + tuple(slice(p, p + S) for p, S in zip(down.p, x.spatialShape))]   # input grid
	ref = F.pad(ref, [a for S, T in zip(reversed(x.spatialShape), reversed(ref.shape[2:])) for a in (0, S - T)])   # zero-pad to it
	ref = ref + b.reshape((1, -1) + (1,) * D)
	assert t.allclose(y.features, at(ref, x.indices), atol = 1e-10)
	gy = t.randn_like(y.features)
	y.features.backward(gy)
	at(ref, x.indices).backward(gy)
	assert t.allclose(h.features.grad, at(img.grad, h.indices), atol = 1e-10)
	assert t.allclose(up.weight.grad, w.grad, atol = 1e-9) and t.allclose(up.bias.grad, b.grad, atol = 1e-9)

def test_inverse_rejects_other_sites():
	x = randomTensor(2, SHAPES[2], 2, 0.3, 3, 9)
	z = R.SparseConv(2, 3, 4, 3, 2, key = "a").double().to(DEVICE)(x)
	z = R.SparseConv(2, 4, 4, 3, 1, 2, 2).double().to(DEVICE)(z)            # dilated regular: different sites
	with pytest.raises(AssertionError):
		R.SparseInverseConv(2, 4, 3, 3, "a").double().to(DEVICE)(z)

# ---------------------------------------------------------------- pooling

POOL = [(1, 2, 2, 0, 1), (1, 3, 2, 1, 1), (1, 3, 1, 0, 2), (2, 2, 2, 0, 1), (2, 3, 2, 1, 1), (2, 3, 3, 0, 2), (2, (3, 2), (2, 1), (1, 0), 1), (3, 2, 2, 0, 1), (3, 3, 2, 1, 1)]

@pytest.mark.parametrize("D,k,s,p,d", POOL)
@pytest.mark.parametrize("density", [0.05, 0.5])
def test_maxpool_vs_dense(D, k, s, p, d, density):
	x = randomTensor(D, SHAPES[D], 3, density, 3, seed(D, k, s, p, d, density))
	pool = R.SparseMaxPool(D, k, s, p, d)
	x.features.requires_grad_()
	y = pool(x)
	img = t.where(indicator(x) > 0, x.dense(), t.tensor(-math.inf, dtype = t.float64, device = DEVICE)).detach().requires_grad_()
	ref = MAXP[D](img, pool.k, pool.s, pool.p, pool.d)
	assert tuple(ref.shape[2:]) == y.spatialShape
	assert sorted(map(tuple, y.indices.tolist())) == sorted(map(tuple, t.nonzero((ref > -math.inf).all(1)).tolist()))
	assert t.allclose(y.features, at(ref, y.indices), atol = 1e-12)
	gy = t.randn_like(y.features)
	y.features.backward(gy)
	at(ref, y.indices).backward(gy)
	assert t.allclose(x.features.grad, at(img.grad, x.indices), atol = 1e-12)

@pytest.mark.parametrize("D,k,s,p,d", POOL)
@pytest.mark.parametrize("density", [0.05, 0.5])
def test_avgpool_vs_dense(D, k, s, p, d, density):
	x = randomTensor(D, SHAPES[D], 3, density, 3, seed(D, k, s, p, d, density))
	pool = R.SparseAvgPool(D, k, s, p, d)
	x.features.requires_grad_()
	y = pool(x)
	img = x.dense().detach().requires_grad_()
	ones = t.ones((1, 1) + pool.k, dtype = t.float64, device = DEVICE)
	B, C = img.shape[:2]
	tot = CONV[D](img.reshape((B * C, 1) + img.shape[2:]), ones, stride = pool.s, padding = pool.p, dilation = pool.d)
	cnt = CONV[D](indicator(x), ones, stride = pool.s, padding = pool.p, dilation = pool.d)
	ref = tot.reshape((B, C) + tot.shape[2:]) / cnt.clamp(min = 1)
	assert sorted(map(tuple, y.indices.tolist())) == sorted(map(tuple, t.nonzero(cnt[:, 0] > 0).tolist()))
	assert t.allclose(y.features, at(ref, y.indices), atol = 1e-12)
	gy = t.randn_like(y.features)
	y.features.backward(gy)
	at(ref, y.indices).backward(gy)
	assert t.allclose(x.features.grad, at(img.grad, x.indices), atol = 1e-12)

def test_global_pool():
	x = randomTensor(2, SHAPES[2], 4, 0.2, 3, 1)                         # sample 1 has no sites
	img, m = x.dense(), indicator(x)
	flat, mf = img.reshape(4, 3, -1), m.reshape(4, 1, -1)
	assert t.allclose(R.globalPool(x, "sum"), flat.sum(2))
	assert t.allclose(R.globalPool(x, "mean"), flat.sum(2) / mf.sum(2).clamp(min = 1))
	mx = t.where(mf > 0, flat, t.tensor(-math.inf, dtype = t.float64, device = DEVICE)).amax(2)
	assert t.allclose(R.globalPool(x, "max"), t.where(mx.isinf(), t.zeros_like(mx), mx))

# ---------------------------------------------------------------- fixed sites == SparseTensor

def fixedCase(D, seedv):
	rng = np.random.default_rng(seedv)
	shape = SHAPES[D]
	keys = rng.choice(math.prod(shape), max(2, math.prod(shape) // 5), replace = False)
	coords = np.stack(np.unravel_index(np.sort(keys), shape), 1)
	x = t.randn(len(coords), 3, 3, dtype = t.float64, device = DEVICE)
	xs = tensorFrom(coords, shape, 3, 3)
	xs.features = x.transpose(0, 1).reshape(-1, 3).clone()
	return coords, shape, x, xs

def sameAsTensor(y, ys, outCoords, B = 3):
	pos = {tuple(r): i for i, r in enumerate(ys.indices.tolist())}
	rows = t.as_tensor([[pos[(b,) + tuple(c)] for b in range(B)] for c in np.asarray(outCoords).tolist()], device = DEVICE)
	assert len(pos) == B * len(outCoords)
	assert t.allclose(y, ys.features[rows], atol = 1e-12)

@pytest.mark.parametrize("D", [1, 2, 3])
def test_fixed_layers_equal_tensor_layers(D):
	coords, shape, x, xs = fixedCase(D, D)
	cases = [(R.FixedSubMConv(coords, shape, 3, 4, 3, d = 2), R.SubMConv(D, 3, 4, 3, d = 2)),
		(R.FixedSparseConv(coords, shape, 3, 4, 3, 2, 1, 2), R.SparseConv(D, 3, 4, 3, 2, 1, 2)),
		(R.FixedSparseConvTranspose(coords, shape, 3, 4, 3, 2, 1), R.SparseConvTranspose(D, 3, 4, 3, 2, 1))]
	for fixed, layer in cases:
		fixed, layer = fixed.double().to(DEVICE), layer.double().to(DEVICE)
		layer.load_state_dict(fixed.state_dict())
		sameAsTensor(fixed(x), layer(xs), fixed.outCoords)
	for mode in ("max", "avg"):
		fp = R.FixedSparsePool(coords, shape, 2, mode = mode).to(DEVICE)
		sameAsTensor(fp(x), R.SparsePool(D, 2, mode = mode)(xs), fp.outCoords)
	down = R.FixedSparseConv(coords, shape, 3, 5, 3, 2).double().to(DEVICE)
	inv = R.FixedSparseInverseConv(down, 5, 4).double().to(DEVICE)
	downT, invT = R.SparseConv(D, 3, 5, 3, 2, key = "k").double().to(DEVICE), R.SparseInverseConv(D, 5, 4, 3, "k").double().to(DEVICE)
	downT.load_state_dict(down.state_dict()); invT.load_state_dict(inv.state_dict())
	assert np.array_equal(inv.outCoords, coords)
	sameAsTensor(inv(down(x)), invT(downT(xs)), coords)

# ---------------------------------------------------------------- storage

@pytest.mark.parametrize("values,bits,dtype", [(np.array([1, 2, 2, 1, 2, 1, 1]), 2, None), (np.array([0, 1, 1, 0, 1]), 1, None), (np.arange(16) % 13, 4, None),
	(np.array([0, 200, 255]), None, np.uint8), (np.array([-3, 100]), None, np.int8), (np.array([0, 70000]), None, np.uint32), (np.array([-40000, 3]), None, np.int32),
	(np.array([0.5, -1.25], dtype = np.float16), None, np.float16), (np.random.default_rng(0).integers(0, 3, (101, 3)), 2, None)])
def test_packed_array(values, bits, dtype):
	a = R.PackedArray(values)
	assert a.bits == bits and (dtype is None or a.data.dtype == dtype)
	rows = np.random.default_rng(1).permutation(len(values))
	assert np.array_equal(a[rows], values[rows]) and np.array_equal(a[np.arange(len(values))], values)
	if bits:
		assert a.nbytes == math.ceil(values.size * bits / 8)

def test_packed_constant():
	a = R.PackedArray(np.full(1000, 7))
	assert a.nbytes == 0 and np.array_equal(a[[3, 999]], [7, 7])

def test_sparse_images(tmp_path):
	rng = np.random.default_rng(0)
	shape = (300, 70000)
	coords = [np.unique(np.stack([rng.integers(0, 300, n), rng.integers(0, 70000, n)], 1), axis = 0) for n in (5, 0, 40, 1)]
	values = [rng.integers(1, 3, len(c)) for c in coords]
	im = R.SparseImages.fromLists(coords, values, shape)
	assert [c.dtype for c in im.coords] == [np.uint16, np.uint32] and im.values.bits == 2
	assert im.nbytes == sum(len(c) for c in coords) * 6 + math.ceil(sum(len(c) for c in coords) / 4) + 5 * 8
	for i in range(len(coords)):
		c, v = im.sample(i)
		assert np.array_equal(c, coords[i]) and np.array_equal(v, values[i])
	ids = [2, 0, 3, 1]
	x = im.batch(ids, DEVICE)
	assert x.batchSize == 4 and x.spatialShape == shape
	for b, i in enumerate(ids):
		m = x.indices[:, 0] == b
		assert np.array_equal(x.indices[m, 1:].cpu().numpy(), coords[i]) and np.array_equal(x.features[m, 0].cpu().numpy(), values[i])
	im.save(tmp_path / "im.npz")
	im2 = R.SparseImages.load(tmp_path / "im.npz")
	y = im2.batch(ids, DEVICE)
	assert t.equal(x.indices, y.indices) and t.equal(x.features, y.features)
	ones = R.SparseImages.fromLists(coords, None, shape)
	assert ones.values.nbytes == 0 and (ones.batch([0, 2]).features == 1).all()
	with pytest.raises(AssertionError):
		R.SparseImages.fromLists([np.array([[1, 1], [1, 1]])], None, shape)

def test_sparse_images_from_dense():
	rng = np.random.default_rng(0)
	imgs = rng.integers(0, 3, (5, 6, 7, 8)) * (rng.random((5, 6, 7, 8)) < 0.2)
	im = R.SparseImages.fromDense(imgs)
	x = im.batch(range(5), DEVICE)
	assert t.equal(x.dense()[:, 0].cpu(), t.as_tensor(imgs, dtype = t.float32))
	assert [c.dtype for c in im.coords] == [np.uint8] * 3 and im.values.bits == 2

def test_sparsetensor_from_dense():
	img = t.zeros(2, 3, 5, 6, dtype = t.float64)
	img[0, 1, 2, 3] = 1.5; img[1, :, 4, 0] = t.tensor([1., 0., -2.])
	x = R.SparseTensor.fromDense(img)
	assert x.indices.tolist() == [[0, 2, 3], [1, 4, 0]] and t.equal(x.dense(), img)

@pytest.mark.parametrize("doc", ["README.md", "docs/API.md"])
def test_documentation_code_runs(doc, tmp_path, monkeypatch):
	"""every ```python block of the documentation runs as written (in order, sharing names)"""
	import re
	src = open(os.path.join(os.path.dirname(__file__), "..", doc)).read()
	monkeypatch.chdir(tmp_path)
	g = {}
	exec("import torch as t, numpy as np, reticulae as R", g)
	for block in re.findall(r"```python\n(.*?)```", src, re.S):
		exec(block, g)

def test_quickstart_runs():
	r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "..", "examples", "quickstart.py")], capture_output = True, text = True)
	assert r.returncode == 0, r.stderr

# ---------------------------------------------------------------- spconv (optional)

def spconvOrSkip():
	try:
		import spconv.pytorch as spconv
	except ImportError:
		pytest.skip("spconv not installed")
	if DEVICE == "cpu":
		pytest.skip("spconv tests run on a GPU (DEVICE=cuda:N)")
	return spconv

def spWeight(w, k):
	D = len(k)
	return w.reshape(*k, w.shape[1], w.shape[2]).permute(D + 1, *range(D), D).contiguous()

def toSp(x):
	import spconv.pytorch as spconv
	return spconv.SparseConvTensor(x.features, x.indices.int(), list(x.spatialShape), x.batchSize)

def compareSp(y, ys):
	pos = {tuple(r): i for i, r in enumerate(ys.indices.long().tolist())}
	assert len(pos) == len(y.indices) and list(ys.spatial_shape) == list(y.spatialShape)
	rows = t.as_tensor([pos[tuple(r)] for r in y.indices.tolist()], device = DEVICE)
	assert t.allclose(y.features, ys.features[rows], atol = 1e-4)

@pytest.mark.parametrize("D", [2, 3])
def test_new_ops_vs_spconv(D):
	spconv = spconvOrSkip()
	x = randomTensor(D, SHAPES[D], 3, 0.2, 3, 5, t.float32)
	pairs = [(R.SparseConv(D, 3, 4, 3, 1, 2, 2), getattr(spconv, "SparseConv%dd" % D)(3, 4, 3, 1, 2, dilation = 2)),
		(R.SubMConv(D, 3, 4, 3, 2), getattr(spconv, "SubMConv%dd" % D)(3, 4, 3, dilation = 2)),
		(R.SparseConvTranspose(D, 3, 4, 3, 2, 0), getattr(spconv, "SparseConvTranspose%dd" % D)(3, 4, 3, 2, 0))]
	for ours, sp in pairs:
		ours, sp = ours.to(DEVICE), sp.to(DEVICE)
		ours.bias.data.normal_()
		sp.weight.data.copy_(spWeight(ours.weight.data, ours.k)); sp.bias.data.copy_(ours.bias.data)
		compareSp(ours(x), sp(toSp(x)))
	xr = x.replace(x.features.relu())        # spconv's max pool returns max(0, max of the active inputs): equal after a ReLU
	compareSp(R.SparseMaxPool(D, 3, 2, 1)(xr), getattr(spconv, "SparseMaxPool%dd" % D)(3, 2, 1)(toSp(xr)))
	down, up = R.SparseConv(D, 3, 4, 3, 2, key = "a").to(DEVICE), R.SparseInverseConv(D, 4, 2, 3, "a").to(DEVICE)
	spd, spu = getattr(spconv, "SparseConv%dd" % D)(3, 4, 3, 2, indice_key = "a").to(DEVICE), getattr(spconv, "SparseInverseConv%dd" % D)(4, 2, 3, indice_key = "a").to(DEVICE)
	for a, b in ((down, spd), (up, spu)):
		b.weight.data.copy_(spWeight(a.weight.data, a.k)); b.bias.data.copy_(a.bias.data)
	compareSp(up(down(x)), spu(spd(toSp(x))))

# ---------------------------------------------------------------- determinism

def twice(run):
	"""run() -> list of tensors, twice with the same seed: bitwise equal"""
	res = []
	for rep in range(2):
		t.manual_seed(1)
		res.append([a.detach().clone() for a in run()])
	return all(t.equal(a, b) for a, b in zip(*res))

def test_deterministic():
	"""DETERMINISTIC (default): two identical forward + backward passes give bitwise-identical outputs and gradients for
	every layer, on a fully active grid with a batch large enough for colliding additions (on the GPU, the atomic
	scatter-add of DETERMINISTIC = False typically fails this test)"""
	assert R.Convolutions.DETERMINISTIC
	S, B, C = 64, 16, 8
	coords = np.stack(np.meshgrid(np.arange(S), np.arange(S), indexing = "ij"), -1).reshape(-1, 2)
	t.manual_seed(0)
	xf = t.randn(len(coords), B, C, device = DEVICE)
	xs = tensorFrom(coords, (S, S), B, C); xs.features = xs.features.float()
	fixed = [lambda: R.FixedSparseConv(coords, (S, S), C, C, 3, 2), lambda: R.FixedSubMConv(coords, (S, S), C, C, 3),
		lambda: R.FixedSparseConvTranspose(coords, (S, S), C, C, 3, 2), lambda: R.FixedSparsePool(coords, (S, S), 3, 2, mode = "avg"),
		lambda: R.FixedSparsePool(coords, (S, S), 3, 2, mode = "max")]
	tensor = [lambda: R.SparseConv2d(C, C, 3, 2), lambda: R.SubMConv2d(C, C, 3), lambda: R.SparseConvTranspose2d(C, C, 3, 2),
		lambda: R.SparseAvgPool2d(3, 2), lambda: R.SparseMaxPool2d(3, 2),
		lambda: t.nn.Sequential(R.SparseConv2d(C, C, 3, 2, key = "d"), R.SparseInverseConv2d(C, C, 3, "d"))]
	def run(make, x0, sparse, pool = None):
		layer = make().to(DEVICE)
		x = x0.clone().requires_grad_()
		y = layer(xs.replace(x)) if sparse else layer(x)
		y = pool(y) if pool else (y.features if sparse else y)
		y.backward(t.cos(y.detach()))
		return [y, x.grad] + [p.grad for p in layer.parameters()]
	for make in fixed:
		assert twice(lambda: run(make, xf, False))
	for make in tensor:
		assert twice(lambda: run(make, xs.features, True))
	for mode in ("sum", "mean"):
		assert twice(lambda: run(lambda: R.SubMConv2d(C, C, 3), xs.features, True, lambda h: R.globalPool(h, mode)))
	xy, order = R.hilbertXY(["1"] * 400, np.repeat(np.arange(1, 201) * 1000, 2))   # pairs of loci sharing a pixel
	g = t.randint(0, 3, (B, 400), dtype = t.int8)
	def hc():
		net = R.SparseHilbertCNN(400, 2, xy, order).to(DEVICE); net.prepare(g.numpy())
		out = net(g.to(DEVICE))["pred"]
		out.pow(2).sum().backward()
		return [out] + [p.grad for p in net.parameters()]
	assert twice(hc)
