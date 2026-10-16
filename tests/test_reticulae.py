"""Correctness tests:  python -m pytest -q tests      (DEVICE=cuda:N to run on a GPU; the spconv tests are skipped
if spconv is not importable)

Every layer is checked against independent references:
	dense      torch conv1d/conv2d/conv3d of the zero-filled image (float64): regular convolutions at the active output
	           sites, which must be exactly the sites whose receptive field contains an active input; submanifold
	           convolutions = dense convolution with padding (k-1)/2 at the input sites
	brute      a pure-Python loop over the active sites, on grids far too large to be dense (2^40, 2^30 x 2^30, 2^20^3)
	reference  a per-offset loop implementation of the Hilbert network (tests/referenceHilbertCNN.py)
	spconv     spconv.pytorch SparseConvNd / SubMConvNd with the same weights (optional)
on random sites (different in every sample), anisotropic kernels/strides/padding, edge cases (one site, borders, fully
dense, one row, empty samples), with gradcheck/gradgradcheck, gradients of input/weight/bias against the dense
convolution (including the chunked weight-gradient path), FixedSparseConv == SparseTensor path, rulebook caching,
batch independence and CPU == GPU. The Hilbert network is checked against a dense masked re-implementation, against the
reference loop implementation (predictions and gradients) and on a small learning problem.
"""
import math, os, sys, zlib
import numpy as np
import torch as t
import torch.nn.functional as F
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from reticulae import SparseTensor, SparseConv, SubMConv, FixedSparseConv, FixedSubMConv, SparseConvFn, SparseHilbertCNN, d2xy, hilbertXY
from reticulae import Rulebook as RB
import referenceHilbertCNN as ZB

DEVICE = os.environ.get("DEVICE", "cpu")
CONV = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}
SHAPES = {1: (97,), 2: (29, 23), 3: (11, 9, 13)}
REGULAR = [(1, 3, 2, 0), (1, 8, 8, 0), (1, 5, 1, 2), (1, 2, 3, 0), (1, 4, 3, 1), (1, 1, 1, 0),
	(2, 3, 2, 0), (2, 8, 8, 0), (2, 3, 1, 1), (2, 2, 3, 0), (2, 5, 3, 2), (2, 1, 1, 0), (2, (3, 1), (2, 1), (1, 0)), (2, (2, 4), (3, 2), (0, 1)),
	(3, 3, 2, 0), (3, 2, 2, 0), (3, 3, 1, 1), (3, 1, 2, 0), (3, (3, 2, 1), (1, 2, 1), (1, 0, 0))]
SUBM = [(1, 1), (1, 3), (1, 7), (2, 1), (2, 3), (2, 5), (2, (3, 1)), (3, 3), (3, 5), (3, (1, 3, 5))]

def randomTensor(D, shape, B, density, C, seed, dtype = t.float64):
	"""random sites, different in every sample (sample 1 empty when B > 2)"""
	rng = np.random.default_rng(seed)
	idx = []
	for b in range(B):
		if B > 2 and b == 1:
			continue
		n = max(1, int(density * math.prod(shape)))
		keys = rng.choice(math.prod(shape), n, replace = False)
		c = np.stack(np.unravel_index(keys, shape), 1)
		idx.append(np.concatenate([np.full((n, 1), b), c], 1))
	idx = t.as_tensor(np.concatenate(idx), device = DEVICE)
	return SparseTensor(t.randn(len(idx), C, dtype = dtype, device = DEVICE), idx, shape, B)

def denseWeight(w, k):
	"""[prod(k), cin, cout] (row-major offsets) -> convNd weight [cout, cin, *k]"""
	D = len(k)
	return w.reshape(*k, w.shape[1], w.shape[2]).permute(D + 1, D, *range(D))

def at(img, idx):
	"""[B, C, *S] values at the sites idx [n, 1 + D] -> [n, C]"""
	return img.permute(0, *range(2, img.dim()), 1)[tuple(idx.t())]

def makeConv(D, cin, cout, k, s, p, subm):
	conv = (SubMConv(D, cin, cout, k) if subm else SparseConv(D, cin, cout, k, s, p)).double().to(DEVICE)
	conv.bias.data.normal_()
	return conv

def checkRegular(x, conv):
	y = conv(x)
	ref = CONV[conv.D](x.dense(), denseWeight(conv.weight, conv.k), conv.bias, stride = conv.s, padding = conv.p)
	ind = SparseTensor(t.ones(len(x.indices), 1, dtype = t.float64, device = DEVICE), x.indices, x.spatialShape, x.batchSize).dense()
	reach = CONV[conv.D](ind, t.ones((1, 1) + conv.k, dtype = t.float64, device = DEVICE), stride = conv.s, padding = conv.p)[:, 0] > 0
	assert tuple(ref.shape[2:]) == y.spatialShape
	assert sorted(map(tuple, y.indices.tolist())) == sorted(map(tuple, t.nonzero(reach).tolist()))
	assert t.allclose(y.features, at(ref, y.indices), atol = 1e-10)
	return y

def checkSubm(x, conv):
	y = conv(x)
	ref = CONV[conv.D](x.dense(), denseWeight(conv.weight, conv.k), conv.bias, padding = tuple(a // 2 for a in conv.k))
	assert t.equal(y.indices, x.indices) and y.spatialShape == x.spatialShape
	assert t.allclose(y.features, at(ref, x.indices), atol = 1e-10)
	return y

# ---------------------------------------------------------------- single layers vs dense convolutions

@pytest.mark.parametrize("D,k,s,p", REGULAR)
@pytest.mark.parametrize("density", [0.02, 0.2, 0.9])
def test_regular_vs_dense(D, k, s, p, density):
	x = randomTensor(D, SHAPES[D], 3, density, 3, zlib.crc32(str((D, k, s, p, density)).encode()))
	checkRegular(x, makeConv(D, 3, 4, k, s, p, False))

@pytest.mark.parametrize("D,k", SUBM)
@pytest.mark.parametrize("density", [0.02, 0.2, 0.9])
def test_subm_vs_dense(D, k, density):
	x = randomTensor(D, SHAPES[D], 3, density, 3, zlib.crc32(str((D, k, density)).encode()))
	checkSubm(x, makeConv(D, 3, 4, k, 1, 0, True))

def tensorFrom(coords, shape, B = 2, C = 2):
	coords = t.as_tensor(np.asarray(coords), device = DEVICE)
	idx = t.cat([t.arange(B, device = DEVICE).repeat_interleave(len(coords))[:, None], coords.repeat(B, 1)], 1)
	return SparseTensor(t.randn(len(idx), C, dtype = t.float64, device = DEVICE), idx, shape, B)

@pytest.mark.parametrize("k,s,p", [(3, 2, 0), (8, 8, 0), (2, 3, 0), (3, 1, 1), (5, 3, 2), (4, 4, 0)])
def test_edge_cases(k, s, p):
	S = 23
	cases = [[[11, 7]], [[0, 0], [0, S - 1], [S - 1, 0], [S - 1, S - 1], [S - 1, 5], [4, S - 2]],
		np.stack(np.meshgrid(np.arange(S), np.arange(S), indexing = "ij"), -1).reshape(-1, 2),
		np.stack([np.full(S, 6), np.arange(S)], 1), np.stack([np.arange(0, S, 2), np.arange(0, S, 2)], 1)]
	for c in cases:
		x = tensorFrom(c, (S, S))
		checkRegular(x, makeConv(2, 2, 3, k, s, p, False))
		if k % 2:
			checkSubm(x, makeConv(2, 2, 3, k, 1, 0, True))

# ---------------------------------------------------------------- huge grids vs brute force

def bruteRegular(idx, f, w, b, k, s, p, outShape):
	out = {}
	offs = RB.kernelOffsets(k, "cpu").tolist()
	for i, row in enumerate(idx.tolist()):
		for d, delta in enumerate(offs):
			o = [row[j + 1] + p[j] - delta[j] for j in range(len(k))]
			if any(a < 0 or a % s[j] or a // s[j] >= outShape[j] for j, a in enumerate(o)):
				continue
			key = (row[0],) + tuple(a // s[j] for j, a in enumerate(o))
			out[key] = out.get(key, b) + f[i] @ w[d]
	return out

def bruteSubm(idx, f, w, b, k):
	pos = {tuple(r): i for i, r in enumerate(idx.tolist())}
	offs = RB.kernelOffsets(k, "cpu").tolist()
	out = {}
	for o, row in enumerate(idx.tolist()):
		acc = b.copy()
		for d, delta in enumerate(offs):
			nb = (row[0],) + tuple(row[j + 1] + delta[j] - k[j] // 2 for j in range(len(k)))
			if nb in pos:
				acc = acc + f[pos[nb]] @ w[d]
		out[tuple(row)] = acc
	return out

@pytest.mark.parametrize("shape,k,s,p", [((2 ** 40,), 8, 8, 0), ((2 ** 40,), 5, 3, 2), ((2 ** 30, 2 ** 30), 3, 2, 0), ((2 ** 30, 2 ** 30), 8, 8, 0), ((2 ** 17, 2 ** 17), 3, 2, 1), ((2 ** 20,) * 3, 3, 2, 0), ((2 ** 20,) * 3, 2, 2, 0)])
def test_regular_vs_bruteforce_huge_grid(shape, k, s, p):
	D = len(shape)
	rng = np.random.default_rng(D + k)
	c = rng.integers(0, shape[0], (300, D))
	c = np.concatenate([c, c[:100] + rng.integers(-3, 4, (100, D))])          # clustered sites share outputs
	c = np.unique(np.clip(c, 0, shape[0] - 1), axis = 0)
	idx = t.as_tensor(np.concatenate([rng.integers(0, 2, (len(c), 1)), c], 1))
	idx = t.unique(idx, dim = 0)
	x = SparseTensor(t.randn(len(idx), 2, dtype = t.float64), idx, shape, 2)
	conv = SparseConv(D, 2, 3, k, s, p).double()
	conv.bias.data.normal_()
	y = conv(x)
	ref = bruteRegular(idx, x.features.numpy(), conv.weight.detach().numpy(), conv.bias.detach().numpy(), conv.k, conv.s, conv.p, y.spatialShape)
	assert sorted(ref) == sorted(map(tuple, y.indices.tolist()))
	got = {tuple(r): y.features[i].detach().numpy() for i, r in enumerate(y.indices.tolist())}
	assert max(np.abs(got[o] - ref[o]).max() for o in ref) < 1e-10

@pytest.mark.parametrize("shape,k", [((2 ** 40,), 5), ((2 ** 30, 2 ** 30), 3), ((2 ** 20,) * 3, 3)])
def test_subm_vs_bruteforce_huge_grid(shape, k):
	D = len(shape)
	rng = np.random.default_rng(D)
	c = rng.integers(0, shape[0], (200, D))
	c = np.unique(np.clip(np.concatenate([c, c + rng.integers(-1, 2, (200, D)), c + rng.integers(-1, 2, (200, D))]), 0, shape[0] - 1), axis = 0)
	idx = t.as_tensor(np.concatenate([np.zeros((len(c), 1), dtype = np.int64), c], 1))
	x = SparseTensor(t.randn(len(idx), 2, dtype = t.float64), idx, shape, 1)
	conv = SubMConv(D, 2, 3, k).double()
	conv.bias.data.normal_()
	y = conv(x)
	ref = bruteSubm(idx, x.features.numpy(), conv.weight.detach().numpy(), conv.bias.detach().numpy(), conv.k)
	assert max(np.abs(y.features[i].detach().numpy() - ref[tuple(r)]).max() for i, r in enumerate(idx.tolist())) < 1e-10

# ---------------------------------------------------------------- gradients

@pytest.mark.parametrize("D,k,s,subm", [(1, 3, 2, False), (2, 8, 8, False), (2, 3, 1, True), (3, 3, 2, False), (3, 3, 1, True)])
def test_gradcheck(D, k, s, subm):
	x = randomTensor(D, tuple(min(a, 9) for a in SHAPES[D]), 2, 0.3, 2, 7)
	conv = makeConv(D, 2, 3, k, s, 0, subm)
	outIdx, _, _, tabs = conv.rulebook(x.indices, x.spatialShape, x.batchSize)
	f = lambda a, w: SparseConvFn.apply(a, w, tabs[0], tabs[1], tabs[2], len(outIdx), tabs[3])
	a = x.features[:, None].clone().requires_grad_()
	w = conv.weight.detach().clone().requires_grad_()
	assert t.autograd.gradcheck(f, (a, w))
	assert t.autograd.gradgradcheck(f, (a, w))

@pytest.mark.parametrize("D,k,s,p,subm", [(1, 5, 2, 1, False), (2, 3, 2, 0, False), (2, 8, 8, 0, False), (2, 3, 1, 0, True), (3, 3, 2, 1, False), (3, 3, 1, 0, True)])
def test_gradients_vs_dense(D, k, s, p, subm):
	x = randomTensor(D, SHAPES[D], 3, 0.3, 3, 11)
	conv = makeConv(D, 3, 4, k, s, p, subm)
	x.features.requires_grad_()
	y = conv(x)
	gy = t.randn_like(y.features)
	y.features.backward(gy)
	img = x.dense().detach().requires_grad_()
	w = conv.weight.detach().clone().requires_grad_()
	b = conv.bias.detach().clone().requires_grad_()
	ref = CONV[D](img, denseWeight(w, conv.k), b, stride = conv.s, padding = tuple(a // 2 for a in conv.k) if subm else conv.p)
	at(ref, y.indices).backward(gy)
	assert t.allclose(x.features.grad, at(img.grad, x.indices), atol = 1e-10)
	assert t.allclose(conv.weight.grad, w.grad, atol = 1e-9)
	assert t.allclose(conv.bias.grad, b.grad, atol = 1e-9)

@pytest.mark.parametrize("subm", [False, True])
def test_chunked_weight_gradient(subm):
	"""fully active 64 x 64 grid, batch 8: M * B >= 4096 triggers the chunked (padded) weight-gradient reduction"""
	S = 64
	coords = np.stack(np.meshgrid(np.arange(S), np.arange(S), indexing = "ij"), -1).reshape(-1, 2)
	conv = (FixedSubMConv(coords, (S, S), 2, 3, 3) if subm else FixedSparseConv(coords, (S, S), 2, 3, 3, 2)).double().to(DEVICE)
	R, M = conv.inTab.shape
	assert (M * 8) // 2048 > 1
	x = t.randn(len(coords), 8, 2, dtype = t.float64, device = DEVICE)
	conv(x).pow(2).sum().backward()
	w = conv.weight.detach().clone().requires_grad_()
	img = x.permute(1, 2, 0).reshape(8, 2, S, S)
	ref = F.conv2d(img, denseWeight(w, (3, 3)), conv.bias.detach(), stride = conv.s, padding = 1 if subm else 0)
	oc = t.as_tensor(conv.outCoords, device = DEVICE)
	ref[:, :, oc[:, 0], oc[:, 1]].pow(2).sum().backward()
	assert t.allclose(conv.weight.grad, w.grad, atol = 1e-8)

@pytest.mark.parametrize("D,k,s,subm", [(2, 3, 2, False), (3, 3, 1, True), (2, 8, 8, False)])
def test_offset_chunks(D, k, s, subm, monkeypatch):
	"""TEMP_BYTES = 1 (one kernel offset per chunk, forward and backward) == a single chunk"""
	import reticulae.Convolutions as SC
	x = randomTensor(D, SHAPES[D], 3, 0.3, 3, 4)
	conv = makeConv(D, 3, 4, k, s, 0, subm)
	res = []
	for budget in (2 ** 40, 1):
		monkeypatch.setattr(SC, "TEMP_BYTES", budget)
		conv.zero_grad()
		f = x.features.clone().requires_grad_()
		y = conv(x.replace(f)).features
		y.backward(t.ones_like(y) + y.detach())
		res.append((y.detach(), f.grad, conv.weight.grad.clone(), conv.bias.grad.clone()))
	for a, b in zip(*res):
		assert t.allclose(a, b, atol = 1e-12)

# ---------------------------------------------------------------- fixed sites, caching, batch, device

@pytest.mark.parametrize("D,k,s,subm", [(1, 3, 2, False), (2, 8, 8, False), (2, 3, 1, True), (3, 3, 2, False), (3, 3, 1, True)])
def test_fixed_equals_sparsetensor(D, k, s, subm):
	rng = np.random.default_rng(D)
	shape = SHAPES[D]
	keys = rng.choice(math.prod(shape), max(2, math.prod(shape) // 5), replace = False)
	coords = np.stack(np.unravel_index(np.sort(keys), shape), 1)
	B = 3
	fixed = (FixedSubMConv(coords, shape, 3, 4, k) if subm else FixedSparseConv(coords, shape, 3, 4, k, s)).double().to(DEVICE)
	conv = makeConv(D, 3, 4, k, s, 0, subm)
	conv.load_state_dict(fixed.state_dict())
	x = t.randn(len(coords), B, 3, dtype = t.float64, device = DEVICE, requires_grad = True)
	xs = tensorFrom(coords, shape, B, 3)
	xs.features = x.detach().transpose(0, 1).reshape(-1, 3).clone().requires_grad_()   # sample-major rows
	y = fixed(x)
	ys = conv(xs)
	pos = {tuple(r): i for i, r in enumerate(ys.indices.tolist())}
	rows = t.as_tensor([[pos[(b,) + tuple(c)] for b in range(B)] for c in fixed.outCoords.tolist()], device = DEVICE)
	assert t.allclose(y, ys.features[rows], atol = 1e-12)
	gy = t.randn_like(y)
	y.backward(gy)
	ys.features.backward(t.zeros_like(ys.features).index_put_((rows.reshape(-1),), gy.reshape(-1, 4)))
	assert t.allclose(x.grad.transpose(0, 1).reshape(-1, 3), xs.features.grad, atol = 1e-12)
	assert t.allclose(fixed.weight.grad, conv.weight.grad, atol = 1e-10)

def test_subm_cache():
	x = randomTensor(3, SHAPES[3], 2, 0.2, 3, 5)
	a, b = makeConv(3, 3, 4, 3, 1, 0, True), makeConv(3, 4, 4, 3, 1, 0, True)
	y = b(x.__class__(a(x).features, x.indices, x.spatialShape, x.batchSize))        # no cache
	z = b(a(x))                                                                       # rulebook reused
	assert ("subm", (3, 3, 3), (1, 1, 1)) in x.cache and t.allclose(y.features, z.features)

def test_batch_independence_and_device():
	coords = np.stack(np.unravel_index(np.random.default_rng(3).choice(64 * 64, 500, replace = False), (64, 64)), 1)
	conv = FixedSparseConv(coords, (64, 64), 3, 8, 3, 2).to(DEVICE)
	x = t.randn(len(coords), 6, 3, device = DEVICE)
	y = conv(x)
	perm = t.randperm(6)
	assert t.allclose(conv(x[:, perm]), y[:, perm], atol = 1e-5)
	assert t.allclose(conv(x[:, 2:3]), y[:, 2:3], atol = 1e-5)
	if DEVICE != "cpu":
		assert t.allclose(conv.cpu()(x.cpu()), y.cpu(), atol = 1e-4)
		xs = randomTensor(3, SHAPES[3], 3, 0.3, 3, 1, t.float32)
		sm = SubMConv(3, 3, 4, 3).to(DEVICE)
		ys = sm(xs)
		xc = SparseTensor(xs.features.cpu(), xs.indices.cpu(), xs.spatialShape, xs.batchSize)
		assert t.allclose(sm.cpu()(xc).features, ys.features.cpu(), atol = 1e-4)

def test_autocast():
	if DEVICE == "cpu":
		pytest.skip("autocast test on a GPU")
	x = randomTensor(3, SHAPES[3], 2, 0.3, 16, 2, t.float32)
	for conv in (SubMConv(3, 16, 32, 3).to(DEVICE), SparseConv(3, 16, 32, 3, 2).to(DEVICE)):
		x.features.requires_grad_()
		y = conv(x).features
		y.pow(2).sum().backward()
		g32, w32 = x.features.grad.clone(), conv.weight.grad.clone()
		x.features.grad = None; conv.weight.grad = None
		with t.autocast("cuda", dtype = t.bfloat16):
			yh = conv(x).features
		assert yh.dtype == t.bfloat16
		yh.float().pow(2).sum().backward()
		assert x.features.grad.dtype == t.float32 and conv.weight.grad.dtype == t.float32
		rel = lambda a, b: ((a - b).norm() / b.norm()).item()
		assert rel(yh.float(), y) < 2e-2 and rel(x.features.grad, g32) < 5e-2 and rel(conv.weight.grad, w32) < 5e-2
		x.features = x.features.detach()

# ---------------------------------------------------------------- Hilbert network

def randomGenome(order, m, B, seed, shared = 20):
	"""m loci on a 2^order curve (the last `shared` on the pixel of another locus), dosages with missing values"""
	rng = np.random.default_rng(seed)
	g = np.sort(rng.choice(1 << (2 * order), m - shared, replace = False))
	g = np.concatenate([g, g[rng.choice(len(g), shared, replace = False)]])
	xy = np.stack(d2xy(order, g), 1)
	X = rng.integers(0, 3, (B, m)).astype(np.int8)
	X[rng.random((B, m)) < 0.05] = -127
	return xy, X

def perturb(net):
	for p in net.parameters():   # non-trivial LayerNorm affine and biases
		p.data.add_(0.1 * t.randn_like(p))

def test_network_vs_dense():
	t.manual_seed(0)
	order, m, B = 9, 400, 4
	xy, X = randomGenome(order, m, B, 0)
	net = SparseHilbertCNN(m, 3, xy, order, dropout = 0.0).double().to(DEVICE).eval()
	perturb(net)
	net.prepare(X)
	out = net(t.as_tensor(X, device = DEVICE))["pred"]
	size = 1 << order
	keys = np.unique(xy[:, 0] * size + xy[:, 1])
	G = np.where(X == -127, net.mean.cpu().numpy(), X.astype(np.float64))
	pix = np.searchsorted(keys, xy[:, 0] * size + xy[:, 1])
	f = np.stack([np.bincount(pix, G[b], len(keys)) / np.bincount(pix, minlength = len(keys)) for b in range(B)], 1)   # loci sharing a pixel -> mean
	h = t.zeros(B, 1, size, size, dtype = t.float64, device = DEVICE)
	h[:, 0, keys // size, keys % size] = t.as_tensor(f, device = DEVICE).t()
	mask = t.zeros_like(h)
	mask[:, :, keys // size, keys % size] = 1
	for conv, norm in zip(net.convs, net.norms):
		h = F.conv2d(h, denseWeight(conv.weight, conv.k), conv.bias, stride = conv.s)
		mask = (F.conv2d(mask, t.ones((1, 1) + conv.k, dtype = t.float64, device = DEVICE), stride = conv.s) > 0).double()
		h = F.leaky_relu(norm(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)) * mask
	ref = net.head(h.permute(0, 2, 3, 1).reshape(B, -1))
	assert len(net.convs) >= 3
	assert t.allclose(out, ref, atol = 1e-9)

@pytest.mark.parametrize("order", [12, 17])
def test_network_vs_reference(order):
	t.manual_seed(0)
	m, B = 3000, 6
	xy, X = randomGenome(order, m, B, order)
	net = SparseHilbertCNN(m, 4, xy, order, dropout = 0.0).double().to(DEVICE).eval()
	old = ZB.SparseHilbertCNN(m, 4, xy, order, dropout = 0.0).double().to(DEVICE).eval()
	perturb(net)
	res = old.load_state_dict(net.state_dict(), strict = False)
	assert all(k.endswith((".ins", ".outIdx")) for k in res.missing_keys) and not res.unexpected_keys
	net.prepare(X); old.prepare(X)
	X = t.as_tensor(X, device = DEVICE)
	y, yo = net(X)["pred"], old(X)["pred"]
	assert t.allclose(y, yo, atol = 1e-10)
	gy = t.randn_like(y)
	y.backward(gy); yo.backward(gy)
	go = dict(old.named_parameters())
	for n, p in net.named_parameters():
		assert t.allclose(p.grad, go[n].grad, atol = 1e-9), n

def test_learns_local_pattern():
	"""the label is the dosage sum of 30 neighbouring loci (of 600): 2,400 training samples, test r must exceed 0.85"""
	t.manual_seed(0)
	order, m, n, ntr = 10, 600, 3000, 2400
	rng = np.random.default_rng(1)
	xy = np.stack(d2xy(order, np.sort(rng.choice(1 << (2 * order), m, replace = False))), 1)
	X = rng.integers(0, 3, (n, m)).astype(np.int8)
	y = X[:, 200:230].sum(1, keepdims = True).astype(np.float32)
	y = (y - y.mean()) / y.std()
	net = SparseHilbertCNN(m, 1, xy, order, dropout = 0.0).to(DEVICE)
	net.prepare(X[:ntr])
	Xt, yt = t.as_tensor(X, device = DEVICE), t.as_tensor(y, device = DEVICE)
	opt = t.optim.Adam(net.parameters(), lr = 1e-3)
	for e in range(60):
		for i in range(0, ntr, 50):
			opt.zero_grad()
			F.mse_loss(net(Xt[i:i + 50])["pred"], yt[i:i + 50]).backward()
			opt.step()
	net.eval()
	with t.no_grad():
		r = np.corrcoef(net(Xt[ntr:])["pred"][:, 0].cpu().numpy(), y[ntr:, 0])[0, 1]
	assert r > 0.85, r

def test_hilbert():
	order = 5
	x, y = d2xy(order, np.arange(1 << (2 * order)))
	assert len(set(zip(x, y))) == 1 << (2 * order)
	assert np.all(np.abs(np.diff(x)) + np.abs(np.diff(y)) == 1)
	xy, order = hilbertXY(["1", "1", "2", "2"], [1, 100, 1, 50])
	assert order == 4                                     # genome = 101 + 51 bp <= 2^4 x 2^4 pixels
	assert np.array_equal(xy, np.stack(d2xy(order, np.array([0, 99, 101, 150])), 1))

def test_hilbert_layout_shared_by_samples():
	"""with chromLengths, a locus maps to the same pixel whatever the other loci of the sample"""
	L = {"1": 1000, "2": 500, "X": 300}
	a, oa = hilbertXY(["2", "X"], [10, 7], L)                 # sample without mutations on chromosome 1
	b, ob = hilbertXY(["1", "2", "X"], [5, 10, 7], L)
	assert oa == ob == 6 and np.array_equal(a, b[1:])
	assert np.array_equal(a[0], np.stack(d2xy(6, np.array([1000 + 9])), 1)[0])
	with pytest.raises(AssertionError):
		hilbertXY(["Y"], [1], L)

# ---------------------------------------------------------------- spconv (optional)

def spconvOrSkip():
	try:
		import spconv.pytorch as spconv
	except ImportError:
		pytest.skip("spconv not installed")
	if DEVICE == "cpu":
		pytest.skip("spconv tests run on a GPU (DEVICE=cuda:N)")
	return spconv

def toSpconv(x):
	import spconv.pytorch as spconv
	return spconv.SparseConvTensor(x.features, x.indices.int(), list(x.spatialShape), x.batchSize)

def spWeight(w, k):
	"""[prod(k), cin, cout] -> spconv 2.x weight [cout, *k, cin]"""
	D = len(k)
	return w.reshape(*k, w.shape[1], w.shape[2]).permute(D + 1, *range(D), D).contiguous()

def sameSites(y, ys):
	pos = {tuple(r): i for i, r in enumerate(ys.indices.long().tolist())}
	assert len(pos) == len(y.indices) and list(ys.spatial_shape) == list(y.spatialShape)
	return t.as_tensor([pos[tuple(r)] for r in y.indices.tolist()], device = y.features.device)

@pytest.mark.parametrize("D,k,s,p,subm", [(1, 3, 2, 0, False), (1, 5, 1, 0, True), (2, 3, 2, 0, False), (2, 8, 8, 0, False), (2, 3, 1, 1, False), (2, 3, 1, 0, True), (2, 5, 1, 0, True), (3, 3, 2, 0, False), (3, 2, 2, 0, False), (3, 3, 1, 0, True)])
def test_vs_spconv(D, k, s, p, subm):
	spconv = spconvOrSkip()
	x = randomTensor(D, SHAPES[D], 4, 0.2, 3, 3, t.float32)
	conv = (SubMConv(D, 3, 5, k) if subm else SparseConv(D, 3, 5, k, s, p)).to(DEVICE)
	conv.bias.data.normal_()
	name = ("SubMConv%dd" if subm else "SparseConv%dd") % D
	sp = (getattr(spconv, name)(3, 5, k) if subm else getattr(spconv, name)(3, 5, k, s, p)).to(DEVICE)
	sp.weight.data.copy_(spWeight(conv.weight.data, conv.k))
	sp.bias.data.copy_(conv.bias.data)
	y, ys = conv(x), sp(toSpconv(x))
	assert t.allclose(y.features, ys.features[sameSites(y, ys)], atol = 1e-4)
