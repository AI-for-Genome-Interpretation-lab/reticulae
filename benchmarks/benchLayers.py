"""Single-layer benchmark, reticulae vs spconv: forward + backward time and peak GPU memory of one sparse convolution on
random inputs whose active sites differ in every sample (the rulebook is built at every call by both libraries).
Every configuration first checks that the two outputs agree (max |difference| is stored).
USAGE: python benchmarks/benchLayers.py cuda:N out.tsv"""
import os, sys, time, math
import numpy as np
import torch as t
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reticulae import SparseTensor, SparseConv, SubMConv
from reticulae import Rulebook as RB

# name, D, grid side, kernel, stride, subm, cin, cout, batch, active sites per sample
CONFIGS = []
for n in (1e3, 1e4, 1e5, 5e5):
	CONFIGS += [("1D regular k9 s4", 1, 2 ** 30, 9, 4, False, 16, 32, 8, n), ("1D submanifold k5", 1, 2 ** 30, 5, 1, True, 32, 32, 8, n),
		("2D regular 3x3 s2", 2, 4096, 3, 2, False, 32, 64, 8, n), ("2D submanifold 3x3", 2, 4096, 3, 1, True, 32, 32, 8, n),
		("3D regular 3x3x3 s2", 3, 512, 3, 2, False, 32, 64, 4, n), ("3D submanifold 3x3x3", 3, 512, 3, 1, True, 32, 32, 4, n)]

def sites(D, S, n, rng):
	"""n sites: clustered (random walks, as surfaces / reads), so that neighbourhoods are populated"""
	steps = rng.integers(-1, 2, (int(n), D))
	starts = rng.integers(0, S, (max(1, int(n) // 200), D))
	walk = np.cumsum(steps.reshape(-1, 200 if n >= 200 else int(n), D), 1) + starts[:, None, :]
	c = np.unique(np.clip(walk.reshape(-1, D), 0, S - 1), axis = 0)
	return c

def inputs(D, S, B, n, C, seed, device):
	rng = np.random.default_rng(seed)
	idx = np.concatenate([np.concatenate([np.full((len(c), 1), b), c], 1) for b in range(B) for c in [sites(D, S, n, rng)]])
	idx = t.as_tensor(idx, device = device)
	return idx, t.randn(len(idx), C, device = device)

def timeit(f, device, reps):
	for _ in range(2):
		f()
	t.cuda.synchronize(device)
	t.cuda.reset_peak_memory_stats(device)
	base = t.cuda.memory_allocated(device)
	ts = []
	for _ in range(reps):
		t0 = time.time()
		f()
		t.cuda.synchronize(device)
		ts.append(time.time() - t0)
	return 1000 * np.median(ts), (t.cuda.max_memory_allocated(device) - base) / 2 ** 20

def main(args):
	device = t.device(args[1])
	import spconv.pytorch as spconv
	out = open(args[2], "w")
	out.write("config\tD\tactive_per_sample\tbatch\tnnz\tcin\tcout\tlibrary\tfwd_ms\tfwd_bwd_ms\tpeak_MiB\tmax_abs_diff\n")
	for name, D, S, k, s, subm, cin, cout, B, n in CONFIGS:
		idx, f = inputs(D, S, B, n, cin, 0, device)
		shape = [S] * D
		ours = (SubMConv(D, cin, cout, k) if subm else SparseConv(D, cin, cout, k, s)).to(device)
		sp = (getattr(spconv, "SubMConv%dd" % D)(cin, cout, k) if subm else getattr(spconv, "SparseConv%dd" % D)(cin, cout, k, s)).to(device)
		sp.weight.data.copy_(ours.weight.data.reshape(*([k] * D), cin, cout).permute(D + 1, *range(D), D))
		sp.bias.data.copy_(ours.bias.data)
		fr = f.clone().requires_grad_()
		fs = f.clone().requires_grad_()
		runOurs = lambda: ours(SparseTensor(fr, idx, shape, B))
		runSp = lambda: sp(spconv.SparseConvTensor(fs, idx.int(), shape, B))
		with t.no_grad():
			y, ys = runOurs(), runSp()
			ka = RB.linear(y.indices[:, 0], y.indices[:, 1:], y.spatialShape)
			kb = RB.linear(ys.indices[:, 0].long(), ys.indices[:, 1:].long(), y.spatialShape)
			assert len(ka) == len(kb) and t.equal(t.sort(ka)[0], t.sort(kb)[0]), "different output sites"
			diff = (y.features[t.argsort(ka)] - ys.features[t.argsort(kb)]).abs().max().item()
		reps = 20 if len(idx) < 1e6 else 8
		for lib, run in (("reticulae", runOurs), ("spconv", runSp)):
			def fwd():
				with t.no_grad():
					run()
			def step():
				run().features.sum().backward()
			try:
				tf, _ = timeit(fwd, device, reps)
				ts, mem = timeit(step, device, reps)
			except t.OutOfMemoryError:
				tf = ts = mem = float("nan")
			t.cuda.empty_cache()
			out.write("%s\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%.3f\t%.3f\t%.1f\t%.2e\n" % (name, D, n, B, len(idx), cin, cout, lib, tf, ts, mem, diff))
			out.flush()
			print(name, int(n), lib, "%.2f ms fwd, %.2f ms fwd+bwd, %.0f MiB, diff %.1e" % (tf, ts, mem, diff), flush = True)
		del ours, sp, idx, f, fr, fs
		t.cuda.empty_cache()

if __name__ == "__main__":
	main(sys.argv)
