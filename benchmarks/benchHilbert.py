"""Network benchmark on genome images, forward + backward time and peak GPU memory of the Hilbert-curve network
(SparseHilbertCNN architecture), two scenarios (synthetic data; loci half clustered, half uniform):

  panels      the same loci in every sample (a genotyping panel), dosage at every locus; batch sizes 8..128
              panelA: 7,300 loci on a 1.4e10 bp genome (2^17 x 2^17 curve, 9 conv layers)
              panelB: 75,000 loci on a 7.4e8 bp genome (2^15 x 2^15 curve, 7 conv layers)
              reticulae-fixed   SparseHilbertCNN (FixedSparseConv: rulebook built once, batch as a dense dimension)
              reticulae-tensor  the same layers as SparseConv on a SparseTensor (rulebook rebuilt for every batch)
              spconv            spconv.pytorch SparseConv2d layers
  mutations   every sample with its own mutations (300 .. 30,000 per sample) on a 2^30 bp genome (2^15 x 2^15
              curve), batch 32: reticulae (SparseTensor) vs spconv
max_abs_diff = largest difference of the predictions with respect to the first implementation (same weights).
USAGE: python benchmarks/benchHilbert.py cuda:N panels.tsv mutations.tsv"""
import os, sys
import numpy as np
import torch as t
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reticulae import SparseHilbertCNN, SparseTensor, SparseConv, d2xy
from benchLayers import timeit

BATCHES = (8, 16, 24, 32, 64, 128)
PANELS = {"panelA": (7300, 1.4e10), "panelB": (75000, 7.4e8)}

def syntheticLoci(n, genome, rng):
	"""n sorted unique positions: half uniform, half in clusters"""
	centres = rng.uniform(0, genome, n // 20)
	g = np.concatenate([rng.uniform(0, genome, n // 2), rng.choice(centres, n - n // 2) + rng.exponential(genome / n, n - n // 2)])
	return np.unique(np.clip(g, 0, genome - 1).astype(np.int64))

def layers(net, device, spconv):
	"""the conv layers of a SparseHilbertCNN as reticulae SparseConv and spconv SparseConv2d, same weights"""
	ours, sps = [], []
	for conv in net.convs:
		k, s, (ci, co) = conv.k[0], conv.s[0], conv.weight.shape[1:]
		a = SparseConv(2, ci, co, k, s).to(device)
		a.load_state_dict(conv.state_dict())
		b = spconv.SparseConv2d(ci, co, k, s).to(device)
		b.weight.data.copy_(conv.weight.data.reshape(k, k, ci, co).permute(3, 0, 1, 2))
		b.bias.data.copy_(conv.bias.data)
		ours.append(a); sps.append(b)
	return ours, sps

def runners(net, ours, sps, size, spconv):
	head = lambda d, B: net.head(d.permute(0, 2, 3, 1).reshape(B, -1))   # (row, column, channel), as SparseHilbertCNN
	def runTensor(f, idx, B):
		x = SparseTensor(f, idx, (size, size), B)
		for conv, norm in zip(ours, net.norms):
			x = conv(x)
			x = x.replace(net.act(norm(x.features)))
		return head(x.dense(), B)
	def runSpconv(f, idx, B):
		x = spconv.SparseConvTensor(f, idx.int(), [size, size], B)
		for conv, norm in zip(sps, net.norms):
			x = conv(x)
			x = x.replace_feature(net.act(norm(x.features)))
		return head(x.dense(), B)
	return runTensor, runSpconv

def measure(out, prefix, impls, params, device):
	ref = None
	for name, run in impls:
		with t.no_grad():
			y = run()
		ref = y if ref is None else ref
		diff = (y - ref).abs().max().item()
		def fwd():
			with t.no_grad():
				run()
		def step():
			for p in params:
				p.grad = None
			run().sum().backward()
		try:
			tf, _ = timeit(fwd, device, 10)
			ts, mem = timeit(step, device, 10)
		except t.OutOfMemoryError:
			tf = ts = mem = float("nan")
		t.cuda.empty_cache()
		out.write("%s\t%s\t%.3f\t%.3f\t%.1f\t%.2e\n" % (prefix, name, tf, ts, mem, diff))
		out.flush()
		print(prefix.replace("\t", " "), name, "%.1f ms fwd, %.1f ms fwd+bwd, %.0f MiB, diff %.1e" % (tf, ts, mem, diff), flush = True)

def panels(device, spconv, path):
	out = open(path, "w")
	out.write("layout\tloci\tbatch\tlibrary\tfwd_ms\tfwd_bwd_ms\tpeak_MiB\tmax_abs_diff\n")
	for name, (n, genome) in PANELS.items():
		rng = np.random.default_rng(0)
		order = int(np.ceil(np.log2(np.sqrt(genome))))
		xy = np.stack(d2xy(order, syntheticLoci(n, int(genome), rng)), 1)
		m, size = len(xy), 1 << order
		t.manual_seed(0)
		net = SparseHilbertCNN(m, 3, xy, order).to(device).eval()      # eval: no dropout, identical computations
		ours, sps = layers(net, device, spconv)
		runTensor, runSpconv = runners(net, ours, sps, size, spconv)
		keys = np.unique(xy[:, 0] * size + xy[:, 1])
		coords = t.as_tensor(np.stack([keys // size, keys % size], 1), device = device)
		params = list(net.parameters()) + [p for c in ours + sps for p in c.parameters()]
		for B in BATCHES:
			X = t.as_tensor(rng.integers(0, 3, (B, m)), dtype = t.int8, device = device)
			def sites():
				f = net.sparseInput(X)                                        # [N, B, 1]
				idx = t.cat([t.arange(B, device = device).repeat_interleave(len(coords))[:, None], coords.repeat(B, 1)], 1)
				return f.transpose(0, 1).reshape(-1, 1), idx, B
			impls = [("reticulae-fixed", lambda: net(X)["pred"]), ("reticulae-tensor", lambda: runTensor(*sites())), ("spconv", lambda: runSpconv(*sites()))]
			measure(out, "%s\t%d\t%d" % (name, m, B), impls, params, device)

def mutations(device, spconv, path, B = 32, order = 15):
	out = open(path, "w")
	out.write("mutations_per_sample\tactive_sites\tbatch\tlibrary\tfwd_ms\tfwd_bwd_ms\tpeak_MiB\tmax_abs_diff\n")
	size = 1 << order
	t.manual_seed(0)
	net = SparseHilbertCNN(10, 3, np.zeros((10, 2), dtype = np.int64), order).to(device).eval()   # its layers only
	ours, sps = layers(net, device, spconv)
	runTensor, runSpconv = runners(net, ours, sps, size, spconv)
	params = list(net.parameters()) + [p for c in ours + sps for p in c.parameters()]
	rng = np.random.default_rng(0)
	for n in (300, 1000, 3000, 10000, 30000):
		idx = []
		for b in range(B):
			x, y = d2xy(order, syntheticLoci(n, size * size, rng))
			idx.append(np.stack([np.full(len(x), b), x, y], 1))
		idx = t.as_tensor(np.concatenate(idx), device = device)
		f = t.as_tensor(rng.integers(1, 3, (len(idx), 1)), dtype = t.float32, device = device)
		impls = [("reticulae", lambda: runTensor(f, idx, B)), ("spconv", lambda: runSpconv(f, idx, B))]
		measure(out, "%d\t%d\t%d" % (n, len(idx), B), impls, params, device)

if __name__ == "__main__":
	import spconv.pytorch as spconv
	device = t.device(sys.argv[1])
	panels(device, spconv, sys.argv[2])
	mutations(device, spconv, sys.argv[3])
