"""Cost of the deterministic scatter-add (DETERMINISTIC = True, the default) against one scatter of all the kernel
offsets with CUDA atomics (DETERMINISTIC = False): forward + backward time of the layers of benchLayers.py, and whether
two identical passes give bitwise-identical gradients.
USAGE: python benchmarks/benchDeterminism.py cuda:N out.tsv"""
import os, sys
import torch as t
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import benchLayers as BL
import reticulae.Convolutions as C
from reticulae import SparseTensor, SparseConv, SubMConv

def main(args):
	device = t.device(args[1])
	out = open(args[2], "w")
	out.write("config\tD\tactive_per_sample\tbatch\tnnz\tdeterministic\tfwd_bwd_ms\tbitwise_repeatable\n")
	for name, D, S, k, s, subm, cin, cout, B, n in BL.CONFIGS:
		idx, f = BL.inputs(D, S, B, n, cin, 0, device)
		conv = (SubMConv(D, cin, cout, k) if subm else SparseConv(D, cin, cout, k, s)).to(device)
		for det in (False, True):
			C.DETERMINISTIC = det
			fr = f.clone().requires_grad_()
			def step():
				conv(SparseTensor(fr, idx, [S] * D, B)).features.sum().backward()
			g = []
			for _ in range(2):
				fr.grad = None; conv.zero_grad(); step()
				g.append((fr.grad.clone(), conv.weight.grad.clone()))
			same = t.equal(g[0][0], g[1][0]) and t.equal(g[0][1], g[1][1])
			ms, _ = BL.timeit(step, device, 20 if len(idx) < 1e6 else 8)
			out.write("%s\t%d\t%d\t%d\t%d\t%s\t%.3f\t%s\n" % (name, D, n, B, len(idx), det, ms, same))
			out.flush()
			print(name, int(n), "deterministic" if det else "atomic", "%.2f ms fwd+bwd, repeatable %s" % (ms, same), flush = True)
		del conv, idx, f
		t.cuda.empty_cache()
	C.DETERMINISTIC = True

if __name__ == "__main__":
	main(sys.argv)
