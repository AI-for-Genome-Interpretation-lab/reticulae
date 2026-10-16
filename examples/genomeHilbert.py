"""Genome images with the same loci in every sample (a genotyping panel): SparseHilbertCNN on 7,300 simulated loci of
a 1.4e10 bp genome (2^17 x 2^17 Hilbert curve), simulated genotypes and two traits, each the sum of the dosages of one
window of 100 neighbouring loci plus noise (heritability 0.8).
Writes examples/results/genomeHilbert.{preds.tsv,perf.tsv}.   USAGE: python examples/genomeHilbert.py cuda:N|cpu"""
import os, sys
import numpy as np
import torch as t
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reticulae import SparseHilbertCNN, hilbertXY

HERE = os.path.dirname(os.path.abspath(__file__))

def simulate(m, n, rng):
	X = rng.binomial(2, rng.uniform(0.05, 0.5, m)[None], (n, m)).astype(np.int8)
	Y = []
	for tr in range(2):
		a = rng.integers(0, m - 100)
		g = X[:, a:a + 100].sum(1).astype(np.float64)
		g = (g - g.mean()) / g.std()
		Y.append(np.sqrt(0.8) * g + np.sqrt(0.2) * rng.standard_normal(n))
	return X, np.stack(Y, 1).astype(np.float32)

def main(args):
	device = t.device(args[1] if len(args) > 1 else "cpu")
	t.manual_seed(0)
	rng = np.random.default_rng(0)
	pos = np.unique(rng.integers(1, 7e8, 7400))[:7300]                       # 7,300 loci on 20 chromosomes of 7e8 bp
	xy, order = hilbertXY(np.repeat(np.arange(20), 365), np.sort(pos.reshape(20, 365), 1).reshape(-1), {str(c): 7e8 for c in range(20)})
	X, Y = simulate(len(xy), 3000, rng)
	test = rng.permutation(len(X)) < 600
	net = SparseHilbertCNN(len(xy), Y.shape[1], xy, order).to(device)
	print(net.describe())
	net.prepare(X[~test])
	Xd, Yd = t.as_tensor(X, device = device), t.as_tensor(Y, device = device)
	tr = t.as_tensor(np.nonzero(~test)[0], device = device)
	opt = t.optim.AdamW(net.parameters(), lr = 1e-4, weight_decay = 1e-4)
	for e in range(40):
		net.train()
		perm = tr[t.randperm(len(tr), device = device)]
		tot = 0.0
		for i in range(0, len(perm), 32):
			b = perm[i:i + 32]
			loss = F.mse_loss(net(Xd[b])["pred"], Yd[b])
			opt.zero_grad(); loss.backward()
			t.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
			opt.step()
			tot += loss.item() * len(b)
		print("epoch %2d train MSE %.3f" % (e + 1, tot / len(perm)), flush = True)
	net.eval()
	with t.no_grad():
		P = t.cat([net(Xd[i:i + 128])["pred"] for i in range(0, len(Xd), 128)]).cpu().numpy()
	os.makedirs(os.path.join(HERE, "results"), exist_ok = True)
	with open(os.path.join(HERE, "results", "genomeHilbert.perf.tsv"), "w") as f:
		f.write("trait\tset\tpearson_r\n")
		for j in range(Y.shape[1]):
			for name, m in (("train", ~test), ("test", test)):
				r = np.corrcoef(P[m, j], Y[m, j])[0, 1]
				f.write("trait%d\t%s\t%.4f\n" % (j, name, r))
				print("trait %d %s r = %.3f" % (j, name, r))
	with open(os.path.join(HERE, "results", "genomeHilbert.preds.tsv"), "w") as f:
		f.write("sample\tset\t" + "\t".join("y%d\tpred%d" % (j, j) for j in range(Y.shape[1])) + "\n")
		for i in range(len(Y)):
			f.write("%d\t%s\t%s\n" % (i, "test" if test[i] else "train", "\t".join("%.4f\t%.4f" % (Y[i, j], P[i, j]) for j in range(Y.shape[1]))))

if __name__ == "__main__":
	main(sys.argv)
