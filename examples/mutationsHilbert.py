"""Genome images with DIFFERENT active sites in every sample: each sample is the list of its own mutations (~300 on a
2^30 bp genome, Hilbert curve 2^15 x 2^15) with their dosage (1 = heterozygous, 2 = homozygous; reference bases are
absent). The genomes are kept in a SparseImages (uint16 coordinates, 2-bit dosages) and every batch becomes a
SparseTensor: memory grows with the number of mutations, never with the 10^9 pixels of the images. The phenotype is a
rare-variant burden: the summed dosage of the mutations in 10 gene regions of 2 Mb, plus noise (heritability 0.7). The
dense head on the final 63 x 63 map (~270 kb per cell) locates the regions.
Writes examples/results/mutationsHilbert.{preds.tsv,perf.tsv}.   USAGE: python examples/mutationsHilbert.py cuda:N|cpu"""
import os, sys
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reticulae import SparseImages, SparseConv2d, d2xy

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER, REGION = 15, 2000000

def simulate(n, rng):
	"""-> SparseImages of the genomes, phenotypes"""
	genes = rng.integers(0, 2 ** (2 * ORDER) - REGION, 10)
	muts = [np.unique(rng.integers(0, 2 ** (2 * ORDER), rng.poisson(300))) for i in range(n)]
	dosage = [rng.choice([1, 2], len(m), p = [0.8, 0.2]) for m in muts]
	g = np.array([(((m[:, None] >= genes) & (m[:, None] < genes + REGION)).any(1) * v).sum() for m, v in zip(muts, dosage)], dtype = np.float64)
	g = (g - g.mean()) / g.std()
	y = np.sqrt(0.7) * g + np.sqrt(0.3) * rng.standard_normal(n)
	images = SparseImages.fromLists([np.stack(d2xy(ORDER, m), 1) for m in muts], dosage, (2 ** ORDER, 2 ** ORDER))
	return images, y.astype(np.float32)

class Net(nn.Module):
	"""8x8/8 x 2 (learned pooling of 64 x 64 pixel blocks), then 3x3/2 down to 63 x 63, dense head"""

	def __init__(self):
		super().__init__()
		ch = [(1, 8, 8, 8), (8, 16, 8, 8), (16, 16, 3, 2), (16, 16, 3, 2), (16, 8, 3, 2)]
		self.convs = nn.ModuleList([SparseConv2d(a, b, k, s) for a, b, k, s in ch])
		self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(8 * 63 * 63, 1))

	def forward(self, x):
		for conv in self.convs:
			x = conv(x)
			x = x.replace(F.leaky_relu(x.features))
		return self.head(x.dense().reshape(x.batchSize, -1))[:, 0]

def main(args):
	device = t.device(args[1] if len(args) > 1 else "cpu")
	t.manual_seed(0)
	rng = np.random.default_rng(0)
	images, y = simulate(6000, rng)
	test = rng.permutation(len(y)) < 1000
	tr, te = np.nonzero(~test)[0], np.nonzero(test)[0]
	n = images.offsets[-1]
	print("%d genomes, %.0f mutations per genome, stored in %.1f MB (%.2f bytes per mutation); one dense image: %.1e pixels" % (len(images), n / len(images), images.nbytes / 1e6, images.nbytes / n, 4.0 ** ORDER))
	net = Net().to(device)
	opt = t.optim.AdamW(net.parameters(), lr = 1e-3, weight_decay = 1e-4)
	yd = t.as_tensor(y, device = device)
	for e in range(25):
		net.train()
		perm = rng.permutation(tr)
		tot = 0.0
		for i in range(0, len(perm), 32):
			b = perm[i:i + 32]
			loss = F.mse_loss(net(images.batch(b, device)), yd[b])
			opt.zero_grad(); loss.backward()
			t.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
			opt.step()
			tot += loss.item() * len(b)
		print("epoch %2d train MSE %.3f" % (e + 1, tot / len(perm)), flush = True)
	net.eval()
	with t.no_grad():
		p = np.concatenate([net(images.batch(np.arange(i, min(i + 128, len(y))), device)).cpu().numpy() for i in range(0, len(y), 128)])
	os.makedirs(os.path.join(HERE, "results"), exist_ok = True)
	with open(os.path.join(HERE, "results", "mutationsHilbert.perf.tsv"), "w") as f:
		f.write("set\tpearson_r\n")
		for name, m in (("train", ~test), ("test", test)):
			r = np.corrcoef(p[m], y[m])[0, 1]
			f.write("%s\t%.4f\n" % (name, r))
			print("%s r = %.3f" % (name, r))
	with open(os.path.join(HERE, "results", "mutationsHilbert.preds.tsv"), "w") as f:
		f.write("sample\tset\ty\tpred\n")
		for i in range(len(y)):
			f.write("%d\t%s\t%.4f\t%.4f\n" % (i, "test" if test[i] else "train", y[i], p[i]))

if __name__ == "__main__":
	main(sys.argv)
