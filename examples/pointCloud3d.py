"""3D point clouds with submanifold convolutions: classify voxelised sphere, cube and torus surfaces (random size,
position and noise) on a 128^3 grid. SubMConv3d keeps the active sites of the surface (no dilation), strided
SparseConv3d downsamples, a per-sample max over the remaining sites feeds a linear classifier.
Writes examples/results/pointCloud3d.{preds.tsv,perf.tsv}.   USAGE: python examples/pointCloud3d.py cuda:N|cpu"""
import os, sys
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reticulae import SparseTensor, SparseConv3d, SubMConv3d

HERE = os.path.dirname(os.path.abspath(__file__))
S = 128

def shape(kind, rng):
	u = rng.standard_normal((4000, 3))
	if kind == 0:                                                        # sphere
		p = u / np.linalg.norm(u, axis = 1, keepdims = True)
	elif kind == 1:                                                      # cube surface
		p = rng.uniform(-1, 1, (4000, 3))
		a = rng.integers(0, 3, 4000)
		p[np.arange(4000), a] = np.sign(p[np.arange(4000), a])
	else:                                                                # torus
		th, ph = rng.uniform(0, 2 * np.pi, (2, 4000))
		p = np.stack([(0.7 + 0.3 * np.cos(ph)) * np.cos(th), (0.7 + 0.3 * np.cos(ph)) * np.sin(th), 0.3 * np.sin(ph)], 1)
	R = np.linalg.qr(rng.standard_normal((3, 3)))[0]
	r = rng.uniform(15, 35)
	p = (p @ R) * r + rng.uniform(r + 2, S - r - 2, 3) + rng.normal(0, 0.5, p.shape)
	return np.unique(np.clip(p.astype(np.int64), 0, S - 1), axis = 0)

class Net(nn.Module):

	def __init__(self):
		super().__init__()
		self.layers = nn.ModuleList([SubMConv3d(1, 16, 3), SparseConv3d(16, 32, 3, 2), SubMConv3d(32, 32, 3), SparseConv3d(32, 64, 3, 2), SubMConv3d(64, 64, 3), SparseConv3d(64, 64, 3, 2)])
		self.norms = nn.ModuleList([nn.LayerNorm(c) for c in (16, 32, 32, 64, 64, 64)])
		self.out = nn.Linear(64, 3)

	def forward(self, x):
		for conv, norm in zip(self.layers, self.norms):
			x = conv(x)
			x = x.replace(F.relu(norm(x.features)))
		pooled = x.features.new_full((x.batchSize, x.features.shape[1]), -1e9).index_reduce_(0, x.indices[:, 0], x.features, "amax")
		return self.out(pooled)

def batch(clouds, ids, device):
	idx = t.as_tensor(np.concatenate([np.concatenate([np.full((len(clouds[i]), 1), b), clouds[i]], 1) for b, i in enumerate(ids)]), device = device)
	return SparseTensor(t.ones(len(idx), 1, device = device), idx, (S, S, S), len(ids))

def main(args):
	device = t.device(args[1] if len(args) > 1 else "cpu")
	t.manual_seed(0)
	rng = np.random.default_rng(0)
	y = rng.integers(0, 3, 900)
	clouds = [shape(k, rng) for k in y]
	test = rng.permutation(len(y)) < 300
	tr = np.nonzero(~test)[0]
	print("active voxels per sample: %.0f of %d" % (np.mean([len(c) for c in clouds]), S ** 3))
	net = Net().to(device)
	opt = t.optim.AdamW(net.parameters(), lr = 2e-3, weight_decay = 1e-4)
	yd = t.as_tensor(y, device = device)
	for e in range(15):
		net.train()
		perm = rng.permutation(tr)
		tot = 0.0
		for i in range(0, len(perm), 16):
			b = perm[i:i + 16]
			loss = F.cross_entropy(net(batch(clouds, b, device)), yd[b])
			opt.zero_grad(); loss.backward()
			t.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
			opt.step()
			tot += loss.item() * len(b)
		print("epoch %2d train loss %.3f" % (e + 1, tot / len(perm)), flush = True)
	net.eval()
	with t.no_grad():
		p = np.concatenate([net(batch(clouds, np.arange(i, min(i + 64, len(y))), device)).argmax(1).cpu().numpy() for i in range(0, len(y), 64)])
	os.makedirs(os.path.join(HERE, "results"), exist_ok = True)
	with open(os.path.join(HERE, "results", "pointCloud3d.perf.tsv"), "w") as f:
		f.write("set\taccuracy\n")
		for name, m in (("train", ~test), ("test", test)):
			acc = (p[m] == y[m]).mean()
			f.write("%s\t%.4f\n" % (name, acc))
			print("%s accuracy %.3f" % (name, acc))
	with open(os.path.join(HERE, "results", "pointCloud3d.preds.tsv"), "w") as f:
		f.write("sample\tset\tlabel\tpred\n")
		for i in range(len(y)):
			f.write("%d\t%s\t%d\t%d\n" % (i, "test" if test[i] else "train", y[i], p[i]))

if __name__ == "__main__":
	main(sys.argv)
