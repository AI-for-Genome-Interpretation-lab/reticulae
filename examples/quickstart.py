"""Every feature of reticulae in a few lines each (runs on CPU in seconds):  python examples/quickstart.py"""
import os, sys, tempfile
import numpy as np
import torch as t
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import reticulae as R

# 1. A batch of 2D sparse images, every sample with its own active sites ---------------------------------------------
# indices: one row (sample, row, column) per active pixel; features: one row of channel values per active pixel
indices = t.tensor([[0, 1, 1], [0, 5, 2], [1, 3, 3], [1, 3, 4], [1, 7, 7]])
features = t.tensor([[1.0], [2.0], [1.0], [1.0], [2.0]])
x = R.SparseTensor(features, indices, spatialShape = (8, 8), batchSize = 2)
print("input:", x.features.shape, "active pixels, grid", x.spatialShape)

# 2. Convolutions ------------------------------------------------------------------------------------------------------
y = R.SubMConv2d(1, 8, 3)(x)                       # submanifold: same active pixels as the input
print("SubMConv2d:", y.features.shape, "(same sites)")
y = R.SparseConv2d(8, 16, 3, s = 2, p = 1)(y)      # regular, stride 2: new sites on a 4 x 4 grid
print("SparseConv2d stride 2:", y.features.shape, "grid", y.spatialShape)
y = R.SparseConv2d(16, 16, 3, d = 2, p = 2)(y)     # dilated (d = 2)
y = R.SparseConvTranspose2d(16, 8, 2, s = 2)(y)    # transposed: back to an 8 x 8 grid (more sites)
print("SparseConvTranspose2d:", y.features.shape, "grid", y.spatialShape)

# 3. A small U-Net step: down with a key, back to exactly the input sites with SparseInverseConv ------------------------
down = R.SparseConv2d(1, 8, 2, s = 2, key = "down1")
up = R.SparseInverseConv2d(8, 4, 2, key = "down1")
z = up(R.SubMConv2d(8, 8, 3)(down(x)))
print("U-Net step: input sites == output sites:", t.equal(z.indices, x.indices))

# 4. Pooling and activations ------------------------------------------------------------------------------------------
h = R.SubMConv2d(1, 8, 3)(x)
h = h.replace(t.relu(h.features))                  # any per-site operation: replace the features, keep the sites
h = R.SparseMaxPool2d(2)(h)                        # max over the active pixels of every 2 x 2 window
v = R.globalPool(h, "mean")                        # one vector per sample
print("globalPool:", v.shape, "->", t.nn.Linear(8, 3)(v).shape)
print("dense():", h.dense().shape)                 # [B, C, rows, columns], zeros where inactive

# 5. Same active sites in every sample (e.g. all genomes genotyped at the same loci) ----------------------------------
coords = np.array([[0, 0], [0, 1], [5, 5], [6, 5], [7, 7]])          # the active pixels, shared by all samples
conv = R.FixedSparseConv(coords, (8, 8), 1, 4, k = 2, s = 2)           # rulebook built once here
xf = t.randn(len(coords), 32, 1)                                      # [sites, batch, channels]
print("FixedSparseConv:", tuple(xf.shape), "->", tuple(conv(xf).shape), "on sites", conv.outCoords.tolist())

# 6. Genomes on a Hilbert curve -----------------------------------------------------------------------------------------
xy, order = R.hilbertXY(chrom = ["1"] * 50 + ["2"] * 50, pos = list(range(1, 5001, 100)) * 2)   # loci -> pixels
net = R.SparseHilbertCNN(nSnps = 100, nTraits = 2, xy = xy, order = order)
dosages = t.randint(0, 3, (4, 100), dtype = t.int8)                   # 0/1/2, -127 = missing
net.prepare(dosages.numpy())
print("SparseHilbertCNN:", net(dosages)["pred"].shape)

# 7. Compact storage of many sparse images ----------------------------------------------------------------------------
rng = np.random.default_rng(0)
mutations = [np.unique(rng.integers(0, 2 ** 20, 300)) for i in range(1000)]          # mutated positions per genome
dosage = [rng.integers(1, 3, len(m)) for m in mutations]                              # 1 or 2 (0 = absent)
xys = [np.stack(R.d2xy(10, m), 1) for m in mutations]                                 # 2^20 bp -> 1024 x 1024 image
images = R.SparseImages.fromLists(xys, dosage, (1024, 1024))
print("SparseImages: %d genomes, %d bytes (%.1f bytes per mutation: 2 x uint16 coordinates + 2-bit dosage)" % (len(images), images.nbytes, images.nbytes / sum(len(m) for m in mutations)))
batch = images.batch([0, 1, 2, 3])                                                      # -> SparseTensor
print("batch:", batch.features.shape, batch.indices.dtype)
path = os.path.join(tempfile.mkdtemp(), "images.npz")
images.save(path)
print("saved and reloaded:", t.equal(R.SparseImages.load(path).batch([5]).features, images.batch([5]).features))
