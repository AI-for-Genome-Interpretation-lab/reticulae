# reticulae API reference

Everything the library exports, with its shapes, its exact formula and a minimal example. Every example runs as
written after `import torch as t, numpy as np, reticulae as R`. A runnable tour of all of them is in
`examples/quickstart.py`.

## Contents

1. [Concepts](#1-concepts)
2. [SparseTensor](#2-sparsetensor)
3. [Convolutions (own sites per sample)](#3-convolutions-own-sites-per-sample)
4. [Pooling](#4-pooling)
5. [Fixed-sites layers (same sites in every sample)](#5-fixed-sites-layers-same-sites-in-every-sample)
6. [Storage: PackedArray and SparseImages](#6-storage-packedarray-and-sparseimages)
7. [Genomes on a Hilbert curve](#7-genomes-on-a-hilbert-curve)
8. [Low-level functions](#8-low-level-functions)
9. [Not implemented](#9-not-implemented)

## 1. Concepts

**Sparse image.** An image of D dimensions (D = 1, 2, 3, …) on a grid `spatialShape = (S_1, …, S_D)`, of which only
the *active sites* (pixels or voxels) are stored. Every other site is absent: it is treated as 0 by convolutions and
ignored by pooling.

**Two kinds of active sites.**
- **Own sites per sample.** The active pixels differ between samples: each genome stores only its own mutations, or
  each point cloud its own points. An absent site costs nothing and activates nothing. The batch is a `SparseTensor`.
- **Same sites in every sample.** Every sample has a value at the same pixels: a genotyping panel, with a dosage at
  every locus for every genome. A value of 0 is still an active site, with value 0, and the outputs around it are
  active (they get the bias). The batch is a plain tensor `[N, B, C]` (sites, batch, channels), and every `Fixed*`
  layer receives the site coordinates `[N, D]` once, when it is created.

| | own sites per sample | same sites in every sample |
|---|---|---|
| data | `SparseTensor`: `features [n, C]`, `indices [n, 1 + D]` = (sample, x_1, …, x_D) | tensor `[N, B, C]` + coordinates `[N, D]` given to each layer |
| layers | `SparseConv`, `SubMConv`, `SparseConvTranspose`, `SparseInverseConv`, `SparseMaxPool`, `SparseAvgPool`, `globalPool` | `FixedSparseConv`, `FixedSubMConv`, `FixedSparseConvTranspose`, `FixedSparseInverseConv`, `FixedSparsePool` |
| rulebook | rebuilt on the device at every call | built once when the layer is created; faster |

When every sample does have the same sites, the two give identical results. The README illustrates both on two small
genomes ([Two kinds of active sites](../README.md#two-kinds-of-active-sites)).

**Rulebook.** The list of (input site, output site, kernel offset) triples of a layer. A layer gathers the input rows,
multiplies each by the weight matrix of its offset, and adds the results into the output rows.

**Parameters** `k` (kernel size), `s` (stride), `p` (padding) and `d` (dilation) are integers or per-dimension tuples,
with the same meaning as in `torch.nn.ConvNd`.

**Weights** are `[prod(k), cin, cout]`, with the kernel offsets in row-major order (the first dimension slowest).
- To `torch.nn.functional.convNd` weights: `w.reshape(*k, cin, cout).permute(D + 1, D, *range(D))`.
- From spconv 2.x weights `[cout, *k, cin]`: `w.permute(*range(1, D + 1), D + 1, 0).reshape(-1, cin, cout)`.

**Devices, dtypes, gradients.** Every layer works on CPU and GPU, in any float dtype, with autograd, and under
`torch.autocast` (the matmuls run in the autocast dtype; gradients keep the parameter dtypes). Coordinates are int64,
and any grid with batch × prod(spatialShape) < 2^62 is supported.

## 2. SparseTensor

```text
R.SparseTensor(features, indices, spatialShape, batchSize)
```

| field | meaning |
|---|---|
| `features` | float tensor `[n, C]`, one row per active site |
| `indices` | int tensor `[n, 1 + D]`, rows (sample, x_1, …, x_D); every site at most once per sample (not checked) |
| `spatialShape` | tuple `(S_1, …, S_D)` |
| `batchSize` | number of samples B (a sample may have no active site) |
| `cache`, `history` | rulebooks carried along by the layers (internal) |

| method | result |
|---|---|
| `x.replace(f)` | same sites, new features `f [n, C']`. Use it for any per-site operation: `x.replace(t.relu(x.features))`, `x.replace(norm(x.features))` |
| `x.dense()` | dense tensor `[B, C, S_1, …, S_D]`, zeros at inactive sites |
| `R.SparseTensor.fromDense(img)` | from a dense `[B, C, S_1, …, S_D]`: the sites where any channel is non-zero |

```python
x = R.SparseTensor(t.tensor([[1.0], [2.0]]), t.tensor([[0, 1, 1], [1, 3, 0]]), (4, 4), 2)
x.dense().shape                     # torch.Size([2, 1, 4, 4])
```

## 3. Convolutions (own sites per sample)

All take a `SparseTensor` and return a `SparseTensor`. Each class has 1d/2d/3d aliases that fix D: for example
`R.SparseConv2d(cin, cout, k, ...)` = `R.SparseConv(2, cin, cout, k, ...)`.

| class | formula (b = bias) | active output sites | output shape |
|---|---|---|---|
| `SparseConv(D, cin, cout, k, s=1, p=0, d=1, bias=True, key=None)` | out[o] = b + Σ_δ W[δ] in[s·o + d·δ − p] | those whose receptive field contains an active input | (S + 2p − d(k−1) − 1) // s + 1, as `torch.nn.ConvNd` |
| `SubMConv(D, cin, cout, k, d=1, bias=True)` (k odd) | out[o] = b + Σ_δ W[δ] in[o + d·(δ − (k−1)/2)] | the input sites (sparsity never grows) | S |
| `SparseConvTranspose(D, cin, cout, k, s=1, p=0, d=1, bias=True)` | out[s·x + d·δ − p] += W[δ] in[x], plus b | those receiving from an active input | (S − 1)s − 2p + d(k−1) + 1, as `torch.nn.ConvTransposeNd` |
| `SparseInverseConv(D, cin, cout, k, key, bias=True)` | out[x] = b + Σ W[δ] in[o], over the pairs x →δ o of the `SparseConv(..., key=key)` applied earlier | exactly the input sites of that convolution | its input shape |

- **Sums run over active inputs only**, so each layer equals the dense PyTorch operation on the zero-filled image,
  read at the active output sites. The test suite checks this.
- **`SubMConv`** caches its rulebook on the tensor. Following `SubMConv` layers with the same `k` and `d` reuse it.
- **`SparseInverseConv`** undoes the downsampling of a keyed `SparseConv` (sparse U-Nets). Its input must have the
  output sites of that convolution, for example after `SubMConv` layers or pooling-free per-site operations;
  otherwise it raises an error. It needs the same kernel size.

```python
x = R.SparseTensor(t.randn(5, 1), t.tensor([[0, 1, 1], [0, 5, 2], [1, 3, 3], [1, 3, 4], [1, 7, 7]]), (8, 8), 2)
y = R.SubMConv2d(1, 8, 3)(x)                            # 5 sites, 8 channels
y = y.replace(t.relu(y.features))
z = R.SparseConv2d(8, 16, 3, s = 2, p = 1, key = "down")(y)   # grid 4 x 4
z = R.SubMConv2d(16, 16, 3, d = 2)(z)                    # dilated submanifold: same sites as z
u = R.SparseInverseConv2d(16, 8, 3, key = "down")(z)      # back to the 5 sites of y, grid 8 x 8
w = R.SparseConvTranspose2d(16, 8, 2, s = 2)(z)           # grid 8 x 8, every site reached by z
r = R.SparseConv2d(16, 16, 3, p = 2, d = 2)(z)           # dilated regular: new sites (no inverse possible from here)
```

## 4. Pooling

| object | result |
|---|---|
| `SparseMaxPool(D, k, s=None, p=0, d=1)` | max over the **active** inputs of every window (s defaults to k) |
| `SparseAvgPool(D, k, s=None, p=0, d=1)` | mean over the active inputs of every window |
| `SparsePool(D, k, s, p, d, mode)` | the same, with `mode` = `"max"` or `"avg"` |
| aliases | `SparseMaxPool1d/2d/3d(k, s, p, d)`, `SparseAvgPool1d/2d/3d(k, s, p, d)` |
| `globalPool(x, mode="max")` | `[B, C]`: max, `"mean"` or `"sum"` of every sample's active sites (0 for a sample without sites) |

- **Windows** are those of `SparseConv` with the same k, s, p and d. An output site is active iff its window contains
  an active input.
- **Max pooling** equals dense `max_poolNd` with inactive sites set to −∞. (spconv instead returns max(0, max of the
  active inputs). The two agree after a ReLU.)
- **Average pooling** divides by the number of active inputs, not by the window size.

```python
h = R.SparseMaxPool2d(2)(x)                  # 2 x 2 windows, stride 2
v = R.globalPool(R.SubMConv2d(1, 8, 3)(x), "mean")   # [2, 8] -> e.g. t.nn.Linear(8, n_classes)
```

## 5. Fixed-sites layers (same sites in every sample)

Each layer is created from the active sites `coords [N, D]` (numpy or tensor) and the grid `spatialShape`. It takes
`[N, B, cin]`, sites in the order of `coords`, and returns `[nOut, B, cout]`, sites in the order of `outCoords`. The
formulas are those of sections 3 and 4.

| class | |
|---|---|
| `FixedSparseConv(coords, spatialShape, cin, cout, k, s=1, p=0, d=1, bias=True)` | regular |
| `FixedSubMConv(coords, spatialShape, cin, cout, k, d=1, bias=True)` | submanifold (`outCoords` = `coords`) |
| `FixedSparseConvTranspose(coords, spatialShape, cin, cout, k, s=1, p=0, d=1, bias=True)` | transposed |
| `FixedSparseInverseConv(conv, cin, cout, bias=True)` | inverse of the `FixedSparseConv` `conv`: back to `conv.inCoords` |
| `FixedSparsePool(coords, spatialShape, k, s=None, p=0, d=1, mode="max")` | pooling, `mode` = `"max"` or `"avg"` |

Attributes: `outCoords` (numpy `[nOut, D]`), `outShape`, `nIn`, `nOut`, `nPairs` (rulebook size), `inCoords` and
`inShape` (convolutions). To stack layers, pass `conv.outCoords, conv.outShape` to the next one.

```python
coords = np.array([[0, 0], [0, 1], [5, 5], [6, 5], [7, 7]])
c1 = R.FixedSparseConv(coords, (8, 8), 1, 4, k = 2, s = 2)
c2 = R.FixedSubMConv(c1.outCoords, c1.outShape, 4, 4, k = 3)
y = c2(c1(t.randn(5, 32, 1)))                # [c1.nOut, 32, 4]
```

## 6. Storage: PackedArray and SparseImages

**`PackedArray(values)`.** A numpy array `[n]` or `[n, C]` stored with the fewest bits:

| values | storage |
|---|---|
| all equal | the single value, 0 bytes of data |
| integers in [0, 2), [0, 4) or [0, 16) | bit-packed, 1, 2 or 4 bits per value (8, 4 or 2 values per byte) |
| other integers | the smallest of int8/uint8 … int64/uint64 that holds them |
| floats | unchanged (pass float16 to halve the size) |

`a[rows]` decodes rows (numpy). `a.nbytes` is the data size, `len(a)` the number of rows. Dosages 0/1/2 with missing
recoded as 3 take 2 bits per genotype: `R.PackedArray(np.where(X == -127, 3, X))`.

**`SparseImages`.** Many sparse images, each with its own active sites, kept compactly in CPU memory:
- `coords`: one array per dimension, in the smallest unsigned type that holds the grid size (uint8 up to 256,
  uint16 up to 65,536, uint32 up to 2^32);
- `values`: a `PackedArray`;
- `offsets`: int64 `[n_samples + 1]`; sample i is rows `offsets[i]:offsets[i+1]`.

| member | |
|---|---|
| `SparseImages.fromLists(coords, values, spatialShape)` | `coords`: one int array `[n_i, D]` per sample (unique sites); `values`: one array `[n_i]` or `[n_i, C]` per sample, or `None` (every site = 1, stored in 0 bytes) |
| `SparseImages.fromDense(images)` | numpy `[N, S_1, …, S_D]`: the non-zero pixels |
| `images.batch(ids, device="cpu", dtype=t.float32)` | a `SparseTensor` of the samples `ids` (sample b of the batch = `ids[b]`), features `[n, C]` |
| `images.sample(i)` | (coordinates int64 `[n_i, D]`, values) |
| `images.save(path)`, `SparseImages.load(path)` | `.npz` file, exact round trip |
| `len(images)`, `images.nbytes` | number of samples, bytes used |

For genomes, see the README section [Compact storage of genomic data](../README.md#compact-storage-of-genomic-data-sparseimages)
(sizes measured by `examples/storageGenomes.py`, and the recipe from per-sample chromosome / position / genotype lists
through `hilbertXY(..., chromLengths)` to `SparseImages`).

```python
rng = np.random.default_rng(0)
muts = [np.unique(rng.integers(0, 2 ** 20, 300)) for i in range(1000)]          # mutated positions
dosage = [rng.integers(1, 3, len(m)) for m in muts]                              # 1 or 2 -> 2 bits
images = R.SparseImages.fromLists([np.stack(R.d2xy(10, m), 1) for m in muts], dosage, (1024, 1024))
images.nbytes / sum(map(len, muts))          # ~4.3 bytes per mutation: 2 x uint16 + 2 bits + offsets
x = images.batch([0, 1, 2], device = "cpu")  # SparseTensor
```

## 7. Genomes on a Hilbert curve

| function / class | |
|---|---|
| `d2xy(order, d)` | Hilbert-curve index `d` (numpy int array) → (x, y) on a 2^order × 2^order grid. Neighbouring positions are neighbouring pixels |
| `hilbertXY(chrom, pos, chromLengths=None)` | loci (chromosome, 1-based position) → `xy [m, 2]`, `order`. The chromosomes are concatenated, one base pair per pixel, and order = ceil(log2(sqrt(genome length))). With `chromLengths` (dict {name: length}), the chromosome order and lengths come from the dict, so every call (every sample) uses the same layout and a base always maps to the same pixel. Use it for per-sample mutation lists. Without it, chromosomes are taken in order of appearance, with the last locus as length (a panel mapped once) |
| `SparseHilbertCNN(nSnps, nTraits, xy, order, channels=(1, 4, 16, 32, 64, 128, 128, 128), dropout=0.3, hidden=50)` | the HC network of SchizophreniaGI on a panel of loci (same loci in every sample). Layers: 8×8/8 sparse convolutions while the grid stays ≥ 32 after the layer, then 3×3/2 until the grid is ≤ 20, LayerNorm and LeakyReLU, dense head |

`SparseHilbertCNN`:
- `net.prepare(X)` sets the training means used for missing genotypes (`X` int8 `[samples, loci]`, −127 = missing).
- `net(X)["pred"]` returns `[B, nTraits]`; loci sharing a pixel are averaged.
- `net.describe()` prints the grid and active sites of every level.

```python
xy, order = R.hilbertXY(["1"] * 50 + ["2"] * 50, list(range(1, 5001, 100)) * 2)
net = R.SparseHilbertCNN(100, 2, xy, order)
X = t.randint(0, 3, (4, 100), dtype = t.int8)
net.prepare(X.numpy()); net(X)["pred"].shape  # [4, 2]
```

## 8. Low-level functions

| object | |
|---|---|
| `R.SparseConvFn.apply(x, weight, inTab, outTab, offs, nOut, centre=-1)` | the autograd kernel: `x [N, B, cin]` → `[nOut, B, cout]` from padded rulebook tables |
| `reticulae.Rulebook.regular / transposed / submanifold / tables` | rulebook builders (torch, on the device of the indices) |
| `reticulae.Convolutions.TEMP_BYTES` | budget for the temporary gathered rows of a layer (default 2^28 bytes). Beyond it the kernel offsets are processed in chunks: less memory, slightly more time |

## 9. Not implemented

- **Grouped convolutions** (`groups` in `torch.nn.ConvNd`). The channels are split into g groups, and each group of
  output channels sees only its own group of input channels: g times fewer weights. With g = cin it is a depthwise
  convolution, one filter per channel, as in MobileNet-style networks.
- **Duplicate sites in a `SparseTensor`.** These are not merged; `SparseImages.fromLists` rejects them.
