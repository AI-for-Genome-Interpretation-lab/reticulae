"""Compact storage of genomic data with SparseImages and PackedArray (simulated data; CPU, seconds):
	python examples/storageGenomes.py
1. per-sample mutation lists (chromosome, position, genotype 1 = heterozygous / 2 = homozygous) of 1,000 genomes of
   3.1e9 bp -> pixels of one shared Hilbert-curve layout -> SparseImages -> .npz -> SparseTensor batches;
2. a genotyping panel ([samples, loci] dosages 0/1/2, missing) -> PackedArray.
Prints the memory of each representation (results/storageGenomes.tsv)."""
import os, sys, tempfile
import numpy as np
import torch as t
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import reticulae as R

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
	rng = np.random.default_rng(0)
	rows = []
	# 1. mutation lists --------------------------------------------------------------------------------------------
	layout = {"chr%d" % (i + 1): 129_000_000 for i in range(24)}                  # 24 chromosomes, 3.1e9 bp in total
	names = list(layout)
	samples = []
	for i in range(1000):
		n = rng.poisson(3000)
		chrom = rng.choice(names, n)
		pos = rng.integers(1, 129_000_001, n)
		genotype = rng.choice([1, 2], n, p = [0.7, 0.3])
		samples.append((chrom, pos, genotype))
	coords, values = [], []
	for chrom, pos, genotype in samples:
		xy, order = R.hilbertXY(chrom, pos, layout)                               # same layout for every sample
		xy, first = np.unique(xy, axis = 0, return_index = True)                  # a base mutated twice: keep one
		coords.append(xy); values.append(genotype[first])
	size = 1 << order
	images = R.SparseImages.fromLists(coords, values, (size, size))
	nm = int(images.offsets[-1])
	print("1,000 genomes, %d mutations, Hilbert curve %d x %d (order %d); coordinates %s, genotypes %d bits" % (nm, size, size, order, images.coords[0].dtype, images.values.bits))
	rows += [("mutation lists", "dense uint8 images", float(size) ** 2 * len(images)),
		("mutation lists", "SparseTensor (int64 indices + float32 values)", nm * (3 * 8 + 4)),
		("mutation lists", "int64 genome position + int8 genotype", nm * 9),
		("mutation lists", "SparseImages (genotypes)", images.nbytes),
		("mutation lists", "SparseImages (presence only, values=None)", R.SparseImages.fromLists(coords, None, (size, size)).nbytes)]
	path = os.path.join(tempfile.mkdtemp(), "genomes.npz")
	images.save(path)
	rows.append(("mutation lists", ".npz file of SparseImages (genotypes)", os.path.getsize(path)))
	x = R.SparseImages.load(path).batch([0, 1, 2, 3])                            # -> SparseTensor for the layers
	print("batch of 4 genomes:", tuple(x.features.shape), "features,", tuple(x.indices.shape), "indices")
	# 2. genotyping panel ------------------------------------------------------------------------------------------
	X = rng.integers(0, 3, (1000, 100_000)).astype(np.int8)
	X[rng.random(X.shape) < 0.02] = -127                                          # missing
	packed = R.PackedArray(np.where(X == -127, 3, X))                             # 0/1/2, 3 = missing -> 2 bits
	assert np.array_equal(packed[np.arange(10)], np.where(X[:10] == -127, 3, X[:10]))
	rows += [("panel", "int8 matrix", X.nbytes), ("panel", "PackedArray (2 bits per genotype)", packed.nbytes)]
	os.makedirs(os.path.join(HERE, "results"), exist_ok = True)
	with open(os.path.join(HERE, "results", "storageGenomes.tsv"), "w") as f:
		f.write("data\trepresentation\tbytes\tbytes_per_entry\n")
		for data, rep, b in rows:
			per = b / (nm if data == "mutation lists" else X.size)
			f.write("%s\t%s\t%d\t%.3f\n" % (data, rep, b, per))
			print("%-15s %-48s %14.0f bytes  %10.3f bytes per %s" % (data, rep, b, per, "mutation" if data == "mutation lists" else "genotype"))

if __name__ == "__main__":
	main()
