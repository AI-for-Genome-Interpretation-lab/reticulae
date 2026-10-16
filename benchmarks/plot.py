"""Figures of the benchmarks (results/*.tsv -> docs/*.png). USAGE: python benchmarks/plot.py"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs")
COL = {"reticulae": "#1b7837", "reticulae-fixed": "#1b7837", "reticulae-tensor": "#7fbf7b", "spconv": "#c51b7d"}

def layers():
	d = pd.read_csv(os.path.join(HERE, "results", "layers.tsv"), sep = "\t")
	d["sites"] = d.nnz / d.batch
	cfg = list(dict.fromkeys(d.config))
	fig, ax = plt.subplots(2, len(cfg), figsize = (3.1 * len(cfg), 6.2), sharex = "col")
	for j, c in enumerate(cfg):
		for lib in ("reticulae", "spconv"):
			e = d[(d.config == c) & (d.library == lib)].sort_values("sites")
			ax[0, j].plot(e.sites, e.fwd_bwd_ms, "o-", color = COL[lib], label = lib + " fwd+bwd")
			ax[0, j].plot(e.sites, e.fwd_ms, "o--", color = COL[lib], alpha = 0.7, label = lib + " fwd")
			ax[1, j].plot(e.sites, e.peak_MiB, "o-", color = COL[lib], label = lib)
		e = d[d.config == c]
		ax[0, j].set_title("%s\nbatch %d, %d->%d ch" % (c, e.batch.iloc[0], e.cin.iloc[0], e.cout.iloc[0]), fontsize = 9)
		for a in ax[:, j]:
			a.set_xscale("log"); a.set_yscale("log"); a.grid(alpha = 0.3, which = "both")
		ax[1, j].set_xlabel("active sites per sample")
	ax[0, 0].set_ylabel("time (ms)")
	ax[1, 0].set_ylabel("peak memory (MiB)")
	ax[0, 0].legend(fontsize = 7)
	fig.suptitle("One layer, different active sites in every sample (rulebook built at every call); NVIDIA L40S, fp32", fontsize = 10)
	fig.tight_layout()
	fig.savefig(os.path.join(DOCS, "benchmark_layers.png"), dpi = 110)

def genomes():
	d = pd.read_csv(os.path.join(HERE, "results", "panels.tsv"), sep = "\t")
	mu = pd.read_csv(os.path.join(HERE, "results", "mutations.tsv"), sep = "\t")
	fig, ax = plt.subplots(2, 3, figsize = (15, 7.5))
	titles = {"panelA": "same loci in every sample: 7,300 loci, 2^17 x 2^17 curve", "panelB": "same loci in every sample: 75,000 loci, 2^15 x 2^15 curve"}
	for j, org in enumerate(("panelA", "panelB")):
		e0 = d[d.layout == org]
		for lib in ("reticulae-fixed", "reticulae-tensor", "spconv"):
			e = e0[e0.library == lib].sort_values("batch")
			ax[0, j].plot(e.batch, e.fwd_bwd_ms, "o-", color = COL[lib], label = lib)
			ax[1, j].plot(e.batch, e.peak_MiB, "o-", color = COL[lib], label = lib)
			bad = e[e.max_abs_diff > 1e-3]
			if len(bad):
				ax[0, j].plot(bad.batch, bad.fwd_bwd_ms, "x", color = "red", ms = 11, mew = 2, label = "%s: wrong output (max |diff| %.1f)" % (lib, bad.max_abs_diff.max()))
		ax[0, j].set_title(titles[org], fontsize = 9)
		for a in ax[:, j]:
			a.set_xscale("log", base = 2); a.set_yscale("log"); a.grid(alpha = 0.3, which = "both")
			a.set_xticks(sorted(d.batch.unique())); a.set_xticklabels(sorted(d.batch.unique()))
		ax[1, j].set_xlabel("batch size")
	for lib in ("reticulae", "spconv"):
		e = mu[mu.library == lib].sort_values("mutations_per_sample")
		ax[0, 2].plot(e.mutations_per_sample, e.fwd_bwd_ms, "o-", color = COL[lib], label = lib + " (SparseTensor)" if lib == "reticulae" else lib)
		ax[1, 2].plot(e.mutations_per_sample, e.peak_MiB, "o-", color = COL[lib], label = lib)
	ax[0, 2].set_title("own mutations in every sample: 2^30 bp, 2^15 curve, batch 32", fontsize = 9)
	for a in ax[:, 2]:
		a.set_xscale("log"); a.set_yscale("log"); a.grid(alpha = 0.3, which = "both")
	ax[1, 2].set_xlabel("mutations per sample")
	ax[0, 0].set_ylabel("forward + backward time (ms)")
	ax[1, 0].set_ylabel("peak memory (MiB)")
	ax[0, 0].legend(fontsize = 8); ax[0, 2].legend(fontsize = 8)
	fig.suptitle("Hilbert-curve network (SparseHilbertCNN layers) on synthetic genomes; NVIDIA L40S, fp32", fontsize = 10)
	fig.tight_layout()
	fig.savefig(os.path.join(DOCS, "benchmark_genomes.png"), dpi = 110)

if __name__ == "__main__":
	os.makedirs(DOCS, exist_ok = True)
	layers()
	genomes()
