#!/usr/bin/env python3
"""
01_snp_matrix_construction.py — Variant Calling & SNP Matrix Construction

Simulates bacterial genome data for 100 strains across a core genome of ~1000 SNPs.
Constructs a binary SNP matrix (presence/absence of the minor allele) and applies
quality filters based on minor allele frequency and missing data thresholds.

Biological context:
    In bacterial GWAS, the first step is constructing a high-quality SNP matrix from
    whole-genome alignments or pangenome analyses. Core genome SNPs — those present
    in all (or nearly all) strains — provide the most informative sites for association
    testing. Quality filtering removes rare variants and poorly-called sites that can
    inflate association statistics.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

np.random.seed(42)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

N_STRAINS = 100
N_SNPS = 1000
N_GENES = 500

STRAIN_NAMES = [f"strain_{i:03d}" for i in range(1, N_STRAINS + 1)]

CHROMOSOME_NAMES = ["chromosome", "plasmid_1", "plasmid_2"]
CHROM_SIZES = [800, 150, 50]


def simulate_bacterial_genomes():
    """
    Simulate SNP data for bacterial strains.

    Each SNP is a biallelic site with an ancestral and derived allele.
    Allele frequencies are drawn from a beta distribution to create a realistic
    distribution of rare and common variants.
    """
    snp_ids = []
    positions = []
    chromosomes = []

    snp_idx = 0
    for chrom, size in zip(CHROMOSOME_NAMES, CHROM_SIZES):
        for _ in range(size):
            snp_ids.append(f"SNP_{snp_idx:05d}")
            positions.append(np.random.randint(1, 1_000_000))
            chromosomes.append(chrom)
            snp_idx += 1

    allele_freqs = np.random.beta(0.5, 0.5, size=N_SNPS)

    snp_matrix = np.zeros((N_STRAINS, N_SNPS), dtype=int)
    for j in range(N_SNPS):
        p = allele_freqs[j]
        snp_matrix[:, j] = np.random.binomial(1, p, size=N_STRAINS)

    missing_rate = 0.02
    missing_mask = np.random.random((N_STRAINS, N_SNPS)) < missing_rate
    snp_matrix = snp_matrix.astype(float)
    snp_matrix[missing_mask] = np.nan

    df = pd.DataFrame(snp_matrix, index=STRAIN_NAMES, columns=snp_ids)
    meta = pd.DataFrame({
        "snp_id": snp_ids,
        "chromosome": chromosomes,
        "position": positions,
        "ancestral_allele": ["A"] * N_SNPS,
        "derived_allele": ["G"] * N_SNPS,
    })

    return df, meta, allele_freqs


def calculate_summary_statistics(snp_matrix):
    """Calculate per-SNP summary statistics before and after filtering."""
    n_strains, n_snps = snp_matrix.shape
    stats = []

    for snp in snp_matrix.columns:
        calls = snp_matrix[snp].dropna()
        n_called = len(calls)
        n_missing = n_strains - n_called
        maf = calls.mean()
        if maf > 0.5:
            maf = 1 - maf
        stats.append({
            "snp_id": snp,
            "n_called": n_called,
            "n_missing": n_missing,
            "missing_rate": n_missing / n_strains,
            "allele_freq": calls.mean(),
            "maf": maf,
        })

    return pd.DataFrame(stats)


def apply_quality_filters(snp_matrix, stats, maf_threshold=0.05, missing_threshold=0.10):
    """Filter SNPs by MAF and missing data rate."""
    pass_maf = stats["maf"] >= maf_threshold
    pass_missing = stats["missing_rate"] <= missing_threshold
    passing = stats[pass_maf & pass_missing]["snp_id"].tolist()

    filtered = snp_matrix[passing].copy()
    return filtered, passing, stats[pass_maf & pass_missing]


def plot_allele_frequency_distribution(stats, stats_filtered):
    """Plot minor allele frequency distributions before and after filtering."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(stats["maf"], bins=50, color="#2c3e50", edgecolor="white", alpha=0.85)
    axes[0].axvline(x=0.05, color="#e74c3c", linestyle="--", linewidth=1.5, label="MAF = 0.05")
    axes[0].set_xlabel("Minor Allele Frequency", fontsize=12)
    axes[0].set_ylabel("Number of SNPs", fontsize=12)
    axes[0].set_title("MAF Distribution (All SNPs)", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)

    axes[1].hist(stats_filtered["maf"], bins=50, color="#2980b9", edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("Minor Allele Frequency", fontsize=12)
    axes[1].set_ylabel("Number of SNPs", fontsize=12)
    axes[1].set_title("MAF Distribution (Filtered SNPs)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_maf_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_snp_matrix_heatmap(snp_matrix):
    """Heatmap of the filtered SNP matrix for a subset of strains and SNPs."""
    subset_strains = snp_matrix.index[:25]
    subset_snps = snp_matrix.columns[:50]
    subset = snp_matrix.loc[subset_strains, subset_snps]

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        subset, cmap="RdYlBu_r", center=0.5,
        xticklabels=False, yticklabels=True,
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Minor Allele (0/1)"},
        ax=ax,
    )
    ax.set_title("Filtered SNP Matrix (subset: 25 strains × 50 SNPs)", fontsize=13, fontweight="bold")
    ax.set_xlabel("SNP Loci", fontsize=12)
    ax.set_ylabel("Strain", fontsize=12)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_snp_matrix_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_missing_data(stats):
    """Plot missing data rate per SNP."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(stats)), stats["missing_rate"].sort_values().values, color="#7f8c8d", width=1.0)
    ax.axhline(y=0.10, color="#e74c3c", linestyle="--", linewidth=1.5, label="Missing = 10%")
    ax.set_xlabel("SNP Index (sorted)", fontsize=12)
    ax.set_ylabel("Missing Data Rate", fontsize=12)
    ax.set_title("Missing Data Rate per SNP", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(stats["missing_rate"]) * 1.2)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_missing_data_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_genomic_distribution(meta, stats):
    """Plot SNP density across chromosomes."""
    merged = meta.merge(stats, on="snp_id")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["#2c3e50", "#27ae60", "#8e44ad"]

    for i, chrom in enumerate(CHROMOSOME_NAMES):
        chrom_data = merged[merged["chromosome"] == chrom].sort_values("position")
        axes[i].hist(chrom_data["position"], bins=30, color=colors[i], edgecolor="white", alpha=0.85)
        axes[i].set_title(chrom.replace("_", " ").title(), fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Genomic Position (bp)", fontsize=10)
        axes[i].set_ylabel("SNP Count", fontsize=10)

    fig.suptitle("Genomic Distribution of SNPs", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_genomic_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 60)
    print("STEP 1: SNP Matrix Construction")
    print("=" * 60)

    print(f"\nSimulating {N_STRAINS} bacterial strains × {N_SNPS} core genome SNPs...")
    snp_matrix, meta, true_freqs = simulate_bacterial_genomes()
    print(f"  Raw SNP matrix: {snp_matrix.shape[0]} strains × {snp_matrix.shape[1]} SNPs")

    print("Calculating summary statistics...")
    stats = calculate_summary_statistics(snp_matrix)
    print(f"  Mean MAF: {stats['maf'].mean():.3f}")
    print(f"  Mean missing rate: {stats['missing_rate'].mean():.3f}")

    print("Applying quality filters (MAF > 0.05, missing < 10%)...")
    filtered, passing_snps, stats_filtered = apply_quality_filters(snp_matrix, stats)
    print(f"  SNPs passing filters: {filtered.shape[1]} / {N_SNPS} ({filtered.shape[1]/N_SNPS*100:.1f}%)")

    filtered.to_csv(OUTPUT_DIR / "snp_matrix.csv")
    stats.to_csv(OUTPUT_DIR / "01_snp_statistics_unfiltered.csv", index=False)
    stats_filtered.to_csv(OUTPUT_DIR / "01_snp_statistics_filtered.csv", index=False)
    meta.to_csv(OUTPUT_DIR / "01_snp_metadata.csv", index=False)

    print("\nGenerating plots...")
    plot_allele_frequency_distribution(stats, stats_filtered)
    plot_snp_matrix_heatmap(filtered)
    plot_missing_data(stats)
    plot_genomic_distribution(meta, stats)

    print(f"\nOutputs saved to {OUTPUT_DIR}/")
    print("  - snp_matrix.csv")
    print("  - 01_snp_statistics_unfiltered.csv")
    print("  - 01_snp_statistics_filtered.csv")
    print("  - 01_snp_metadata.csv")
    print("  - 01_maf_distribution.png")
    print("  - 01_snp_matrix_heatmap.png")
    print("  - 01_missing_data_rate.png")
    print("  - 01_genomic_distribution.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
