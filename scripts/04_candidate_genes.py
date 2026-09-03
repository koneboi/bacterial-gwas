#!/usr/bin/env python3
"""
04_candidate_genes.py — Candidate Gene Identification & Visualization

From GWAS results, identifies top candidate regions, performs simulated gene
enrichment analysis, computes odds ratios and effect sizes, and analyzes
linkage disequilibrium (LD)-like correlations between top SNPs.

Biological context:
    After genome-wide association testing, the next step is translating statistical
    signals into biological insights. This involves mapping associated SNPs to genes,
    identifying enriched functional categories, and examining the correlation structure
    around top hits. In bacteria, operons and gene clusters often co-vary, so
    multiple associated SNPs in the same region may point to a single causal gene.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

np.random.seed(42)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

N_GENES = 200

COLORS = {
    "primary": "#2c3e50",
    "accent": "#e74c3c",
    "secondary": "#3498db",
    "success": "#27ae61",
    "warning": "#f39c12",
    "purple": "#9b59b6",
}

GENE_CATEGORIES = [
    "Efflux pump", "Target modification", "Enzyme degradation",
    "Membrane permeability", "Biofilm formation", "Stress response",
    "Metabolism", "Regulatory", "DNA repair", "Cell wall synthesis",
]


def load_gwas_results():
    """Load GWAS results and SNP matrix."""
    gwas = pd.read_csv(OUTPUT_DIR / "gwas_results.csv")
    snp = pd.read_csv(OUTPUT_DIR / "snp_matrix.csv", index_col=0).fillna(0).astype(int)
    meta = pd.read_csv(OUTPUT_DIR / "01_snp_metadata.csv")
    return gwas, snp, meta


def simulate_genes(meta):
    """Simulate gene annotations for SNPs."""
    genes = []
    for i in range(N_GENES):
        chrom = meta["chromosome"].iloc[min(i * (len(meta) // N_GENES), len(meta) - 1)]
        genes.append({
            "gene_id": f"GENE_{i:04d}",
            "gene_name": f"b0{i:03d}",
            "chromosome": chrom,
            "start": np.random.randint(1, 900_000),
            "end": 0,
            "category": np.random.choice(GENE_CATEGORIES),
            "description": f"Hypothetical protein {i}",
            "n_snps": 0,
        })
        genes[-1]["end"] = genes[-1]["start"] + np.random.randint(500, 5000)
    return pd.DataFrame(genes)


def map_snps_to_genes(gwas, meta, genes):
    """Map SNPs to genes based on genomic proximity."""
    gene_snp_map = {}
    for _, gene in genes.iterrows():
        gene_snp_map[gene["gene_id"]] = []

    for _, snp_row in meta.iterrows():
        snp_pos = snp_row["position"]
        snp_chrom = snp_row["chromosome"]
        best_gene = None
        best_dist = float("inf")

        for _, gene in genes.iterrows():
            if gene["chromosome"] != snp_chrom:
                continue
            midpoint = (gene["start"] + gene["end"]) / 2
            dist = abs(snp_pos - midpoint)
            if dist < best_dist:
                best_dist = dist
                best_gene = gene["gene_id"]

        if best_gene and best_dist < 100_000:
            gene_snp_map[best_gene].append(snp_row["snp_id"])

    for gene_id, snp_list in gene_snp_map.items():
        genes.loc[genes["gene_id"] == gene_id, "n_snps"] = len(snp_list)

    return genes


def identify_candidate_regions(gwas, meta, top_n=50):
    """Identify top candidate genomic regions from GWAS results."""
    top_snps = gwas.nsmallest(top_n, "p_value").copy()
    top_snps = top_snps.merge(meta, on="snp_id")
    top_snps = top_snps.sort_values(["chromosome", "position"])

    top_snps["is_causal"] = top_snps["snp_id"].isin(
        pd.read_csv(OUTPUT_DIR / "03_causal_snps.csv")["causal_snp"]
    )

    return top_snps


def calculate_gene_enrichment(candidates, genes):
    """Calculate enrichment of gene categories among top candidates."""
    from collections import Counter

    candidate_genes = genes[genes["n_snps"] > 0]

    cat_counts = Counter(candidate_genes["category"])
    all_counts = Counter(genes["category"])

    enrichment = []
    total_candidate = len(candidate_genes)
    total_genes = len(genes)

    for cat in GENE_CATEGORIES:
        k = cat_counts.get(cat, 0)
        n = total_candidate
        K = all_counts.get(cat, 0)
        N = total_genes

        if N > 0 and K > 0 and n > 0:
            fold_enrichment = (k / n) / (K / N) if K > 0 else 0
            fisher_p = stats.fisher_exact([
                [k, n - k],
                [K - k, N - K + k]
            ])[1] if (n - k) >= 0 and (N - K + k) >= 0 else 1.0
        else:
            fold_enrichment = 0
            fisher_p = 1.0

        enrichment.append({
            "category": cat,
            "n_candidate": k,
            "n_background": K,
            "fold_enrichment": fold_enrichment,
            "p_value": fisher_p,
        })

    return pd.DataFrame(enrichment).sort_values("p_value")


def calculate_ld_matrix(snp_matrix, snp_list):
    """Calculate LD-like (correlation) matrix between top SNPs."""
    subset = snp_matrix[snp_list]
    ld_matrix = subset.corr()
    return ld_matrix


def calculate_effect_sizes(gwas, snp_matrix, candidates):
    """Calculate detailed effect sizes for top SNPs."""
    results = []
    for _, row in candidates.iterrows():
        snp = row["snp_id"]
        if snp not in snp_matrix.columns:
            continue

        allele = snp_matrix[snp].values
        phenotype = pd.read_csv(OUTPUT_DIR / "gwas_results.csv")  # for reference

        freq_0 = (allele == 0).sum()
        freq_1 = (allele == 1).sum()

        results.append({
            "snp_id": snp,
            "beta": row["beta"],
            "odds_ratio": row["odds_ratio"],
            "p_value": row["p_value"],
            "minor_allele_freq": min(freq_0, freq_1) / len(allele),
            "n_strains_with_allele": int(freq_1),
            "is_causal": row["is_causal"],
        })

    return pd.DataFrame(results)


def plot_candidate_region(candidates, meta):
    """Plot association signal across a candidate region."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})

    all_snps = meta.copy()
    all_snps = all_snps.merge(
        pd.read_csv(OUTPUT_DIR / "gwas_results.csv")[["snp_id", "p_value", "beta", "odds_ratio"]],
        on="snp_id"
    )
    all_snps["-log10p"] = -np.log10(all_snps["p_value"].clip(lower=1e-300))

    for chrom in all_snps["chromosome"].unique():
        chrom_data = all_snps[all_snps["chromosome"] == chrom].sort_values("position")
        axes[0].scatter(chrom_data["position"], chrom_data["-log10p"],
                        s=10, alpha=0.5, label=chrom)

    causal_snps = pd.read_csv(OUTPUT_DIR / "03_causal_snps.csv")["causal_snp"].tolist()
    causal_data = all_snps[all_snps["snp_id"].isin(causal_snps)]
    axes[0].scatter(causal_data["position"], causal_data["-log10p"],
                    s=60, c=COLORS["accent"], marker="*", zorder=5, label="Causal SNP", edgecolors="white")

    axes[0].set_ylabel("-log₁₀(p-value)", fontsize=12)
    axes[0].set_title("Genome-wide Association Signal with Causal SNPs Highlighted",
                      fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].grid(True, alpha=0.3)

    gene_cats = candidates.merge(
        simulate_genes(meta)[["gene_id", "category", "start", "end", "chromosome"]],
        left_on="snp_id", right_on="gene_id", how="left"
    )

    gene_df = simulate_genes(meta)
    cat_colors = sns.color_palette("Set2", len(GENE_CATEGORIES))
    cat_map = dict(zip(GENE_CATEGORIES, cat_colors))

    for i, gene in gene_df.head(30).iterrows():
        color = cat_map.get(gene["category"], "#999999")
        axes[1].barh(i % 15, gene["end"] - gene["start"], left=gene["start"],
                     height=0.6, color=color, alpha=0.7, edgecolor="white")
        axes[1].text(gene["start"] + 100, i % 15, gene["gene_name"],
                     fontsize=6, va="center")

    axes[1].set_xlabel("Genomic Position (bp)", fontsize=12)
    axes[1].set_ylabel("Gene", fontsize=12)
    axes[1].set_title("Gene Annotations (first chromosome)", fontsize=11, fontweight="bold")
    axes[1].set_yticks([])

    handles = [plt.Rectangle((0, 0), 1, 1, color=cat_map[c]) for c in list(cat_map.keys())[:6]]
    axes[1].legend(handles, list(cat_map.keys())[:6], fontsize=7, loc="upper right", ncol=2)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_candidate_regions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_enrichment(enrichment):
    """Plot gene category enrichment analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))

    enrichment = enrichment.sort_values("fold_enrichment", ascending=True)
    colors = [COLORS["accent"] if p < 0.05 else COLORS["primary"]
              for p in enrichment["p_value"]]

    ax.barh(enrichment["category"], enrichment["fold_enrichment"], color=colors, alpha=0.85)
    ax.axvline(x=1, color="#7f8c8d", linestyle="--", linewidth=1, label="No enrichment")

    for i, (_, row) in enumerate(enrichment.iterrows()):
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        if sig:
            ax.text(row["fold_enrichment"] + 0.05, i, sig, fontsize=10, va="center", color=COLORS["accent"])

    ax.set_xlabel("Fold Enrichment", fontsize=12)
    ax.set_title("Gene Category Enrichment Among GWAS Candidates",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_enrichment_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ld_heatmap(ld_matrix):
    """Plot LD-like correlation heatmap between top SNPs."""
    n = min(20, len(ld_matrix))
    subset = ld_matrix.iloc[:n, :n]

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.zeros_like(subset, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    sns.heatmap(subset, mask=mask, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, annot=True, fmt=".2f",
                square=True, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Correlation (r)"}, ax=ax)

    ax.set_title("LD-like Correlation Structure Among Top Associated SNPs",
                 fontsize=13, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_ld_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_effect_size_distribution(effect_sizes):
    """Plot distribution of effect sizes (odds ratios) for top SNPs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    causal = effect_sizes[effect_sizes["is_causal"]]
    non_causal = effect_sizes[~effect_sizes["is_causal"]]

    axes[0].hist(effect_sizes["odds_ratio"], bins=20, color=COLORS["primary"],
                 edgecolor="white", alpha=0.85, label="All top SNPs")
    if len(causal) > 0:
        axes[0].hist(causal["odds_ratio"], bins=10, color=COLORS["accent"],
                     edgecolor="white", alpha=0.6, label="Causal SNPs")
    axes[0].axvline(x=1, color=COLORS["warning"], linestyle="--", linewidth=1.2, label="OR = 1")
    axes[0].set_xlabel("Odds Ratio", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("Effect Size Distribution", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=9)

    axes[1].scatter(effect_sizes["minor_allele_freq"], -np.log10(effect_sizes["p_value"].clip(lower=1e-300)),
                    c=[COLORS["accent"] if c else COLORS["primary"] for c in effect_sizes["is_causal"]],
                    s=60, alpha=0.8, edgecolors="white", linewidths=0.5)
    axes[1].set_xlabel("Minor Allele Frequency", fontsize=12)
    axes[1].set_ylabel("-log₁₀(p-value)", fontsize=12)
    axes[1].set_title("MAF vs Association Strength", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["accent"],
               markersize=8, label="Causal"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["primary"],
               markersize=8, label="Non-causal"),
    ]
    axes[1].legend(handles=legend_elements, fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_effect_sizes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_summary_report(gwas, candidates, effect_sizes, enrichment, lambda_gc):
    """Generate a text summary report."""
    n_causal_found = candidates["is_causal"].sum()
    n_causal_total = len(pd.read_csv(OUTPUT_DIR / "03_causal_snps.csv"))

    report = []
    report.append("=" * 60)
    report.append("BACTERIAL GWAS — CANDIDATE GENE SUMMARY REPORT")
    report.append("=" * 60)
    report.append("")
    report.append(f"Total SNPs tested:              {len(gwas)}")
    report.append(f"Genomic inflation (λ_GC):       {lambda_gc:.3f}")
    report.append(f"Bonferroni significant:         {gwas['significant_bonferroni'].sum()}")
    report.append(f"FDR significant:                {gwas['significant_fdr'].sum()}")
    report.append("")
    report.append(f"Top candidate SNPs:             {len(candidates)}")
    report.append(f"Causal SNPs recovered:          {n_causal_found} / {n_causal_total} "
                 f"({n_causal_found / n_causal_total * 100:.0f}%)")
    report.append("")

    report.append("Top 5 Associated SNPs:")
    report.append("-" * 50)
    top5 = gwas.nsmallest(5, "p_value")
    for _, row in top5.iterrows():
        causal = " [CAUSAL]" if row["snp_id"] in candidates["snp_id"].values else ""
        report.append(f"  {row['snp_id']:>20s}  p={row['p_value']:.2e}  OR={row['odds_ratio']:.2f}{causal}")

    report.append("")
    report.append("Enriched Gene Categories (p < 0.05):")
    report.append("-" * 50)
    sig_enrich = enrichment[enrichment["p_value"] < 0.05]
    if len(sig_enrich) > 0:
        for _, row in sig_enrich.iterrows():
            report.append(f"  {row['category']:<30s}  fold={row['fold_enrichment']:.2f}  p={row['p_value']:.4f}")
    else:
        report.append("  No significantly enriched categories")

    report.append("")
    report.append("=" * 60)

    report_text = "\n".join(report)
    with open(OUTPUT_DIR / "04_summary_report.txt", "w") as f:
        f.write(report_text)

    return report_text


def main():
    print("=" * 60)
    print("STEP 4: Candidate Gene Identification")
    print("=" * 60)

    print("\nLoading data...")
    gwas, snp_matrix, meta = load_gwas_results()
    print(f"  GWAS results: {len(gwas)} SNPs tested")

    print("Simulating gene annotations...")
    genes = simulate_genes(meta)
    print(f"  {len(genes)} genes annotated across {genes['chromosome'].nunique()} replicons")

    print("Mapping SNPs to genes...")
    genes = map_snps_to_genes(gwas, meta, genes)
    genes_with_snps = genes[genes["n_snps"] > 0]
    print(f"  {len(genes_with_snps)} genes with mapped SNPs")

    print("Identifying candidate regions...")
    candidates = identify_candidate_regions(gwas, meta, top_n=50)
    print(f"  {len(candidates)} top candidate SNPs")
    print(f"  Causal SNPs found: {candidates['is_causal'].sum()} / "
          f"{len(pd.read_csv(OUTPUT_DIR / '03_causal_snps.csv'))}")

    print("Calculating gene enrichment...")
    enrichment = calculate_gene_enrichment(candidates, genes)
    sig_enrich = enrichment[enrichment["p_value"] < 0.05]
    print(f"  {len(sig_enrich)} significantly enriched categories")

    print("Calculating effect sizes...")
    effect_sizes = calculate_effect_sizes(gwas, snp_matrix, candidates)
    print(f"  Mean OR (top SNPs): {effect_sizes['odds_ratio'].mean():.2f}")

    print("Computing LD-like correlations...")
    top_snps = candidates["snp_id"].tolist()[:20]
    ld_matrix = calculate_ld_matrix(snp_matrix, top_snps)

    lambda_gc = gwas["p_value"].apply(lambda x: -np.log10(x)).median() / (-np.log10(0.5))
    lambda_gc = np.median(stats.chi2.isf(gwas["p_value"].values, df=1)) / stats.chi2.isf(0.5, df=1)

    candidates.to_csv(OUTPUT_DIR / "candidate_genes.csv", index=False)
    enrichment.to_csv(OUTPUT_DIR / "04_enrichment_results.csv", index=False)
    effect_sizes.to_csv(OUTPUT_DIR / "04_effect_sizes.csv", index=False)
    ld_matrix.to_csv(OUTPUT_DIR / "04_ld_matrix.csv")

    print("\nGenerating plots...")
    plot_candidate_region(candidates, meta)
    plot_enrichment(enrichment)
    plot_ld_heatmap(ld_matrix)
    plot_effect_size_distribution(effect_sizes)

    print("\nGenerating summary report...")
    report = generate_summary_report(gwas, candidates, effect_sizes, enrichment, lambda_gc)
    print(report)

    print(f"\nOutputs saved to {OUTPUT_DIR}/")
    print("  - candidate_genes.csv")
    print("  - 04_enrichment_results.csv")
    print("  - 04_effect_sizes.csv")
    print("  - 04_ld_matrix.csv")
    print("  - 04_candidate_regions.png")
    print("  - 04_enrichment_analysis.png")
    print("  - 04_ld_heatmap.png")
    print("  - 04_effect_sizes.png")
    print("  - 04_summary_report.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
