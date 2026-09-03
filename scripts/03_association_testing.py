#!/usr/bin/env python3
"""
03_association_testing.py — Core Association Testing

Performs genome-wide association testing using logistic regression for a binary
phenotype (antibiotic resistance). Population structure is accounted for by including
principal components as covariates in the regression model.

Biological context:
    Bacterial GWAS typically tests for associations between core genome SNPs and
    phenotypes such as antibiotic resistance, virulence, or host range. Logistic
    regression models the log-odds of the phenotype as a function of genotype,
    optionally adjusting for population structure via PCs. Multiple testing
    correction is essential given the hundreds to thousands of tests performed.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path

np.random.seed(42)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

N_CCS = 3
SIGNIFICANCE_LEVEL = 0.05

COLORS = {
    "manhattan_sig": "#e74c3c",
    "manhattan_ns": "#2c3e50",
    "qq": "#3498db",
    "qq_line": "#e74c3c",
}


def load_data():
    """Load SNP matrix and PCA results from previous steps."""
    snp = pd.read_csv(OUTPUT_DIR / "snp_matrix.csv", index_col=0)
    snp = snp.fillna(0).astype(int)
    pca = pd.read_csv(OUTPUT_DIR / "pca_results.csv", index_col=0)
    return snp, pca


def simulate_phenotype(snp_matrix, n_causal=8):
    """
    Simulate a binary phenotype (antibiotic resistance) influenced by a subset
    of SNPs plus noise. Some causal SNPs are placed in clusters to simulate
    operonic or gene-cluster effects.
    """
    n_strains = len(snp_matrix)
    snp_cols = snp_matrix.columns.tolist()

    causal_indices = sorted(np.random.choice(len(snp_cols), size=n_causal, replace=False))

    log_odds = np.zeros(n_strains)
    for idx in causal_indices:
        effect = np.random.uniform(0.8, 2.0) * np.random.choice([-1, 1])
        log_odds += effect * snp_matrix.iloc[:, idx].values

    noise = np.random.normal(0, 1.5, size=n_strains)
    prob = 1 / (1 + np.exp(-(log_odds + noise)))
    phenotype = np.random.binomial(1, prob)

    causal_snps = [snp_cols[i] for i in causal_indices]
    return phenotype, causal_snps, prob


def run_logistic_gwas(snp_matrix, phenotype, covariates):
    """Run SNP-by-SNP logistic regression GWAS with covariates."""
    results = []
    y = phenotype

    for snp in snp_matrix.columns:
        X_snp = snp_matrix[snp].values.reshape(-1, 1)
        X = np.column_stack([X_snp, covariates])

        try:
            from statsmodels.api import add_constant, Logit
            X_with_const = add_constant(X)
            model = Logit(y, X_with_const)
            result = model.fit(disp=0, maxiter=50)

            beta = result.params[1]
            se = result.bse[1]
            z = result.tvalues[1]
            pval = result.pvalues[1]
            odds_ratio = np.exp(beta)

            results.append({
                "snp_id": snp,
                "beta": beta,
                "se": se,
                "z_score": z,
                "p_value": pval,
                "odds_ratio": odds_ratio,
            })
        except Exception:
            results.append({
                "snp_id": snp,
                "beta": np.nan,
                "se": np.nan,
                "z_score": np.nan,
                "p_value": 1.0,
                "odds_ratio": np.nan,
            })

    return pd.DataFrame(results)


def correct_multiple_testing(gwas_results):
    """Apply Bonferroni and FDR correction."""
    pvals = gwas_results["p_value"].values
    n_tests = len(pvals)

    bonf_reject, bonf_pvals, _, _ = multipletests(pvals, alpha=SIGNIFICANCE_LEVEL, method="bonferroni")
    fdr_reject, fdr_pvals, _, _ = multipletests(pvals, alpha=SIGNIFICANCE_LEVEL, method="fdr_bh")

    gwas_results["p_bonferroni"] = bonf_pvals
    gwas_results["p_fdr"] = fdr_pvals
    gwas_results["significant_bonferroni"] = bonf_reject
    gwas_results["significant_fdr"] = fdr_reject

    return gwas_results


def calculate_lambda_gc(pvalues):
    """Calculate genomic inflation factor (lambda GC)."""
    median_obs = np.median(stats.chi2.isf(pvalues, df=1))
    median_expected = stats.chi2.isf(0.5, df=1)
    return median_obs / median_expected


def plot_manhattan(gwas_results, causal_snps):
    """Generate a Manhattan plot of GWAS results."""
    df = gwas_results.copy()
    df["-log10p"] = -np.log10(df["p_value"].clip(lower=1e-300))

    chrom_names = sorted(df["snp_id"].apply(lambda x: "_".join(x.split("_")[:2])).unique())
    chrom_map = {c: i for i, c in enumerate(chrom_names)}
    df["chrom_idx"] = df["snp_id"].apply(lambda x: chrom_map["_".join(x.split("_")[:2])])

    positions = []
    for snp in df["snp_id"]:
        parts = snp.split("_")
        positions.append(int(parts[2]) if len(parts) > 2 else 0)
    df["position"] = positions

    df = df.sort_values(["chrom_idx", "position"])

    cum_pos = 0
    tick_positions = []
    tick_labels = []
    for chrom_idx in sorted(df["chrom_idx"].unique()):
        chrom_data = df[df["chrom_idx"] == chrom_idx]
        mid = chrom_data["position"].median() + cum_pos
        tick_positions.append(mid)
        tick_labels.append(f"Chr {chrom_idx + 1}")
        df.loc[df["chrom_idx"] == chrom_idx, "position"] += cum_pos
        cum_pos = chrom_data["position"].max() + 50_000_000

    n_bonf = df["significant_bonferroni"].sum()
    n_fdr = df["significant_fdr"].sum()
    threshold_line = -np.log10(0.05 / len(df))

    fig, ax = plt.subplots(figsize=(16, 6))

    colors_alt = [COLORS["manhattan_ns"], "#7f8c8d"]
    for chrom_idx in sorted(df["chrom_idx"].unique()):
        mask = df["chrom_idx"] == chrom_idx
        color = colors_alt[chrom_idx % 2]
        sig_mask = mask & (df["significant_bonferroni"] | df["significant_fdr"])
        ns_mask = mask & ~sig_mask

        ax.scatter(df.loc[ns_mask, "position"], df.loc[ns_mask, "-log10p"],
                   c=color, s=8, alpha=0.5, rasterized=True)
        ax.scatter(df.loc[sig_mask, "position"], df.loc[sig_mask, "-log10p"],
                   c=COLORS["manhattan_sig"], s=12, alpha=0.8, rasterized=True,
                   edgecolors="white", linewidths=0.3, zorder=3)

    ax.axhline(y=threshold_line, color="#e67e22", linestyle="--", linewidth=1.2,
               label=f"Bonferroni threshold (p = {0.05/len(df):.2e})")

    for snp in causal_snps:
        if snp in df["snp_id"].values:
            row = df[df["snp_id"] == snp].iloc[0]
            ax.annotate(snp, (row["position"], row["-log10p"]),
                        fontsize=6, color="#e74c3c", fontweight="bold",
                        xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Genomic Position", fontsize=12)
    ax.set_ylabel("-log₁₀(p-value)", fontsize=12)
    ax.set_title("Manhattan Plot — Bacterial GWAS (Antibiotic Resistance)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=10)

    stats_text = (f"Bonferroni significant: {n_bonf}\n"
                  f"FDR significant: {n_fdr}\n"
                  f"λ_GC = {calculate_lambda_gc(gwas_results['p_value'].values):.3f}")
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))

    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_manhattan_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_qq(gwas_results):
    """Generate a QQ plot comparing observed vs expected p-values."""
    pvals = gwas_results["p_value"].dropna().sort_values().values
    n = len(pvals)
    expected = -np.log10(np.linspace(1 / n, 1 - 1 / n, n))
    observed = -np.log10(pvals)

    lambda_gc = calculate_lambda_gc(pvals)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(expected, observed, c=COLORS["qq"], s=15, alpha=0.7, edgecolors="white", linewidths=0.3)
    max_val = max(expected.max(), observed.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "--", color=COLORS["qq_line"], linewidth=1.5, label="y = x")

    ax.set_xlabel("Expected -log₁₀(p-value)", fontsize=12)
    ax.set_ylabel("Observed -log₁₀(p-value)", fontsize=12)
    ax.set_title("Q-Q Plot — GWAS p-values", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    ax.text(0.05, 0.92, f"λ_GC = {lambda_gc:.3f}", transform=ax.transAxes, fontsize=11,
            fontweight="bold", color="#e74c3c" if lambda_gc > 1.1 else "#27ae60",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_qq_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pvalue_distribution(gwas_results):
    """Plot histogram of p-value distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gwas_results["p_value"].dropna(), bins=50, color="#2c3e50", edgecolor="white", alpha=0.85)
    ax.axvline(x=0.05 / len(gwas_results), color="#e74c3c", linestyle="--", linewidth=1.2,
               label=f"Bonferroni = {0.05/len(gwas_results):.2e}")
    ax.set_xlabel("p-value", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of GWAS p-values", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_pvalue_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top_snps_forest(gwas_results, causal_snps, top_n=10):
    """Forest plot of odds ratios for top associated SNPs."""
    top = gwas_results.nsmallest(top_n, "p_value").copy()
    top["is_causal"] = top["snp_id"].isin(causal_snps)

    fig, ax = plt.subplots(figsize=(10, 7))

    y_pos = range(len(top))

    for i, (_, row) in enumerate(top.iterrows()):
        color = COLORS["manhattan_sig"] if row["is_causal"] else COLORS["manhattan_ns"]
        ci = 1.96 * row["se"] * row["odds_ratio"]
        ax.errorbar(row["odds_ratio"], i, xerr=ci,
                    fmt="o", color=color, ecolor=color, elinewidth=1.5, capsize=4,
                    markersize=7, markeredgecolor="white", markeredgewidth=0.5)

    ax.axvline(x=1, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7, label="OR = 1 (no effect)")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(top["snp_id"].values, fontsize=9)
    ax.set_xlabel("Odds Ratio (95% CI)", fontsize=12)
    ax.set_title("Top Associated SNPs — Odds Ratios", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["manhattan_sig"],
               markersize=8, label="Causal SNP"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["manhattan_ns"],
               markersize=8, label="Non-causal SNP"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
               markersize=8, label="OR = 1"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_top_snps_forest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 60)
    print("STEP 3: Association Testing")
    print("=" * 60)

    print("\nLoading data...")
    snp_matrix, pca = load_data()
    print(f"  SNP matrix: {snp_matrix.shape[0]} strains × {snp_matrix.shape[1]} SNPs")

    print("Simulating binary phenotype (antibiotic resistance)...")
    phenotype, causal_snps, probs = simulate_phenotype(snp_matrix)
    print(f"  Prevalence: {phenotype.mean():.2%}")
    print(f"  Causal SNPs: {len(causal_snps)}")

    print("Running logistic regression GWAS with population structure covariates...")
    covariates = pca.iloc[:, :N_CCS].values
    gwas_results = run_logistic_gwas(snp_matrix, phenotype, covariates)

    print("Applying multiple testing correction...")
    gwas_results = correct_multiple_testing(gwas_results)

    n_bonf = gwas_results["significant_bonferroni"].sum()
    n_fdr = gwas_results["significant_fdr"].sum()
    lambda_gc = calculate_lambda_gc(gwas_results["p_value"].values)

    print(f"  Bonferroni significant: {n_bonf}")
    print(f"  FDR significant: {n_fdr}")
    print(f"  λ_GC = {lambda_gc:.3f}")

    top10 = gwas_results.nsmallest(10, "p_value")
    print("\n  Top 10 associated SNPs:")
    for _, row in top10.iterrows():
        causal_flag = " *" if row["snp_id"] in causal_snps else ""
        print(f"    {row['snp_id']}: p={row['p_value']:.2e}, OR={row['odds_ratio']:.2f}{causal_flag}")

    gwas_results.to_csv(OUTPUT_DIR / "gwas_results.csv", index=False)
    causal_df = pd.DataFrame({"causal_snp": causal_snps})
    causal_df.to_csv(OUTPUT_DIR / "03_causal_snps.csv", index=False)

    print("\nGenerating plots...")
    plot_manhattan(gwas_results, causal_snps)
    plot_qq(gwas_results)
    plot_pvalue_distribution(gwas_results)
    plot_top_snps_forest(gwas_results, causal_snps)

    print(f"\nOutputs saved to {OUTPUT_DIR}/")
    print("  - gwas_results.csv")
    print("  - 03_causal_snps.csv")
    print("  - 03_manhattan_plot.png")
    print("  - 03_qq_plot.png")
    print("  - 03_pvalue_distribution.png")
    print("  - 03_top_snps_forest.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
