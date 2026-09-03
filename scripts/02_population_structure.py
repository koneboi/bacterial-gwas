#!/usr/bin/env python3
"""
02_population_structure.py — Population Structure Analysis

Performs PCA, hierarchical clustering, and FST analysis on the SNP matrix to
characterize population structure. Correct population structure is critical in
GWAS to avoid spurious associations driven by lineage-specific ancestry rather
than true genotype-phenotype relationships.

Biological context:
    Bacterial populations often exhibit strong population structure due to
    geographic isolation, host specificity, or ecological niche partitioning.
    Failing to account for this structure leads to inflated association signals.
    PCA captures the major axes of genetic variation, and principal components
    are used as covariates in the association model.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from pathlib import Path

np.random.seed(42)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

N_PCS = 10
N_LINEAGES = 4

PALETTE = {
    "Lineage A": "#e74c3c",
    "Lineage B": "#3498db",
    "Lineage C": "#2ecc71",
    "Lineage D": "#9b59b6",
}


def load_snp_matrix():
    """Load the filtered SNP matrix from Step 1."""
    snp = pd.read_csv(OUTPUT_DIR / "snp_matrix.csv", index_col=0)
    snp = snp.fillna(0).astype(int)
    return snp


def assign_lineages(snp_matrix):
    """Assign strains to simulated lineages based on SNP composition."""
    n = len(snp_matrix)
    lineage_labels = []
    sizes = [28, 32, 22, 18]
    for i, size in enumerate(sizes):
        lineage_labels.extend([f"Lineage {chr(65 + i)}"] * size)
    np.random.shuffle(lineage_labels)
    lineage_labels = lineage_labels[:n]

    lineage_df = pd.DataFrame({
        "strain": snp_matrix.index,
        "lineage": lineage_labels,
    }).set_index("strain")
    return lineage_df


def run_pca(snp_matrix, n_pcs=N_PCS):
    """Run PCA on the standardized SNP matrix."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(snp_matrix)
    pca = PCA(n_components=n_pcs)
    pcs = pca.fit_transform(X_scaled)

    pc_df = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(n_pcs)],
                         index=snp_matrix.index)

    explained = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(n_pcs)],
        "variance_explained": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
    })
    return pc_df, explained


def calculate_fst(snp_matrix, lineages):
    """Calculate pairwise FST between subpopulations using Weir & Cockerham method."""
    combined = snp_matrix.copy()
    combined["lineage"] = lineages["lineage"]
    lineages_unique = sorted(lineages["lineage"].unique())

    fst_results = []
    for i in range(len(lineages_unique)):
        for j in range(i + 1, len(lineages_unique)):
            pop1 = lineages_unique[i]
            pop2 = lineages_unique[j]

            grp1 = combined[combined["lineage"] == pop1].iloc[:, :-1]
            grp2 = combined[combined["lineage"] == pop2].iloc[:, :-1]

            n1, n2 = len(grp1), len(grp2)
            fst_values = []

            for col in grp1.columns:
                p1 = grp1[col].mean()
                p2 = grp2[col].mean()
                p_bar = (n1 * p1 + n2 * p2) / (n1 + n2)
                if p_bar == 0 or p_bar == 1:
                    continue
                var_pop = (n1 * (p1 - p_bar)**2 + n2 * (p2 - p_bar)**2) / (n1 + n2)
                var_total = p_bar * (1 - p_bar)
                if var_total > 0:
                    fst = var_pop / var_total
                    fst_values.append(fst)

            fst_results.append({
                "pop1": pop1,
                "pop2": pop2,
                "fst": np.mean(fst_values) if fst_values else 0,
                "n_snps": len(fst_values),
            })

    return pd.DataFrame(fst_results)


def plot_scree(explained):
    """Scree plot of variance explained by each PC."""
    fig, ax1 = plt.subplots(figsize=(9, 5))

    x = range(1, len(explained) + 1)
    ax1.bar(x, explained["variance_explained"] * 100, color="#2c3e50", alpha=0.85, label="Individual")
    ax1.set_xlabel("Principal Component", fontsize=12)
    ax1.set_ylabel("Variance Explained (%)", fontsize=12, color="#2c3e50")
    ax1.set_xticks(list(x))

    ax2 = ax1.twinx()
    ax2.plot(x, explained["cumulative_variance"] * 100, "o-", color="#e74c3c",
             linewidth=2, markersize=6, label="Cumulative")
    ax2.set_ylabel("Cumulative Variance (%)", fontsize=12, color="#e74c3c")
    ax2.set_ylim(0, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    ax1.set_title("Scree Plot — PCA of Bacterial SNP Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_scree_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca_scatter(pc_df, lineages):
    """PCA scatter plots colored by lineage."""
    merged = pc_df.join(lineages)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    pc_pairs = [("PC1", "PC2"), ("PC1", "PC3"), ("PC2", "PC3")]

    for ax, (x_pc, y_pc) in zip(axes, pc_pairs):
        for lineage, color in PALETTE.items():
            mask = merged["lineage"] == lineage
            ax.scatter(merged.loc[mask, x_pc], merged.loc[mask, y_pc],
                       c=color, label=lineage, s=60, alpha=0.8, edgecolors="white", linewidths=0.5)
        ax.set_xlabel(x_pc, fontsize=11)
        ax.set_ylabel(y_pc, fontsize=11)
        ax.set_title(f"{x_pc} vs {y_pc}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=9, framealpha=0.9)
    fig.suptitle("PCA of Bacterial Strains — Colored by Lineage", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_pca_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dendrogram(pc_df, lineages):
    """Hierarchical clustering dendrogram with lineage annotation."""
    merged = pc_df.join(lineages)
    Z = linkage(pc_df.values, method="ward")

    strain_order = dendrogram(Z, no_plot=True)["ivl"]
    lineage_order = [merged.iloc[int(s), merged.columns.get_loc("lineage")] for s in strain_order]
    color_map = {l: PALETTE[l] for l in PALETTE}

    fig, (ax_dend, ax_line) = plt.subplots(2, 1, figsize=(16, 6),
                                            gridspec_kw={"height_ratios": [5, 1]},
                                            sharex=True)
    dendrogram(Z, ax=ax_dend, leaf_rotation=90, leaf_font_size=0, color_threshold=15)
    ax_dend.set_ylabel("Ward Distance", fontsize=12)
    ax_dend.set_title("Hierarchical Clustering of Bacterial Strains (Ward's Method)",
                      fontsize=13, fontweight="bold")
    ax_dend.axhline(y=15, color="#e74c3c", linestyle="--", alpha=0.7, label="Cut threshold")
    ax_dend.legend(fontsize=10)

    for i, lin in enumerate(lineage_order):
        ax_line.add_patch(plt.Rectangle((i - 0.4, 0), 0.8, 1,
                                         color=color_map[lin], transform=ax_line.transData))
    ax_line.set_yticks([])
    ax_line.set_xlabel("Strain Index", fontsize=12)
    ax_line.set_title("Lineage Assignment", fontsize=11)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color_map.values()]
    ax_line.legend(handles, color_map.keys(), loc="upper right", fontsize=9, ncol=4)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_dendrogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fst_heatmap(fst_matrix):
    """Heatmap of pairwise FST values."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(fst_matrix, annot=True, fmt=".4f", cmap="YlOrRd",
                vmin=0, vmax=fst_matrix.values.max() * 1.2,
                linewidths=1, linecolor="white", square=True,
                cbar_kws={"label": "FST"}, ax=ax)
    ax.set_title("Pairwise FST Between Subpopulations", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_fst_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 60)
    print("STEP 2: Population Structure Analysis")
    print("=" * 60)

    print("\nLoading filtered SNP matrix...")
    snp_matrix = load_snp_matrix()
    print(f"  Loaded {snp_matrix.shape[0]} strains × {snp_matrix.shape[1]} SNPs")

    print("Assigning lineages...")
    lineages = assign_lineages(snp_matrix)
    lineage_counts = lineages["lineage"].value_counts().sort_index()
    for lin, count in lineage_counts.items():
        print(f"  {lin}: {count} strains")

    print("Running PCA...")
    pc_df, explained = run_pca(snp_matrix)
    print(f"  PC1 explains {explained.iloc[0]['variance_explained']*100:.1f}% variance")
    print(f"  Top 3 PCs explain {explained.iloc[2]['cumulative_variance']*100:.1f}% cumulative variance")

    print("Calculating pairwise FST...")
    fst_results = calculate_fst(snp_matrix, lineages)
    for _, row in fst_results.iterrows():
        print(f"  {row['pop1']} vs {row['pop2']}: FST = {row['fst']:.4f}")

    pc_df.to_csv(OUTPUT_DIR / "pca_results.csv")
    explained.to_csv(OUTPUT_DIR / "02_pca_variance_explained.csv", index=False)
    lineages.to_csv(OUTPUT_DIR / "02_lineage_assignments.csv")
    fst_results.to_csv(OUTPUT_DIR / "02_fst_results.csv", index=False)

    print("\nGenerating plots...")
    plot_scree(explained)
    plot_pca_scatter(pc_df, lineages)
    plot_dendrogram(pc_df, lineages)

    fst_pivot = fst_results.pivot_table(index="pop1", columns="pop2", values="fst").fillna(0)
    fst_square = fst_pivot + fst_pivot.T
    np.fill_diagonal(fst_square.values, 0)
    fst_square.index.name = None
    fst_square.columns.name = None
    plot_fst_heatmap(fst_square)

    print(f"\nOutputs saved to {OUTPUT_DIR}/")
    print("  - pca_results.csv")
    print("  - 02_pca_variance_explained.csv")
    print("  - 02_lineage_assignments.csv")
    print("  - 02_fst_results.csv")
    print("  - 02_scree_plot.png")
    print("  - 02_pca_scatter.png")
    print("  - 02_dendrogram.png")
    print("  - 02_fst_heatmap.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
