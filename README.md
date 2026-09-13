# Bacterial GWAS Pipeline

Genome-wide association study pipeline for bacterial genomes — identifying genetic variants associated with phenotypes (antibiotic resistance, virulence factors, metabolic traits).

## Overview

This pipeline implements a complete bacterial GWAS workflow from raw SNP data to candidate gene identification:

```
SNP Matrix Construction → Population Structure → Association Testing → Candidate Gene Identification
```

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_snp_matrix_construction.py` | Simulate bacterial genomes, construct SNP matrix, quality filtering |
| 2 | `02_population_structure.py` | PCA, hierarchical clustering, FST analysis |
| 3 | `03_association_testing.py` | Logistic regression GWAS, Manhattan/QQ plots, multiple testing correction |
| 4 | `04_candidate_genes.py` | Candidate region identification, effect sizes, LD analysis |
| **Real** | **`download_real_data.py`** | **Download a real *S. pneumoniae* (Maela) dataset: 5,000 SNPs × 100 isolates + binary phenotype + reference GFF** |
| **Real** | **`real_bacterial_gwas.py`** | **Run the real-data GWAS (QC → PCA → logistic regression + PCs → FDR/Bonferroni → gene annotation + plots)** — see [`RESULTS.md`](RESULTS.md) |

## Pipeline Details

### Step 1 — SNP Matrix Construction
- Simulates 100 bacterial strains with ~1000 core genome SNPs
- Constructs a binary SNP presence/absence matrix
- Calculates minor allele frequency (MAF)
- Applies quality filters: MAF > 0.05, missing data < 10%
- Outputs summary statistics and filtered SNP matrix

### Step 2 — Population Structure
- PCA of the SNP matrix to detect population stratification
- Scree plot and PC scatter plots colored by lineage
- Hierarchical clustering (Ward's method) of strains
- FST calculation between subpopulations (Weir & Cockerham)
- Visualization of population structure to inform association model

### Step 3 — Association Testing
- Simulates a binary phenotype (antibiotic resistance)
- Logistic regression GWAS with population structure covariates (principal components)
- Bonferroni and Benjamini–Hochberg FDR correction
- Manhattan plot, QQ plot, and lambda GC inflation statistic
- Table of top associated SNPs

### Step 4 — Candidate Gene Identification
- Identifies top candidate genomic regions from association results
- Simulated gene enrichment analysis in associated regions
- Association signal visualization around candidate genes
- Odds ratios and effect sizes for top SNPs
- LD-like correlation analysis between top SNPs
- Summary report

## Real-Data Analysis (steps 01–04 on real data)

In addition to the fully simulated pipeline, the repo runs the same GWAS workflow
on **real** bacterial data:

```bash
# 1. Download the real S. pneumoniae (Maela) dataset and reference annotation
python scripts/download_real_data.py

# 2. Run the real-data analysis
python scripts/real_bacterial_gwas.py
```

The real dataset is the 100-isolate *Streptococcus pneumoniae* example shipped
with the `bugwas` R package (Earle et al., Nat Microbiol 2016), based on the
Maela carriage cohort of Chewapreecha et al. (PLoS Genet 2014); SNP positions
use the SPN23F / ATCC 700669 reference (NCBI assembly GCF_000026665.1), which
is downloaded and used for gene annotation of the top hits. Outputs are written
to `output/real_*.png` and `output/real_*.csv`. Full details, honest
interpretation, limitations and a "run with your own data" guide are in
[`RESULTS.md`](RESULTS.md).

## Skills Demonstrated

- Bacterial comparative genomics and pangenomics
- Statistical genetics and GWAS methodology
- Population structure correction in association studies
- Multiple testing correction (Bonferroni, FDR)
- Scientific visualization (Manhattan plots, QQ plots, PCA)
- Python scientific stack (pandas, scipy, scikit-learn, matplotlib)

## Tools Referenced

This pipeline is conceptually based on established bacterial GWAS tools:

- **PLINK** — SNP data manipulation and association testing
- **Roary** — Pangenome construction and core genome SNP extraction
- **Scoary** — Gene-based association testing for bacterial genomes
- **Pyseer** — Linear mixed model association for bacterial populations

The implementation here uses Python with numpy/scipy for the statistical core, providing a pedagogical and reproducible framework.

## Author

**Boï Kone** — [GitHub](https://github.com/koneboi)

## Requirements

```
pip install -r requirements.txt
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline sequentially
python scripts/01_snp_matrix_construction.py
python scripts/02_population_structure.py
python scripts/03_association_testing.py
python scripts/04_candidate_genes.py

# Real-data analysis (downloads ~500 KB)
python scripts/download_real_data.py
python scripts/real_bacterial_gwas.py

# All outputs saved to output/
```

Each script is self-contained with simulated data and saves publication-quality plots to `output/`.

## Output

All plots and tables are saved to the `output/` directory:

- `snp_matrix.csv` — Filtered SNP matrix
- `pca_results.csv` — PCA coordinates and variance explained
- `gwas_results.csv` — Association test results
- `candidate_genes.csv` — Top candidate regions
- `real_*.png` / `real_*.csv` — Real-data GWAS figures and tables (see `RESULTS.md`)
- Multiple `.png` visualization files
