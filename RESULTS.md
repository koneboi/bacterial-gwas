# RESULTS — Real Bacterial GWAS (Streptococcus pneumoniae)

This document reports the results of a genome-wide association analysis run on
**real** bacterial sequencing data, produced by
`scripts/real_bacterial_gwas.py` (data downloaded by
`scripts/download_real_data.py`). It complements the synthetic steps 01–04.

## 1. Dataset

| Property | Value |
|----------|-------|
| Organism | *Streptococcus pneumoniae* (pneumococcus) |
| Population | Carriage isolates sampled in the Maela refugee camp, Thailand |
| Isolates | 100 (50 trait cases / 50 trait controls, binary phenotype) |
| Variants | 5,000 core-genome SNP positions; 3,405 pass MAF ≥ 0.05 QC |
| Reference genome | *S. pneumoniae* ATCC 700669 (SPN23F), NC_011900.1 / EMBL FM211187, NCBI Assembly GCF_000026665.1 |
| Phenotype | Binary 0/1 provided with the example dataset (undocumented in the package; in the source Maela study the analysed clinical trait was β-lactam / penicillin non-susceptibility) |
| Download | `bugwas` R-package example data — https://raw.githubusercontent.com/sgearle/bugwas/master/example/example_data.zip (`gen.txt`, `pheno.txt`, `tree.txt`) |
| Annotation | NCBI datasets API, `GCF_000026665.1` GENOME_GFF |
| License | Package example data distributed with the open-source `bugwas` package (GPL/AGPL + the underlying Maela data are from Chewapreecha et al. 2014, ENA ERP000435 etc.) |

**Provenance.** The genotyping and phylogeny are real. The 100-isolate subset,
SNP table and binary phenotype are distributed with the `bugwas` R package
(Earle *et al.*, *Nature Microbiology* 1:16041, 2016;
doi:10.1038/nmicrobiol.2016.41). The underlying pneumococcal sequences and the
β-lactam susceptibility trait come from the Maela carriage cohort of
Chewapreecha *et al.* (*PLoS Genetics* 10(1):e1004547, 2014;
doi:10.1371/journal.pgen.1004547), where reads were mapped onto the very same
reference (ATCC 700669 / FM211187) with SMALT. SNP positions in `gen.txt` are
therefore directly comparable with the annotation GFF used here.

> **Caveat.** The exact MIC definition behind the 50/50 label of the 100-isolate
> example is not documented in the package. We therefore treat the label as "the
> binary trait shipped with the bugwas example" and interpret associations in
> terms of what the data themselves show (see below), rather than claiming a
> particular antibiotic breakpoint.

## 2. Method

Mirrors the synthetic pipeline:

1. **QC** — encode each SNP as presence/absence of the minor allele; keep sites with MAF ≥ 0.05 (non-monomorphic).
2. **Population structure** — PCA on the standardized minor-allele matrix (10 PCs; PC1 28.3%, top 3 ≈ 64% of variance).
3. **Association** — SNP-by-SNP logistic regression with the top 3 PCs as covariates (Statsmodels `Logit`; Fisher/χ² fallback on separation), matching steps 03 conventions.
4. **Correction** — Benjamini–Hochberg FDR and Bonferroni at α = 0.05; genomic inflation λ_GC.
5. **Annotation** — map each SNP to its containing gene (nearest gene if intergenic, ±1 kb) from the real SPN23F GFF.

## 3. Results

### 3.1 Genome-wide summary
- SNPs tested: **3,405** (of 5,000 raw positions)
- FDR-significant (q < 0.05): **706 variants (21%)**
- Bonferroni-significant: **5 variants**
- Genomic inflation factor **λ_GC = 2.88** — strongly inflated (see below)
- c. 95% of positions could be annotated to a gene from the real GFF

| Output | File |
|--------|------|
| Manhattan plot | `output/real_manhattan_plot.png` |
| Q–Q plot | `output/real_qq_plot.png` |
| p-value distribution | `output/real_pvalue_distribution.png` |
| PCA structure (PC1–3) | `output/real_pca_structure.png` |
| Top-variant forest plot | `output/real_top_effects_forest.png` |
| Peak-region zoom | `output/real_peak_region.png` |
| MAF vs signal | `output/real_maf_signal.png` |
| All results | `output/real_gwas_results.csv` |
| PCA coordinates | `output/real_pca.csv` |
| Gene annotation table | `output/real_gene_annotation.csv` |

### 3.2 Strongest association (naive, marginal scan)
Before any PC adjustment the single strongest signal in the dataset is a clean
block around **position 65,379** (p ≈ 1×10⁻¹¹), sitting between the purine
biosynthesis operon (`purK`/`purB`) upstream and the `strH` β-N-acetylhexos-
aminidase downstream — most likely the AAA-family ATPase `SPN23F_RS00375`.
This association is **completely separated**: all 29 isolates carrying the
minor allele are trait cases (29/29, OR = ∞). The block essentially disappears
once principal components are added as covariates — a textbook
**population-stratification / lineage effect**, i.e. the trait is almost
perfectly confounded with one phylogenetic lineage of the 100 isolates.

### 3.3 PC-adjusted results
After conditioning on the top 3 PCs, the p-value distribution remains heavily
inflated (λ_GC = 2.88; 706 / 3,405 variants FDR-significant), and the top hits
(positions ~34, 240, 63, 482, 506 kb — `radA`, `spuA`, a hypothetical protein,
`infB`, a GNAT acetyltransferase) carry unstable, near-separating effect sizes
(ORs of 0.02–47). In other words, three PCs do not remove the dominant lineage
effect in this highly structured sample.

### 3.4 Penicillin-binding proteins
The two canonical β-lactam determinants that fall inside the 25–536 kb window
covered by this SNP subset, **`pbp2x`** (position 291,868–294,120) and
**`pbp1a`** (331,533–333,692), each contain only a handful of segregating
variants here. Their best p-values (≈ 4.6×10⁻⁵ and ≈ 6.5×10⁻⁵ marginal;
≈ 0.055 after PC-adjustment) do not reach genome-wide significance in a
100-isolate sample.

## 4. Interpretation

Real pneumococcal populations are strongly lineage-structured, and this 100-
isolate subset illustrates the phenomenon cleanly. The most significant naive
hit is a fully confounded lineage marker (~65 kb block); conditioning on few
principal components leaves substantial residual inflation (λ_GC ≈ 2.9, 706
FDR hits) and unstable effect sizes. This is precisely the population-structure
problem that lineage- and mixed-model methods (bugwas, GEMMA/pyseer LMMs) were
introduced to solve — which is why the original Maela study of ≥ 3,000 isolates
used lineage-aware models and recovered the mosaic penicillin-binding-protein
genes (`pbp2x`, `pbp1a`, `pbp2b`) as the true resistance determinants
(Chewapreecha et al. 2014). Here, with only 100 isolates and sparse coverage in
`pbp2x`/`pbp1a`, those loci do not reach genome-wide significance — an honest
illustration both of what real data look like and of the power requirements of
a bacterial GWAS.

## 5. How to reproduce / run with your own data

```bash
# 1. Fetch the real data (cached under data/real/)
python scripts/download_real_data.py

# 2. Run the real-data analysis
python scripts/real_bacterial_gwas.py
```

Both scripts take the same minor-allele binary matrix + phenotype files as the
synthetic pipeline, so the association core can be pointed at your own data by
replacing `data/real/gen.txt` and `data/real/pheno.txt` (any TAB-separated
genotype matrix with a `ps` position column and one column per isolate).