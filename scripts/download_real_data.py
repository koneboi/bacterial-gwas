#!/usr/bin/env python3
"""
download_real_data.py — Download the real bacterial GWAS dataset

Downloads the classic pyseer/bugwas *Streptococcus pneumoniae* (Maela, Thailand)
dataset used in Earle et al. 2016 ('Identifying lineage effects when controlling
for population structure in bacterial GWAS', Nature Microbiology) and based on
the Maela pneumococcal collection of Chewapreecha et al. 2014 (PLoS Genetics):

  1. Genotype matrix (gen.txt) — 5,000 real SNPs × 100 isolates, coded as
     nucleotides (A/C/G/T).  Positions are relative to the S. pneumoniae
     ATCC 700669 (SPN23F) reference genome (NC_011900.1), which is the
     reference that the original Maela reads were mapped onto with SMALT
     (Chewapreecha et al. 2014).
  2. Binary phenotype (pheno.txt) — 0/1 per isolate (50 / 50).
  3. Core-genome phylogeny (tree.txt) enabling lineage effects to be assessed.
  4. NCBI gene annotation of the S. pneumoniae ATCC 700669 reference genome,
     used to annotate top associated SNPs to genes.

Files are cached, subset and already small, under data/real/.
The download uses urllib with an SSL-disabled context and falls back to
`curl -k` if urllib fails, so it works on networks with quirky certificates.
"""

import gzip
import shutil
import ssl
import subprocess
import sys
import urllib.request
import uuid
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "real"
ANNOT_DIR = DATA_DIR / "annotations"

# bugwas R package example data (100 Maela pneumococcal isolates)
BUGWAS_URL = (
    "https://raw.githubusercontent.com/sgearle/bugwas/master/example/example_data.zip"
)
# S. pneumoniae ATCC 700669 (SPN23F), NC_011900.1 — NCBI datasets API (GENOME_GFF)
NCBI_GFF_URL = (
    "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/"
    "GCF_000026665.1/download?include_annotation_type=GENOME_GFF"
)
NCBI_GFF_REFERENCE_ID = "NCBI_Assembly:GCF_000026665.1"


def _ssl_broken_context():
    """An SSL context that skips certificate verification (unverified network)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def download_urllib(url, dest):
    """Download with urllib (unverified SSL context)."""
    req = urllib.request.Request(url, headers={"User-Agent": "bacterial-gwas/1.0"})
    with urllib.request.urlopen(req, context=_ssl_broken_context(), timeout=120) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)
    return dest


def download_curl(url, dest):
    """Download with curl -k as a fallback."""
    subprocess.run(["curl", "-k", "-L", "--max-time", "300", "-o", str(dest), url], check=True)
    return dest


def download(url, dest, label):
    print(f"  Downloading {label}: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        download_urllib(url, dest)
    except Exception as err:
        print(f"    urllib failed ({err}); using curl -k fallback...")
        download_curl(url, dest)
    print(f"    -> {dest} ({dest.stat().st_size:,} bytes)")
    return dest


def main():
    force = "--force" in sys.argv
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DOWNLOAD REAL BACTERIAL GWAS DATA (S. pneumoniae, Maela)")
    print("=" * 60)

    # 1) bugwas example data (genotypes + phenotype + tree)
    zip_path = DATA_DIR / "example_data.zip"
    if not zip_path.exists() or force:
        download(BUGWAS_URL, zip_path, "bugwas example data")
    else:
        print(f"  Using cached {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        for member in ("gen.txt", "pheno.txt", "tree.txt"):
            src = next((n for n in zf.namelist() if n.endswith(member) and "__MACOSX" not in n), None)
            if src is None:
                sys.exit(f"ERROR: {member} not found inside example_data.zip")
            if not (DATA_DIR / member).exists() or force:
                print(f"  Extracting {member}")
                with zf.open(src) as fi, open(DATA_DIR / member, "wb") as fo:
                    shutil.copyfileobj(fi, fo)

    # 2) NCBI annotation for S. pneumoniae ATCC 700669 (reference of the SNPs)
    annot_zip = ANNOT_DIR / "annotation_gcf_000026665.zip"
    if not annot_zip.exists() or force:
        download(NCBI_GFF_URL, annot_zip, "NCBI annotation (GCF_000026665.1)")
    gff_target = ANNOT_DIR / "S_pneumoniae_ATCC700669_NC_011900.gff"
    if not gff_target.exists() or force:
        print("  Extracting genomic.gff -> " + gff_target.name)
        with zipfile.ZipFile(annot_zip) as zf:
            name = next(n for n in zf.namelist() if n.endswith("genomic.gff"))
            with zf.open(name) as fi, open(gff_target, "wb") as fo:
                shutil.copyfileobj(fi, fo)

    # check the annotation has the reference we expect (sanity)
    print("  Verifying annotation reference...")
    first = gff_target.read_text(errors="ignore").splitlines()
    ok = any("GCF_000026665.1" in line or "SPN23F" in line for line in first[:40])
    if not ok:
        print("  WARNING: annotation does not mention GCF_000026665.1/SPN23F; check download")
    else:
        print("  OK: annotation matches S. pneumoniae ATCC 700669 (SPN23F)")

    n_snps = sum(1 for _ in open(DATA_DIR / "gen.txt")) - 1
    n_pheno = sum(1 for _ in open(DATA_DIR / "pheno.txt")) - 1
    print(f"\nReal data ready under {DATA_DIR}/:")
    print(f"  - gen.txt  : {n_snps} SNPs")
    print(f"  - pheno.txt: {n_pheno} isolates (binary phenotype)")
    print(f"  - tree.txt : core-genome phylogeny")
    print(f"  - annotations/S_pneumoniae_ATCC700669_NC_011900.gff : gene annotation")
    print("=" * 60)


if __name__ == "__main__":
    main()