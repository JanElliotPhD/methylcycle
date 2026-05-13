"""Tests for methylcycle.parsers.loader"""
import pytest
import pandas as pd
from methylcycle.parsers.loader import (
    load_methylation_file,
    COL_CHROM, COL_POS, COL_METH, COL_UNMETH, COL_PCT,
)


def _csv(content, tmp_path, suffix=".csv"):
    p = tmp_path / f"sample{suffix}"
    p.write_text(content)
    return p


# ── Count-based files ─────────────────────────────────────────────────────

def test_load_counts_schema(tmp_path):
    content = "Chromosome,Position,Count Methylated,Count Unmethylated\n1,1000,5,3\nchr2,2000,0,10\n"
    df = load_methylation_file(_csv(content, tmp_path), sep=",")
    assert df.attrs["has_counts"] is True
    assert list(df.columns) == [COL_CHROM, COL_POS, COL_METH, COL_UNMETH]


def test_chr_prefix_added_counts(tmp_path):
    content = "Chromosome,Position,Count Methylated,Count Unmethylated\n1,100,1,1\n"
    df = load_methylation_file(_csv(content, tmp_path), sep=",")
    assert df[COL_CHROM].iloc[0] == "chr1"


def test_chr_prefix_not_duplicated(tmp_path):
    content = "Chromosome,Position,Count Methylated,Count Unmethylated\nchr3,100,1,1\n"
    df = load_methylation_file(_csv(content, tmp_path), sep=",")
    assert df[COL_CHROM].iloc[0] == "chr3"


# ── Percentage-only files ─────────────────────────────────────────────────

def test_load_pct_schema(tmp_path):
    content = "Chromosome,Position,Methylation Percentage\nchr1,500,75.0\n"
    df = load_methylation_file(_csv(content, tmp_path), sep=",")
    assert df.attrs["has_counts"] is False
    assert COL_PCT in df.columns
    assert COL_METH not in df.columns
    assert COL_UNMETH not in df.columns


def test_pct_normalised_0_100(tmp_path):
    content = "Chromosome,Position,Methylation Percentage\nchr1,500,75.0\n"
    df = load_methylation_file(_csv(content, tmp_path), sep=",")
    assert abs(df[COL_PCT].iloc[0] - 0.75) < 1e-9


def test_pct_already_0_1(tmp_path):
    content = "Chromosome,Position,Methylation Percentage\nchr1,500,0.5\n"
    df = load_methylation_file(_csv(content, tmp_path), sep=",")
    assert abs(df[COL_PCT].iloc[0] - 0.5) < 1e-9


def test_chr_prefix_added_pct(tmp_path):
    content = "Chromosome,Position,Methylation Percentage\n1,500,50.0\n"
    df = load_methylation_file(_csv(content, tmp_path), sep=",")
    assert df[COL_CHROM].iloc[0] == "chr1"


# ── Error cases ───────────────────────────────────────────────────────────

def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_methylation_file("/nonexistent/path/file.parquet")


def test_missing_columns_raises(tmp_path):
    content = "Chromosome,Position\nchr1,1000\n"
    with pytest.raises(ValueError):
        load_methylation_file(_csv(content, tmp_path), sep=",")
