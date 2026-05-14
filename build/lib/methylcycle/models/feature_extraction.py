"""
methylcycle.models.feature_extraction
======================================
Compute per-region methylation features (mean methylation or entropy)
for one or many samples.

Four computation paths depending on input schema and requested metric:

+------------+---------+----------------------------------------------+
| has_counts | Metric  | Computation                                  |
+============+=========+==============================================+
| True       | MEAN    | sum(meth_reads) / sum(total_reads) per region|
| True       | ENTROPY | unique-CpG binary entropy (read-based)       |
| False      | MEAN    | mean(Methylation Percentage) per region      |
| False      | ENTROPY | mean of per-CpG H(p,1-p) across region      |
+------------+---------+----------------------------------------------+
"""

from __future__ import annotations

import enum
import pathlib
import warnings
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from methylcycle.parsers.regions import parse_regions
from methylcycle.models.maps import MapType, get_region_list
from methylcycle.utils.entropy import compute_entropy_matrix


class Metric(str, enum.Enum):
    """Feature metric to compute per region.

    Attributes
    ----------
    MEAN:
        Mean methylation per region.
        - Count files: sum(methylated_reads) / sum(total_reads)
        - Percentage files: mean(Methylation Percentage)
    ENTROPY:
        Shannon entropy per region.
        - Count files: unique-CpG binary entropy (read-based deduplication)
        - Percentage files: mean of H(p, 1-p) across CpGs in the region
    """
    MEAN    = "mean"
    ENTROPY = "entropy"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(
    files: Sequence[Union[str, pathlib.Path]],
    map_type: MapType,
    metric: Metric = Metric.MEAN,
    *,
    sample_names: List[str] | None = None,
    custom_path: str | pathlib.Path | None = None,
    custom_list: List[str] | None = None,
    entropy_min_abs: int = 5,
    entropy_prior: float = 0.5,
    show_progress: bool = True,
    sep: str = "\t",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract per-region methylation features from a list of sample files.

    Parameters
    ----------
    files:
        Ordered list of sample file paths (output of
        :func:`~methylcycle.parsers.loader.load_methylation_file`, already
        QC-passed).  All files must share the same schema (count-based or
        percentage-only); mixing schemas raises a ``ValueError``.
    map_type:
        Which genomic region map to use.
    metric:
        ``Metric.MEAN`` or ``Metric.ENTROPY``.
    sample_names:
        Column labels.  Defaults to file stems.
    custom_path / custom_list:
        Region source when ``map_type == MapType.CUSTOM``.
    entropy_min_abs:
        Minimum total unique CpGs (count mode) or CpGs with data (pct mode)
        for entropy to be non-NaN.
    entropy_prior:
        Laplace smoothing for count-based entropy.
    show_progress:
        Show ``tqdm`` progress bar.
    sep:
        Column separator for text files (default ``\\t``). Use ``','`` for CSV files.
        Ignored for Parquet files.

    Returns
    -------
    df_meth : pd.DataFrame
        ``(n_regions × n_samples)``.
        MEAN/count  → methylated read counts.
        MEAN/pct    → sum of percentages (divide by n_cpgs for mean).
        ENTROPY/count → unique methylated CpG counts.
        ENTROPY/pct   → sum of per-CpG entropies (divide by n_cpgs for mean).
    df_unmeth : pd.DataFrame
        MEAN/count  → unmethylated read counts.
        MEAN/pct    → number of CpGs contributing (n_cpgs per region).
        ENTROPY/count → unique unmethylated CpG counts.
        ENTROPY/pct   → same n_cpgs as df_meth (for consistency).
    df_metric : pd.DataFrame
        Final derived feature matrix (mean or entropy values).
    """
    files = [pathlib.Path(f) for f in files]
    if sample_names is None:
        sample_names = [f.stem for f in files]

    region_list = get_region_list(
        map_type, custom_path=custom_path, custom_list=custom_list
    )
    regions, regions_by_chr = parse_regions(region_list)
    n_regions = len(regions)
    n_samples  = len(files)

    # Detect schema from first file
    first = _load_sample(files[0], sep=sep)
    has_counts = first.attrs.get("has_counts", True)

    # Accumulation matrices
    mat_a = np.zeros((n_regions, n_samples), dtype=np.float64)  # meth counts / pct sums / entropy sums
    mat_b = np.zeros((n_regions, n_samples), dtype=np.float64)  # unmeth counts / cpg counts

    iterator = (
        tqdm(enumerate(files), total=n_samples, desc=f"Extracting [{metric.value}]")
        if show_progress else enumerate(files)
    )

    for i, file in iterator:
        fd = (first if i == 0 else _load_sample(file, sep=sep))
        file_has_counts = fd.attrs.get("has_counts", True)
        if file_has_counts != has_counts:
            raise ValueError(
                f"Mixed schemas detected: file '{file}' has "
                f"has_counts={file_has_counts} but previous files have "
                f"has_counts={has_counts}. All files must share the same schema."
            )

        if has_counts:
            if metric == Metric.MEAN:
                _accumulate_counts_mean(fd, regions_by_chr, mat_a, mat_b, i)
            else:
                _accumulate_counts_entropy(fd, regions_by_chr, mat_a, mat_b, i)
        else:
            if metric == Metric.MEAN:
                _accumulate_pct_mean(fd, regions_by_chr, mat_a, mat_b, i)
            else:
                _accumulate_pct_entropy(fd, regions_by_chr, mat_a, mat_b, i)

    index = regions["region_id"].tolist()
    df_meth   = pd.DataFrame(mat_a, index=index, columns=sample_names)
    df_unmeth = pd.DataFrame(mat_b, index=index, columns=sample_names)

    # ── Derive final metric ───────────────────────────────────────────────
    if has_counts and metric == Metric.MEAN:
        total = df_meth + df_unmeth
        df_metric = df_meth.div(total.replace(0, np.nan))

    elif has_counts and metric == Metric.ENTROPY:
        df_metric = compute_entropy_matrix(
            df_meth, df_unmeth,
            min_abs=entropy_min_abs,
            prior=entropy_prior,
        )

    elif not has_counts and metric == Metric.MEAN:
        # mat_a = sum of percentages, mat_b = n_cpgs per region
        n_cpgs = df_unmeth.replace(0, np.nan)
        df_metric = df_meth.div(n_cpgs)

    else:  # not has_counts and ENTROPY
        # mat_a = sum of H(p,1-p), mat_b = n_cpgs per region
        n_cpgs = df_unmeth.copy()
        insufficient = n_cpgs < entropy_min_abs
        n_cpgs_safe = n_cpgs.replace(0, np.nan)
        df_metric = df_meth.div(n_cpgs_safe)
        df_metric[insufficient] = np.nan

    return df_meth, df_unmeth, df_metric


# ---------------------------------------------------------------------------
# Internal: load
# ---------------------------------------------------------------------------

def _load_sample(path: pathlib.Path, sep: str = "\t") -> pd.DataFrame:
    """Load a single already-normalised sample file (Parquet or CSV/BED)."""
    from methylcycle.parsers.loader import load_methylation_file
    return load_methylation_file(path, sep=sep)


# ---------------------------------------------------------------------------
# Internal: count-based accumulators
# ---------------------------------------------------------------------------

def _accumulate_counts_mean(
    fd: pd.DataFrame,
    regions_by_chr: Dict[str, pd.DataFrame],
    mat_meth: np.ndarray,
    mat_unmeth: np.ndarray,
    col: int,
) -> None:
    """Accumulate methylated / unmethylated read counts per region."""
    for chrom, sub in fd.groupby("Chromosome"):
        if chrom not in regions_by_chr:
            continue
        reg = regions_by_chr[chrom]
        pos = sub["Position"].values

        idx   = np.searchsorted(reg["start"].values, pos, side="right") - 1
        valid = idx >= 0
        idx_c = np.clip(idx, 0, len(reg) - 1)
        mask  = valid & (pos < reg["end"].values[idx_c])

        ridx = reg["global_idx"].values[idx_c[mask]]
        np.add.at(mat_meth[..., col],   ridx, sub["Count Methylated"].values[mask])
        np.add.at(mat_unmeth[..., col], ridx, sub["Count Unmethylated"].values[mask])


def _accumulate_counts_entropy(
    fd: pd.DataFrame,
    regions_by_chr: Dict[str, pd.DataFrame],
    mat_meth: np.ndarray,
    mat_unmeth: np.ndarray,
    col: int,
) -> None:
    """Accumulate unique methylated / unmethylated CpG positions per region."""
    MAX_POS = 300_000_000

    for chrom, sub in fd.groupby("Chromosome", sort=False):
        if chrom not in regions_by_chr:
            continue
        reg    = regions_by_chr[chrom]
        starts = reg["start"].values
        ends   = reg["end"].values
        gidx   = reg["global_idx"].values

        pos      = sub["Position"].values
        m_reads  = sub["Count Methylated"].values
        u_reads  = sub["Count Unmethylated"].values

        idx   = np.searchsorted(starts, pos, side="right") - 1
        valid = idx >= 0
        idx_c = np.clip(idx, 0, len(reg) - 1)
        mask  = valid & (pos < ends[idx_c])

        pos_m    = pos[mask]
        ridx     = gidx[idx_c[mask]]
        m_reads_m = m_reads[mask]
        u_reads_m = u_reads[mask]

        key = ridx.astype(np.int64) * MAX_POS + pos_m.astype(np.int64)

        unique_meth,   _ = np.unique(key[m_reads_m > 0], return_inverse=True)
        unique_unmeth, _ = np.unique(key[u_reads_m > 0], return_inverse=True)

        np.add.at(mat_meth[..., col],   (unique_meth   // MAX_POS).astype(np.int32), 1)
        np.add.at(mat_unmeth[..., col], (unique_unmeth // MAX_POS).astype(np.int32), 1)


# ---------------------------------------------------------------------------
# Internal: percentage-based accumulators
# ---------------------------------------------------------------------------

def _accumulate_pct_mean(
    fd: pd.DataFrame,
    regions_by_chr: Dict[str, pd.DataFrame],
    mat_sum: np.ndarray,
    mat_n: np.ndarray,
    col: int,
) -> None:
    """Accumulate sum of percentages and CpG count per region (for mean)."""
    for chrom, sub in fd.groupby("Chromosome"):
        if chrom not in regions_by_chr:
            continue
        reg = regions_by_chr[chrom]
        pos = sub["Position"].values
        pct = sub["Methylation Percentage"].values

        idx   = np.searchsorted(reg["start"].values, pos, side="right") - 1
        valid = idx >= 0
        idx_c = np.clip(idx, 0, len(reg) - 1)
        mask  = valid & (pos < reg["end"].values[idx_c])

        ridx = reg["global_idx"].values[idx_c[mask]]
        np.add.at(mat_sum[..., col], ridx, pct[mask])   # sum percentages
        np.add.at(mat_n[..., col],   ridx, 1)            # count CpGs


def _accumulate_pct_entropy(
    fd: pd.DataFrame,
    regions_by_chr: Dict[str, pd.DataFrame],
    mat_H: np.ndarray,
    mat_n: np.ndarray,
    col: int,
) -> None:
    """Accumulate per-CpG H(p, 1-p) entropy sum and CpG count per region."""
    for chrom, sub in fd.groupby("Chromosome"):
        if chrom not in regions_by_chr:
            continue
        reg = regions_by_chr[chrom]
        pos = sub["Position"].values
        pct = sub["Methylation Percentage"].values

        idx   = np.searchsorted(reg["start"].values, pos, side="right") - 1
        valid = idx >= 0
        idx_c = np.clip(idx, 0, len(reg) - 1)
        mask  = valid & (pos < reg["end"].values[idx_c])

        ridx  = reg["global_idx"].values[idx_c[mask]]
        p     = pct[mask].astype(np.float64)
        q     = 1.0 - p

        # Shannon entropy per CpG: H(p) = -p*log2(p) - q*log2(q)
        with np.errstate(divide="ignore", invalid="ignore"):
            H_cpg = -p * np.log2(np.where(p > 0, p, 1.0)) \
                    - q * np.log2(np.where(q > 0, q, 1.0))
        H_cpg = np.where((p == 0) | (p == 1), 0.0, H_cpg)

        np.add.at(mat_H[..., col], ridx, H_cpg)  # sum of H values
        np.add.at(mat_n[..., col], ridx, 1)       # count CpGs