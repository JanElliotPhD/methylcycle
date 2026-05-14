"""
methylcycle.parsers.regions
===========================
Parse genomic region lists used for feature extraction.

Expected region format (one per line)::

    chr1-10000-20000
    chr2-50000-80000
    ...

``parse_regions`` turns that list into a DataFrame and builds the
per-chromosome sorted lookup tables needed by the feature-extraction
routines.
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_region_file(path: Union[str, pathlib.Path]) -> List[str]:
    """Read a region file and return the raw list of region strings.

    Each non-empty, non-comment line is returned as a region string,
    e.g. ``['chr1-0-10000', 'chr1-10000-20000', ...]``.

    Parameters
    ----------
    path:
        Path to a plain-text file with one region per line in the
        ``chr<N>-<start>-<end>`` format.

    Returns
    -------
    list[str]
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Region file not found: {path}")

    regions: List[str] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                regions.append(line)
    return regions


def parse_regions(
    region_list: List[str],
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Parse a list of region strings into a DataFrame and per-chromosome lookup.

    Parameters
    ----------
    region_list:
        Strings of the form ``chr1-10000-20000``.

    Returns
    -------
    regions : pd.DataFrame
        Columns: ``Chromosome``, ``start``, ``end``, ``global_idx``,
        ``region_id`` (the original string).
    regions_by_chr : dict[str, pd.DataFrame]
        Chromosome → sub-DataFrame sorted by ``start``, ready for
        ``np.searchsorted``.

    Raises
    ------
    ValueError
        If any region string cannot be parsed.
    """
    rows = []
    for r in region_list:
        parts = r.split("-")
        if len(parts) != 3:
            raise ValueError(
                f"Cannot parse region '{r}'. "
                "Expected format: chr<N>-<start>-<end>"
            )
        chrom, start, end = parts
        # Normalise chromosome prefix
        if not chrom.startswith("chr"):
            chrom = "chr" + chrom
        try:
            rows.append((chrom, int(start), int(end), r))
        except ValueError:
            raise ValueError(
                f"Non-integer coordinates in region '{r}'."
            )

    regions = pd.DataFrame(rows, columns=["Chromosome", "start", "end", "region_id"])
    regions["global_idx"] = np.arange(len(regions), dtype=np.int64)

    regions_by_chr: Dict[str, pd.DataFrame] = {
        chrom: grp.sort_values("start").reset_index(drop=True)
        for chrom, grp in regions.groupby("Chromosome")
    }

    return regions, regions_by_chr
