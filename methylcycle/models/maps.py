"""
methylcycle.models.maps
=======================
Resolve the built-in genomic region map files (RT_Regions, ChromHMM_Regions)
that ship with the package, and define the ``MapType`` enum used throughout
the API.
"""

from __future__ import annotations

import enum
import importlib.resources as pkg_resources
import pathlib
from typing import List


class MapType(str, enum.Enum):
    """Supported genomic region maps for feature extraction.

    Attributes
    ----------
    RT:
        Replication-timing regions (shipped as ``data/RT_Regions.txt``).
    CHROMHMM:
        ChromHMM chromatin-state regions (shipped as ``data/ChromHMM_Regions.txt``).
    CUSTOM:
        User-supplied region list (file path or Python list).
    """
    RT = "RT"
    CHROMHMM = "ChromHMM"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Internal data paths
# ---------------------------------------------------------------------------
_DATA_DIR = pathlib.Path(__file__).parent.parent / "data"

_MAP_FILES = {
    MapType.RT: _DATA_DIR / "RT_Regions.txt",
    MapType.CHROMHMM: _DATA_DIR / "ChromHMM_Regions.txt",
}


def get_region_list(
    map_type: MapType,
    *,
    custom_path: str | pathlib.Path | None = None,
    custom_list: List[str] | None = None,
) -> List[str]:
    """Return the list of region strings for *map_type*.

    Parameters
    ----------
    map_type:
        One of :class:`MapType`.
    custom_path:
        Required when ``map_type == MapType.CUSTOM`` and *custom_list*
        is not given.  Path to a plain-text region file.
    custom_list:
        Required when ``map_type == MapType.CUSTOM`` and *custom_path*
        is not given.  Python list of region strings.

    Returns
    -------
    list[str]

    Raises
    ------
    ValueError
        For invalid combinations of arguments.
    FileNotFoundError
        If a built-in map file is missing (shouldn't happen in a
        correctly installed package).
    """
    if map_type in _MAP_FILES:
        path = _MAP_FILES[map_type]
        if not path.exists():
            raise FileNotFoundError(
                f"Built-in region map not found: {path}. "
                "Please ensure the package data files are installed correctly."
            )
        from methylcycle.parsers.regions import load_region_file
        return load_region_file(path)

    # MapType.CUSTOM
    if custom_list is not None:
        return list(custom_list)
    if custom_path is not None:
        from methylcycle.parsers.regions import load_region_file
        return load_region_file(custom_path)

    raise ValueError(
        "MapType.CUSTOM requires either 'custom_path' or 'custom_list'."
    )
