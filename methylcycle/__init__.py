"""
MethylCycle
===========
A bioinformatics library for DNA methylation analysis and modeling.

Modules
-------
- parsers   : Load and validate BED / CSV / Parquet methylation files
- models    : Feature extraction (RT, ChromHMM, custom) and prediction
- utils     : QC, normalization, entropy helpers
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("methylcycle")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

__author__ = "Jan Elliot González and Irene Hernando-Herraez"
__license__ = "MIT"

from methylcycle.parsers.loader import load_methylation_file   # noqa: F401
from methylcycle.utils.qc import passes_qc                     # noqa: F401
from methylcycle.models.feature_extraction import extract_features  # noqa: F401
from methylcycle.models.predict import predict                  # noqa: F401
from methylcycle.models.datasets import list_datasets, load_dataset
