# MethylCycle

A bioinformatics Python library for cell cycle prediction based in DNA methylation .

## Installation

```bash
pip install methylcycle

or

pip install git+https://github.com/JanElliotPhD/methylcycle.git

```

For development:

```bash
git clone https://github.com/yourname/methylcycle
cd methylcycle
pip install -e ".[dev]"
```

## Pipeline Overview

```
Input files (BED / CSV / Parquet)
        ↓
   load_methylation_file()     ← normalise columns + chr prefix
        ↓
      passes_qc()              ← ≥ 100k covered CpG positions
        ↓
   extract_features()          ← RT or ChromHMM region map
        ↓
      preprocess()             ← map/metric-specific normalisation
        ↓
       predict()               ← model inference
```

## Quick Start

```python
import methylcycle.models.paths as mp
from methylcycle import predict
from methylcycle.models.maps import MapType
from methylcycle.models.feature_extraction import Metric
from methylcycle.models.datasets import list_datasets, load_dataset

# Ver datasets disponibles
list_datasets()
# ['single_cell_mESC', 'bulk_hepatocytes']

# Obtener rutas y nombres
files, names = load_dataset("single_cell_mESC")

results_mean = predict(
    files=files,
    map_type=MapType.RT,
    metric=Metric.MEAN,
    sep=",",
    min_cpgs = 100000
)

results_mean.columns = ['prob_G1', 'prob_S', 'prob_G2M']

print(results_mean)
```

## Step-by-Step Usage

### 1. Load a methylation file

```python
from methylcycle.parsers.loader import load_methylation_file

df = load_methylation_file("sample.parquet")
# Returns DataFrame with columns:
#   Chromosome | Position | Count Methylated | Count Unmethylated
```

Supported input formats: `.parquet`, `.csv`, `.bed`, `.tsv`.

Supported column schemes:
- `Chromosome`, `Position`, `Count Methylated`, `Count Unmethylated`
- `Chromosome`, `Position`, `Methylation Percentage`

Chromosome prefix `chr` is added automatically if missing.

### 2. Quality Control

```python
from methylcycle.utils.qc import passes_qc

report = passes_qc(df)          # default: 100 000 covered CpGs
print(report)                   # QCReport(PASS | covered=1,234,567 / required=100,000)

if report:
    # proceed
    ...
```

### 3. Feature Extraction

```python
from methylcycle.models.feature_extraction import extract_features, Metric
from methylcycle.models.maps import MapType

# Built-in RT map — mean methylation
df_meth, df_unmeth, df_mean = extract_features(
    files=["s1.parquet", "s2.parquet"],
    map_type=MapType.RT,
    metric=Metric.MEAN,
)

# Built-in ChromHMM map — entropy
df_meth, df_unmeth, df_entropy = extract_features(
    files=["s1.parquet"],
    map_type=MapType.CHROMHMM,
    metric=Metric.ENTROPY,
)

# Custom map — pass a file path or a Python list
df_meth, df_unmeth, df_mean = extract_features(
    files=["s1.parquet"],
    map_type=MapType.CUSTOM,
    metric=Metric.MEAN,
    custom_path="/path/to/my_regions.txt",
    # or: custom_list=["chr1-0-10000", "chr2-50000-80000"]
)
```

### 4. Preprocessing (manual)

```python
from methylcycle.models.preprocessing import preprocess
import joblib

pca = joblib.load("pca_rt.joblib")
X = preprocess(df_mean, MapType.RT, Metric.MEAN, pca_model=pca)
```

### 5. Model Artefact Paths

Place your `.joblib` model files and configure paths:

```python
import methylcycle.models.paths as mp

mp.RT_MEAN_PCA              = "/models/pca_rt.joblib"
mp.RT_MEAN_MODEL            = "/models/clf_rt_mean.joblib"
mp.RT_ENTROPY_MODEL         = "/models/clf_rt_entropy.joblib"
mp.CHROMHMM_REGIONS_ORDER   = "/models/chromhmm_order.txt"
mp.CHROMHMM_MEAN_MODEL      = "/models/clf_chromhmm_mean.joblib"
mp.CHROMHMM_ENTROPY_MODEL   = "/models/clf_chromhmm_entropy.joblib"
```

## Region File Format

Both built-in maps and custom region files use the format:

```
chr1-0-100000
chr1-100000-200000
chr2-50000-80000
```

One region per line. Lines starting with `#` are ignored.

## Normalisation Rules

| Map      | Metric  | Preprocessing                     |
|----------|---------|-----------------------------------|
| RT       | MEAN    | IQR per sample → PCA transform    |
| RT       | ENTROPY | None                              |
| ChromHMM | MEAN    | Z-score per sample                |
| ChromHMM | ENTROPY | None                              |
| Custom   | any     | None (caller handles)             |

## Running Tests

```bash
pytest
```

## License

MIT
