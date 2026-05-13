"""
methylcycle.models.paths
========================
Central registry of file-system paths for model artefacts.

All paths can be overridden at runtime before calling
:func:`~methylcycle.models.predict.predict`:

.. code-block:: python

    import methylcycle.models.paths as mp
    mp.RT_MEAN_PCA     = "/my/models/pca_rt.joblib"
    mp.RT_MEAN_MODEL   = "/my/models/clf_rt_mean.joblib"
    mp.CHROMHMM_REGIONS_ORDER = "/my/models/chromhmm_region_order.txt"
    ...

Alternatively, pass them directly to ``predict()`` as keyword arguments.
"""

import pathlib

# ---------------------------------------------------------------------------
# Base directory — change this to point at your model bundle
# ---------------------------------------------------------------------------
MODEL_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent / "data" / "models"

# ---------------------------------------------------------------------------
# RT pipeline artefacts
# ---------------------------------------------------------------------------

RT_MEAN_PCA: pathlib.Path = MODEL_DIR / "rt_mean_pca.joblib"
"""Fitted sklearn PCA object used in the RT + MEAN preprocessing step."""

RT_MEAN_MODEL: pathlib.Path = MODEL_DIR / "rt_mean_model.joblib"
"""Trained classifier / regressor for RT + MEAN features."""

RT_ENTROPY_MODEL: pathlib.Path = MODEL_DIR / "rt_entropy_model.joblib"
"""Trained classifier / regressor for RT + ENTROPY features (no preprocessing)."""

# ---------------------------------------------------------------------------
# ChromHMM pipeline artefacts
# ---------------------------------------------------------------------------

CHROMHMM_REGIONS_ORDER: pathlib.Path = MODEL_DIR / "chromhmm_region_order.txt"
"""Plain-text file with the ordered list of region IDs expected by the
ChromHMM model (one region per line, same format as region map files)."""

CHROMHMM_MEAN_MODEL: pathlib.Path = MODEL_DIR / "chromhmm_mean_model.joblib"
"""Trained classifier / regressor for ChromHMM + MEAN features."""

CHROMHMM_ENTROPY_MODEL: pathlib.Path = MODEL_DIR / "chromhmm_entropy_model.joblib"
"""Trained classifier / regressor for ChromHMM + ENTROPY features."""
