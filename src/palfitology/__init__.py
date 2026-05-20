"""palfitology -- PSF-aware isophotal PA fitting and GALFIT prep for J-PLUS cutouts.

The public surface for now is the high-level pipeline runner and the data
structures it returns. The CLI (`palfitology fit-pa`) is the supported way to
drive the pipeline end-to-end; importing the module is useful for embedding
the fit logic into notebooks or other scripts.
"""

from __future__ import annotations

__version__ = "0.3.0.dev0"

from .consensus import (
    circular_diff_deg,
    circular_mean_deg,
    circular_std_deg,
    consensus_for_catalog,
    consensus_for_object,
)
from .fit import FitCandidate, fit_pa_with_fallbacks
from .pipeline import fit_catalog
from .psf import PSFInfo, preprocess_for_fit

# Canonical J-PLUS / SDSS band order, broadband and medium/narrowband interleaved.
ALL_BANDS = [
    "uJAVA",
    "J0378",
    "J0395",
    "J0410",
    "J0430",
    "gSDSS",
    "J0515",
    "rSDSS",
    "J0660",
    "iSDSS",
    "J0861",
    "zSDSS",
]

__all__ = [
    "__version__",
    "ALL_BANDS",
    "FitCandidate",
    "fit_pa_with_fallbacks",
    "fit_catalog",
    "PSFInfo",
    "preprocess_for_fit",
    "circular_mean_deg",
    "circular_diff_deg",
    "circular_std_deg",
    "consensus_for_object",
    "consensus_for_catalog",
]
