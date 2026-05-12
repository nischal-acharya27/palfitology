"""PSF loading and deconvolution helpers (planned, V8).

Stage two of palfitology will use these to preprocess cutouts before the
isophotal fit. The current `fit_pa_with_fallbacks` works on raw images; this
module will provide a `deconvolve_with_psf` that the pipeline can call when
the PSF FWHM is comparable to the galaxy size.

Not yet implemented -- the file exists as a placeholder so the import surface
is stable.
"""

from __future__ import annotations

__all__ = []
