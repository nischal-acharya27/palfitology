"""Cross-band PA / ellipticity consensus (planned).

Once per-band fits exist (from `fit_pa`), this module will compute a
per-object consensus PA + ellipticity across the 12 bands, weighted by
pa_err, and flag bands whose PA disagrees with the consensus by more than a
configurable threshold.

Not yet implemented.
"""

from __future__ import annotations

__all__ = []
