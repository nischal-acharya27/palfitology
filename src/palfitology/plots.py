"""Per-band diagnostic PNG and per-object 3x4 mosaic plotting.

These functions are intentionally side-effecting (they call savefig) and don't
return anything -- they're the leaves of the pipeline. The matplotlib backend
is set to 'Agg' here so the module is safe to import in headless processes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  -- after .use()
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
from matplotlib.patches import Ellipse as MplEllipse

from .fit import FitCandidate

logger = logging.getLogger(__name__)


def _format(v: Optional[float], fmt: str) -> str:
    return "nan" if v is None or not np.isfinite(v) else format(v, fmt)


def make_band_plot(
    data: np.ndarray,
    objectid: str,
    band: str,
    cand: Optional[FitCandidate],
    out_path: Path,
    is_imputed: bool,
    fallback_priors: Optional[Tuple[float, float, float, float, float]] = None,
) -> None:
    """Save one PNG showing the cutout with the fitted ellipse overlaid.

    ``fallback_priors = (pa_deg, sma, ell, x0, y0)`` is used only when
    ``cand`` is None (imputed); we still draw the catalog-prior ellipse so the
    user can sanity-check what we fell back to.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch())
    ax.imshow(data, origin="lower", cmap="gray_r", norm=norm)

    if cand is not None:
        pa_deg, sma, ell, x0, y0 = cand.pa_deg, cand.sma, cand.ell, cand.x0, cand.y0
    elif fallback_priors is not None:
        pa_deg, sma, ell, x0, y0 = fallback_priors
    else:
        pa_deg = sma = ell = x0 = y0 = float("nan")

    if all(np.isfinite([pa_deg, sma, ell, x0, y0])):
        smb = sma * (1.0 - ell)
        patch = MplEllipse(
            (x0, y0),
            width=2 * sma,
            height=2 * smb,
            angle=pa_deg,
            edgecolor="red",
            facecolor="none",
            lw=2.0,
        )
        ax.add_patch(patch)

        pa_rad = np.deg2rad(pa_deg)
        dx = sma * np.cos(pa_rad)
        dy = sma * np.sin(pa_rad)
        ax.plot([x0 - dx, x0 + dx], [y0 - dy, y0 + dy], "-", color="orange", lw=1.0)
        ax.plot([x0], [y0], "+", color="cyan", ms=10, mew=1.5)

    title = f"{objectid}  ({band})"
    if is_imputed:
        title += "  [IMPUTED]"
    elif cand is not None and cand.weak:
        title += "  [WEAK FIT]"
    ax.set_title(title)
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")

    lines = [
        f"PA  = {_format(pa_deg, '.3f')} deg",
        f"SMA = {_format(sma, '.3f')} pix",
        f"ell = {_format(ell, '.3f')}",
        f"x0,y0 = ({_format(x0, '.1f')}, {_format(y0, '.1f')})",
    ]
    if cand is not None:
        lines.append(f"pa_err = {_format(cand.pa_err, '.3f')} deg")
        lines.append(f"score = {_format(cand.score, '.4f')}  (pa_err/sma; lower=better)")
        lines.append(f"config = {cand.config_tag}, smoothing = {cand.smoothing}")
    elif is_imputed:
        lines.append("config = imputed (from catalog priors)")

    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="white",
        fontsize=9,
        bbox=dict(facecolor="black", alpha=0.55, pad=4, edgecolor="none"),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def make_summary_mosaic(
    objectid: str,
    band_data: Dict[str, np.ndarray],
    band_cands: Dict[str, Optional[FitCandidate]],
    band_statuses: Dict[str, str],
    bands_order: List[str],
    out_path: Union[Path, Sequence[Path]],
) -> None:
    """3xN grid of all bands for one object, ellipses overlaid.

    ``out_path`` can be a single path or a list of paths -- the same figure is
    written to each (cheap, lets us write per-object + central copies in one
    render).
    """
    n = len(bands_order)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows))
    axes = np.atleast_2d(axes).reshape(nrows, ncols)

    for i, band in enumerate(bands_order):
        ax = axes[i // ncols, i % ncols]
        data = band_data.get(band)
        cand = band_cands.get(band)
        status = band_statuses.get(band, "missing")

        if data is None:
            ax.text(0.5, 0.5, f"{band}\n(missing)", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch())
        ax.imshow(data, origin="lower", cmap="gray_r", norm=norm)

        if cand is not None and all(
            np.isfinite([cand.pa_deg, cand.sma, cand.ell, cand.x0, cand.y0])
        ):
            smb = cand.sma * (1.0 - cand.ell)
            color = "red" if status == "ok" else ("orange" if status == "weak" else "gray")
            ax.add_patch(MplEllipse(
                (cand.x0, cand.y0),
                width=2 * cand.sma,
                height=2 * smb,
                angle=cand.pa_deg,
                edgecolor=color,
                facecolor="none",
                lw=1.4,
            ))

        label = f"{band}"
        if cand is not None:
            label += f"\nPA={cand.pa_deg:.1f}°  ell={cand.ell:.2f}"
        if status != "ok":
            label += f"  [{status}]"
        ax.set_title(label, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.suptitle(f"{objectid} — all bands", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    paths = [out_path] if isinstance(out_path, (str, Path)) else list(out_path)
    for p in paths:
        fig.savefig(p, dpi=110)
    plt.close(fig)
