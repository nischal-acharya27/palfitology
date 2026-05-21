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

from .detect import DetectionResult
from .fit import FitCandidate

logger = logging.getLogger(__name__)


def _detection_crop_slice(
    data: np.ndarray,
    detect_result: Optional[DetectionResult],
    pad_frac: float = 0.20,
) -> Optional[Tuple[slice, slice]]:
    """Return the (row_slice, col_slice) for the detection crop, or None.

    Same logic as ``_detection_crop`` but exposes just the slice so callers
    can apply the *same* window to multiple arrays.  Returns None when there
    isn't a usable detection (status != 'ok', missing stats, empty mask).
    """
    if detect_result is None or detect_result.status != "ok":
        return None

    bg = detect_result.background
    rms = detect_result.background_rms
    if not (np.isfinite(bg) and np.isfinite(rms) and rms > 0):
        return None

    ny, nx = data.shape
    threshold = bg + detect_result.sigma_threshold * rms
    mask = np.isfinite(data) & (data > threshold)
    if not mask.any():
        return None

    rows, cols = np.where(mask)
    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())

    rspan = max(rmax - rmin, 1)
    cspan = max(cmax - cmin, 1)
    pad_r = max(int(round(pad_frac * rspan)), 4)
    pad_c = max(int(round(pad_frac * cspan)), 4)

    r0 = max(0, rmin - pad_r)
    r1 = min(ny, rmax + pad_r + 1)
    c0 = max(0, cmin - pad_c)
    c1 = min(nx, cmax + pad_c + 1)
    return slice(r0, r1), slice(c0, c1)


def _detection_crop(
    data: np.ndarray,
    detect_result: Optional[DetectionResult],
    pad_frac: float = 0.20,
) -> Tuple[np.ndarray, int, int]:
    """Return a cropped view of ``data`` centred on the detected source.

    The crop is the bounding box of all pixels that exceeded the detection
    threshold (i.e. the full detection mask), expanded by ``pad_frac`` of the
    bounding-box size on each side so the galaxy isn't clipped to its edge.

    Parameters
    ----------
    data : 2D array
        The raw cutout.
    detect_result : DetectionResult or None
        When None (or status != 'ok'), the original array is returned
        unchanged with zero offsets.
    pad_frac : float
        Fractional padding added around the bounding box (default 0.20 = 20%).

    Returns
    -------
    (cropped, x_offset, y_offset)
        ``x_offset`` and ``y_offset`` are the pixel offsets of the crop origin
        in the original image coordinate system. Add them back to convert
        cropped pixel coords → original pixel coords (or subtract when
        converting overlay coords → cropped display coords).
    """
    ny, nx = data.shape

    if detect_result is None or detect_result.status != "ok":
        return data, 0, 0

    # Re-derive the detection mask from the stored background + sigma threshold
    # so we don't need to store the mask itself.
    bg = detect_result.background
    rms = detect_result.background_rms
    if not (np.isfinite(bg) and np.isfinite(rms) and rms > 0):
        return data, 0, 0

    threshold = bg + detect_result.sigma_threshold * rms
    mask = np.isfinite(data) & (data > threshold)

    if not mask.any():
        return data, 0, 0

    rows, cols = np.where(mask)
    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())

    # Expand bounding box by pad_frac on each side.
    rspan = max(rmax - rmin, 1)
    cspan = max(cmax - cmin, 1)
    pad_r = max(int(round(pad_frac * rspan)), 4)
    pad_c = max(int(round(pad_frac * cspan)), 4)

    r0 = max(0, rmin - pad_r)
    r1 = min(ny, rmax + pad_r + 1)
    c0 = max(0, cmin - pad_c)
    c1 = min(nx, cmax + pad_c + 1)

    return data[r0:r1, c0:c1], c0, r0


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
    detect_result: Optional[DetectionResult] = None,
) -> None:
    """Save one PNG showing the cutout with the fitted ellipse overlaid.

    When ``detect_result`` is provided and status=='ok', the displayed image is
    cropped to the bounding box of the sigma-clipped source mask (with padding)
    so only the galaxy region is shown.

    ``fallback_priors = (pa_deg, sma, ell, x0, y0)`` is used only when
    ``cand`` is None (imputed); we still draw the catalog-prior ellipse so the
    user can sanity-check what we fell back to.
    """
    # Crop to detection bounding box; x_off/y_off convert overlay coords.
    display, x_off, y_off = _detection_crop(data, detect_result)

    fig, ax = plt.subplots(figsize=(7, 7))

    norm = ImageNormalize(display, interval=ZScaleInterval(), stretch=AsinhStretch())
    ax.imshow(display, origin="lower", cmap="gray_r", norm=norm)

    if cand is not None:
        pa_deg, sma, ell, x0, y0 = cand.pa_deg, cand.sma, cand.ell, cand.x0, cand.y0
    elif fallback_priors is not None:
        pa_deg, sma, ell, x0, y0 = fallback_priors
    else:
        pa_deg = sma = ell = x0 = y0 = float("nan")

    # Shift overlay coords into the cropped display frame.
    x0_d = x0 - x_off
    y0_d = y0 - y_off

    if all(np.isfinite([pa_deg, sma, ell, x0_d, y0_d])):
        smb = sma * (1.0 - ell)
        patch = MplEllipse(
            (x0_d, y0_d),
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
        ax.plot([x0_d - dx, x0_d + dx], [y0_d - dy, y0_d + dy], "-", color="orange", lw=1.0)
        ax.plot([x0_d], [y0_d], "+", color="cyan", ms=10, mew=1.5)

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
    detect_result: Optional[DetectionResult] = None,
) -> None:
    """3xN grid of all bands for one object, ellipses overlaid.

    When ``detect_result`` is provided and status=='ok', every panel is cropped
    to the same bounding box (derived from the rSDSS detection mask) so all
    bands are displayed at the same spatial scale and centred on the galaxy.

    ``out_path`` can be a single path or a list of paths -- the same figure is
    written to each (cheap, lets us write per-object + central copies in one
    render).
    """
    n = len(bands_order)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows))
    axes = np.atleast_2d(axes).reshape(nrows, ncols)

    # Compute a single crop geometry from the first band that has data.
    # We use detect_result against whichever band array is available so all
    # panels share the same spatial window.
    _ref_data = next(
        (d for d in band_data.values() if d is not None), None
    )
    _, x_off, y_off = _detection_crop(_ref_data, detect_result) if _ref_data is not None else (None, 0, 0)

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

        # Apply the shared crop (same x_off/y_off for all bands).
        display, _, _ = _detection_crop(data, detect_result)

        norm = ImageNormalize(display, interval=ZScaleInterval(), stretch=AsinhStretch())
        ax.imshow(display, origin="lower", cmap="gray_r", norm=norm)

        if cand is not None and all(
            np.isfinite([cand.pa_deg, cand.sma, cand.ell, cand.x0, cand.y0])
        ):
            smb = cand.sma * (1.0 - cand.ell)
            color = "red" if status == "ok" else ("orange" if status == "weak" else "gray")
            ax.add_patch(MplEllipse(
                (cand.x0 - x_off, cand.y0 - y_off),
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


# ---------------------------------------------------------------------------
# V0.5 clipped-cutout diagnostic mosaic
# ---------------------------------------------------------------------------
#
# `make_clipped_summary` builds a per-object side-by-side mosaic showing each
# band's raw cutout next to its sigma-clipped sibling, so a user can scan
# at a glance whether the rSDSS mask is doing the right thing across all
# wavelengths.  The layout is:
#
#     row 0: J0378-raw J0378-clip   J0395-raw J0395-clip   J0410-raw ...
#     row 1: ...
#
# i.e. each band occupies *two* horizontally-adjacent panels (raw | clipped).
# We default to 6 columns of band-pairs = 12 columns of axes, 2 rows = 24
# axes for the canonical 12 J-PLUS bands, but the layout adapts to whatever
# bands the caller passes.

def make_clipped_summary(
    *,
    objectid: str,
    band_raw: Dict[str, Optional[np.ndarray]],
    band_clipped: Dict[str, Optional[np.ndarray]],
    bands_order: List[str],
    out_path: Union[Path, Sequence[Path]],
    detect_result: Optional[DetectionResult] = None,
    detect_band: Optional[str] = None,
    pairs_per_row: int = 4,
) -> None:
    """Save a side-by-side raw|clipped diagnostic mosaic for one object.

    Parameters
    ----------
    objectid : str
        Used in the figure suptitle.
    band_raw : dict[str, ndarray | None]
        Per-band raw cutout arrays.  Missing bands -> a "missing" placeholder.
    band_clipped : dict[str, ndarray | None]
        Per-band clipped cutout arrays (NaN outside the source).  Missing -> placeholder.
    bands_order : list[str]
        Canonical band order; both dicts must use the same keys.
    out_path : Path or sequence of Path
        Single path or list of paths the same figure is saved to.
    detect_result : DetectionResult, optional
        When provided and status=='ok', every panel is cropped to the same
        bounding box (same crop strategy as ``make_summary_mosaic``).
    detect_band : str, optional
        If given, the crop window is derived from this band's raw array
        (typically rSDSS) and applied identically to *every* panel — raw,
        clipped, all bands.  This is the correct behaviour for cross-band
        QC because per-band re-thresholding would give differing bboxes
        depending on each band's brightness.  If omitted, the window is
        derived from the first available array.
    pairs_per_row : int
        Number of band-pairs displayed per row.  Default 4 -> 8 columns,
        3 rows for the 12 J-PLUS bands.  Bump to 6 for a wider 2-row layout.

    Notes
    -----
    NaN pixels (the clipped-out background) render in a contrasting colour
    (crimson) so the mask boundary is obvious without needing a contour overlay.
    """
    n = len(bands_order)
    pairs_per_row = max(1, int(pairs_per_row))
    pair_rows = int(np.ceil(n / pairs_per_row))
    ncols = pairs_per_row * 2  # each band: raw column + clipped column

    fig, axes = plt.subplots(
        pair_rows, ncols,
        figsize=(2.4 * ncols, 2.7 * pair_rows),
        squeeze=False,
    )

    # Shared crop window — derived ONCE from the detect band (or, failing
    # that, the first available array) and applied identically to every
    # panel below.  Per-band re-thresholding would otherwise give different
    # bboxes for bands with different background brightness, making the
    # cross-band layout visually inconsistent.
    _ref: Optional[np.ndarray] = None
    if detect_band is not None:
        # Don't use ``or`` on ndarrays — falsy-test isn't defined for them.
        _ref = band_raw.get(detect_band)
        if _ref is None:
            _ref = band_clipped.get(detect_band)
    if _ref is None:
        _ref = next(
            (a for a in list(band_raw.values()) + list(band_clipped.values())
             if a is not None),
            None,
        )
    crop_slices: Optional[Tuple[slice, slice]] = None
    if _ref is not None and detect_result is not None:
        crop_slices = _detection_crop_slice(_ref, detect_result)

    def _apply_crop(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        if crop_slices is None:
            return arr
        rs, cs = crop_slices
        return arr[rs, cs]

    # Colormap that renders NaN visibly (only used on the clipped panel; the
    # raw panel uses the default gray_r without bad-value highlighting).
    cmap_clipped = plt.cm.gray_r.copy()
    cmap_clipped.set_bad("crimson")

    for i, band in enumerate(bands_order):
        row = i // pairs_per_row
        raw_col = (i % pairs_per_row) * 2
        clip_col = raw_col + 1
        ax_raw = axes[row, raw_col]
        ax_clip = axes[row, clip_col]

        raw = band_raw.get(band)
        clipped = band_clipped.get(band)

        # ---- raw panel ------------------------------------------------
        if raw is None:
            ax_raw.text(0.5, 0.5, f"{band}\n(no raw)", ha="center", va="center",
                        transform=ax_raw.transAxes, color="gray", fontsize=8)
        else:
            display_r = _apply_crop(raw)
            try:
                norm_r = ImageNormalize(
                    display_r, interval=ZScaleInterval(), stretch=AsinhStretch()
                )
                ax_raw.imshow(display_r, origin="lower", cmap="gray_r", norm=norm_r)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"[{objectid}/{band}] raw imshow failed ({exc}); "
                    f"falling back to default scaling"
                )
                ax_raw.imshow(display_r, origin="lower", cmap="gray_r")
        ax_raw.set_xticks([]); ax_raw.set_yticks([])
        ax_raw.set_title(f"{band} raw", fontsize=8)

        # ---- clipped panel --------------------------------------------
        if clipped is None:
            ax_clip.text(0.5, 0.5, f"{band}\n(no clip)", ha="center", va="center",
                         transform=ax_clip.transAxes, color="gray", fontsize=8)
        else:
            display_c = _apply_crop(clipped)
            # Use the raw panel's stretch when both are available so the
            # clipped panel doesn't get re-scaled to the surviving range
            # only.  Falls back to clipped's own ZScale when no raw exists.
            try:
                source_for_norm = (
                    _apply_crop(raw) if raw is not None else display_c
                )
                norm_c = ImageNormalize(
                    source_for_norm,
                    interval=ZScaleInterval(),
                    stretch=AsinhStretch(),
                )
                ax_clip.imshow(display_c, origin="lower", cmap=cmap_clipped, norm=norm_c)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"[{objectid}/{band}] clipped imshow failed ({exc}); "
                    f"falling back to default scaling"
                )
                ax_clip.imshow(display_c, origin="lower", cmap=cmap_clipped)
        ax_clip.set_xticks([]); ax_clip.set_yticks([])
        # Annotate the clipped panel with the NaN coverage so the user sees
        # what fraction of pixels survived the mask.
        if clipped is not None:
            n_finite = int(np.isfinite(clipped).sum())
            n_total = int(clipped.size)
            kept_frac = n_finite / n_total if n_total > 0 else 0.0
            ax_clip.set_title(
                f"{band} clip\nkept {kept_frac:.0%}",
                fontsize=8,
            )
        else:
            ax_clip.set_title(f"{band} clip", fontsize=8)

    # Hide leftover axes (e.g. when n % pairs_per_row != 0).
    n_pairs_total = pair_rows * pairs_per_row
    for j in range(n, n_pairs_total):
        row = j // pairs_per_row
        raw_col = (j % pairs_per_row) * 2
        axes[row, raw_col].axis("off")
        axes[row, raw_col + 1].axis("off")

    # Build a concise suptitle that records detection state.
    if detect_result is not None and detect_result.status == "ok":
        det_tag = (
            f"detect ok  σ={detect_result.sigma_threshold:.1f}  "
            f"npix={detect_result.npix}"
        )
    elif detect_result is not None:
        det_tag = f"detect {detect_result.status}"
    else:
        det_tag = "no detection"
    fig.suptitle(f"{objectid} — clipped cutouts  ({det_tag})", fontsize=12)

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    paths = [out_path] if isinstance(out_path, (str, Path)) else list(out_path)
    for p in paths:
        fig.savefig(p, dpi=110)
    plt.close(fig)
