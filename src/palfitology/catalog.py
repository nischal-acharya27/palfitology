"""Catalog CSV loading and validation.

Handles both pre-processed catalogs (with an ``id`` column) and raw J-PLUS
ADQL exports (with ``TILE_ID`` + ``NUMBER`` columns and a leading ``#``
comment line containing the SQL).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Set

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: Set[str] = {"id", "A_WORLD", "B_WORLD", "pa_jplus"}

# CSVs that palfitology *produces* — auto-discover must ignore them, otherwise
# re-running a second command in the same directory fails with
# "Multiple .csv files ... Disambiguate with --catalog".
#
# Match by exact filename when known, and by glob for the ones we name with
# variable suffixes (per-band reconciliation plots, etc).
_PALFITOLOGY_OUTPUT_NAMES: Set[str] = {
    "PA_results.csv",
    "PA_reconciliation.csv",
    "PA_consensus.csv",
    "make_cutouts_report.csv",
}
_PALFITOLOGY_OUTPUT_GLOBS: tuple[str, ...] = (
    "PA_reconciliation_*.csv",
    "PA_fits.csv",  # written per-object; could in theory live in cwd
)


def _is_palfitology_output(path: Path) -> bool:
    """Return True for CSVs that palfitology wrote itself."""
    if path.name in _PALFITOLOGY_OUTPUT_NAMES:
        return True
    for pattern in _PALFITOLOGY_OUTPUT_GLOBS:
        if path.match(pattern):
            return True
    return False


def load_catalog(path: Path) -> pd.DataFrame:
    """Load a catalog CSV, synthesizing ``id`` from ``TILE_ID``/``NUMBER`` if needed.

    Comment lines starting with ``#`` (such as the SQL preamble in raw ADQL
    exports) are skipped. Raises ValueError if any required column is missing
    after synthesis.
    """
    df = pd.read_csv(path, comment="#")

    # Synthesize id from TILE_ID-NUMBER if needed.
    if "id" not in df.columns and {"TILE_ID", "NUMBER"}.issubset(df.columns):
        df["id"] = (
            df["TILE_ID"].astype(int).astype(str)
            + "-"
            + df["NUMBER"].astype(int).astype(str)
        )

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Catalog at {path} is missing required columns: {sorted(missing)}"
        )
    return df


def auto_discover_catalog(project_root: Path) -> Path:
    """Find a single user-supplied ``*.csv`` in ``project_root``.

    Files that palfitology itself produces (PA_results.csv,
    make_cutouts_report.csv, etc.) are skipped so that re-running a command
    in a directory that already contains pipeline outputs still works.

    Raises FileNotFoundError if zero candidates remain, ValueError if
    multiple non-output CSVs are present.
    """
    all_csvs = sorted(project_root.glob("*.csv"))
    candidates = [p for p in all_csvs if not _is_palfitology_output(p)]

    if len(candidates) == 0:
        if all_csvs:
            ignored = ", ".join(p.name for p in all_csvs)
            raise FileNotFoundError(
                f"No catalog .csv file found in {project_root}. "
                f"(Ignored palfitology-output files: {ignored}.) "
                f"Place a catalog CSV in the project root or pass --catalog explicitly."
            )
        raise FileNotFoundError(
            f"No catalog .csv file found in {project_root}. "
            f"Place a catalog CSV in the project root or pass --catalog explicitly."
        )
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise ValueError(
            f"Multiple .csv files in {project_root} ({names}). "
            f"Disambiguate with --catalog <path>."
        )
    return candidates[0]


def filter_to_existing_image_dirs(df: pd.DataFrame, images_root: Path) -> pd.DataFrame:
    """Drop catalog rows whose ``id`` has no corresponding folder under ``images_root``."""
    image_dirs = {p.name for p in images_root.iterdir() if p.is_dir()}
    filtered = df[df["id"].astype(str).isin(image_dirs)].reset_index(drop=True)
    logger.info(
        f"{len(filtered)} catalog rows have a matching object folder under {images_root}"
    )
    return filtered
