"""Tests for palfitology.catalog."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from palfitology.catalog import (
    auto_discover_catalog,
    filter_to_existing_image_dirs,
    load_catalog,
)


def test_load_catalog_with_id_column(tmp_path: Path):
    csv = tmp_path / "cat.csv"
    csv.write_text(
        "id,A_WORLD,B_WORLD,pa_jplus\n"
        "92801-11428,0.013,0.011,-7.2\n"
        "93140-3576,0.003,0.002,-61.6\n"
    )
    df = load_catalog(csv)
    assert list(df["id"]) == ["92801-11428", "93140-3576"]
    assert len(df) == 2


def test_load_catalog_synthesizes_id_from_tile_id_and_number(tmp_path: Path):
    csv = tmp_path / "cat.csv"
    csv.write_text(
        "TILE_ID,NUMBER,A_WORLD,B_WORLD,pa_jplus\n"
        "92801,11428,0.013,0.011,-7.2\n"
        "93140,3576,0.003,0.002,-61.6\n"
    )
    df = load_catalog(csv)
    assert "id" in df.columns
    assert list(df["id"]) == ["92801-11428", "93140-3576"]


def test_load_catalog_skips_sql_comment_preamble(tmp_path: Path):
    csv = tmp_path / "cat.csv"
    csv.write_text(
        "# SELECT TILE_ID, NUMBER, A_WORLD, B_WORLD, pa_jplus FROM jplus\n"
        "TILE_ID,NUMBER,A_WORLD,B_WORLD,pa_jplus\n"
        "92801,11428,0.013,0.011,-7.2\n"
    )
    df = load_catalog(csv)
    assert list(df["id"]) == ["92801-11428"]


def test_load_catalog_raises_on_missing_columns(tmp_path: Path):
    csv = tmp_path / "cat.csv"
    csv.write_text("id,A_WORLD\n92801-11428,0.013\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_catalog(csv)


def test_auto_discover_single_csv(tmp_path: Path):
    (tmp_path / "cat.csv").write_text("id,A_WORLD,B_WORLD,pa_jplus\n92801-11428,1,1,0\n")
    assert auto_discover_catalog(tmp_path).name == "cat.csv"


def test_auto_discover_no_csvs(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        auto_discover_catalog(tmp_path)


def test_auto_discover_multiple_csvs(tmp_path: Path):
    (tmp_path / "a.csv").write_text("id\nx\n")
    (tmp_path / "b.csv").write_text("id\nx\n")
    with pytest.raises(ValueError, match="Multiple"):
        auto_discover_catalog(tmp_path)


def test_filter_to_existing_image_dirs(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "92801-11428").mkdir()
    df = pd.DataFrame({
        "id": ["92801-11428", "missing-object"],
        "A_WORLD": [1.0, 1.0],
        "B_WORLD": [1.0, 1.0],
        "pa_jplus": [0.0, 0.0],
    })
    filtered = filter_to_existing_image_dirs(df, images)
    assert list(filtered["id"]) == ["92801-11428"]
