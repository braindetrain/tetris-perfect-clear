"""Tests for the Tetris Perfect Clear project.

Run from the repo root:  python -m pytest tests -v
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
POSITION_COLS = ["piece_1", "piece_2", "piece_3", "piece_4"]
PIECES = {"I", "J", "L", "O", "S", "T", "Z"}


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(ROOT / "data" / "tetris_perfect_clear_sequences.csv")


@pytest.fixture(scope="module")
def model_run():
    """Run code/model.py once; the scripts use relative paths, so run from the repo root."""
    result = subprocess.run(
        [sys.executable, "code/model.py"], cwd=ROOT, capture_output=True, text=True
    )
    return result


# ---------- Data tests ----------

def test_shape_and_columns(df):
    # 7*6*5*4 = 840 ordered sequences of 4 different pieces
    assert df.shape == (840, 5)
    assert list(df.columns) == POSITION_COLS + ["success"]


def test_no_missing_values(df):
    assert df.isnull().sum().sum() == 0


def test_no_duplicate_sequences(df):
    assert not df.duplicated(subset=POSITION_COLS).any()


def test_piece_labels_are_valid(df):
    for col in POSITION_COLS:
        assert set(df[col]) <= PIECES


def test_each_sequence_has_four_distinct_pieces(df):
    assert (df[POSITION_COLS].nunique(axis=1) == 4).all()


def test_success_label_is_binary(df):
    assert set(df["success"].unique()) <= {0, 1}


def test_success_counts_match_source(df):
    # TTPC reports 711 of the 840 sequences can produce a Perfect Clear
    assert df["success"].sum() == 711
    assert (df["success"] == 0).sum() == 129


@pytest.mark.parametrize("col", POSITION_COLS)
def test_each_piece_appears_equally_often_per_position(df, col):
    # 840 sequences / 7 pieces = 120 appearances of each piece in each position
    assert (df[col].value_counts() == 120).all()


# ---------- Model and output tests ----------

def test_model_script_runs_without_error(model_run):
    assert model_run.returncode == 0, model_run.stderr


@pytest.mark.parametrize("metric", ["tree_accuracy", "majority_baseline", "cv_accuracy"])
def test_metrics_are_valid_proportions(model_run, metric):
    metrics = pd.read_csv(ROOT / "docs" / "tables" / "model_metrics.csv").set_index("metric")["value"]
    assert 0 <= metrics[metric] <= 1


def test_model_is_better_than_a_coin_flip(model_run):
    metrics = pd.read_csv(ROOT / "docs" / "tables" / "model_metrics.csv").set_index("metric")["value"]
    assert metrics["tree_accuracy"] > 0.5


@pytest.mark.parametrize("path", [
    "docs/tables/success_counts.csv",
    "docs/tables/success_by_piece_and_position.csv",
    "docs/tables/success_rate_piece1_piece2.csv",
    "docs/tables/model_metrics.csv",
    "docs/tables/feature_importance.csv",
    "docs/figures/success_counts.png",
    "docs/figures/success_rate_by_piece_position.png",
    "docs/figures/success_rate_heatmap.png",
    "docs/figures/confusion_matrix.png",
    "docs/figures/feature_importance.png",
])
def test_expected_output_files_exist(model_run, path):
    assert (ROOT / path).exists()