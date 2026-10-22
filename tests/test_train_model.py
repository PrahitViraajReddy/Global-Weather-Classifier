"""Tests for training-only weather temperature bins and dataset validation."""
import pandas as pd
import pytest

import train_model


def test_quantile_boundaries_are_derived_from_training_targets():
    train = pd.Series(range(100), dtype=float)
    test = pd.Series([-100.0, 20.0, 1000.0])
    train_labels, test_labels, edges, names = train_model.make_quantile_labels(train, test)
    assert len(names) == len(edges) - 1
    assert train_labels.nunique() == len(names)
    assert test_labels.iloc[0] == 0
    assert test_labels.iloc[-1] == len(names) - 1


def test_missing_dataset_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        train_model.load_data(tmp_path / "missing.csv")


def test_git_lfs_pointer_is_not_parsed_as_csv(tmp_path):
    path = tmp_path / "weather.csv"
    path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        "size 1000\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Git LFS pointer"):
        train_model.load_data(path)


def test_missing_columns_are_reported(tmp_path):
    path = tmp_path / "weather.csv"
    pd.DataFrame({"temperature_celsius": [1.0] * 60}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        train_model.load_data(path)
