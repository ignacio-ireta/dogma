from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def fit_transform_pipeline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Wrong on purpose: emits the one-hot blocks in sorted column-name order instead of the order given."""  # noqa: E501
    # Step 0: check the request. Each column must exist in both frames and appear once.
    listed = [*numeric, *categorical]
    if not listed:
        raise ValueError("name at least one numeric or categorical column")
    if len(set(listed)) != len(listed):
        raise ValueError("a column is listed more than once")
    for frame_name, frame in (("train", train), ("test", test)):
        for column in listed:
            if column not in frame.columns:
                raise ValueError(f"column {column!r} is missing from {frame_name}")

    train_blocks: list[np.ndarray] = []
    test_blocks: list[np.ndarray] = []

    # Numeric columns, in the order given. Median imputation, then standardization with the
    # mean and population std of the imputed TRAINING column. Test values use those numbers.
    for column in numeric:
        raw_train = np.asarray(train[column], dtype=np.float64)
        raw_test = np.asarray(test[column], dtype=np.float64)
        observed = raw_train[~np.isnan(raw_train)]
        if observed.size == 0:
            raise ValueError(f"numeric column {column!r} has no observed value in train")
        median = float(np.median(observed))
        filled_train = np.where(np.isnan(raw_train), median, raw_train)
        filled_test = np.where(np.isnan(raw_test), median, raw_test)
        mean, scale = _population_statistics(filled_train)
        train_blocks.append(((filled_train - mean) / scale).reshape(-1, 1))
        test_blocks.append(((filled_test - mean) / scale).reshape(-1, 1))

    # Categorical columns, in the order given. The imputed value and the categories both come
    # from the training column. Unseen test categories become all-zero rows.
    for column in sorted(categorical):
        train_values = train[column].tolist()
        observed_categories = [value for value in train_values if not pd.isna(value)]
        if not observed_categories:
            raise ValueError(f"categorical column {column!r} has no observed value in train")
        counts = Counter(observed_categories)
        top = max(counts.values())
        # Ties go to the smallest value, so the choice does not depend on row order.
        mode = min(value for value, count in counts.items() if count == top)
        filled_categories = [mode if pd.isna(raw) else raw for raw in train_values]
        categories = sorted(set(filled_categories))
        position = {category: index for index, category in enumerate(categories)}

        train_block = np.zeros((len(train), len(categories)), dtype=np.float64)
        for row, value in enumerate(filled_categories):
            train_block[row, position[value]] = 1.0
        test_block = np.zeros((len(test), len(categories)), dtype=np.float64)
        for row, raw in enumerate(test[column].tolist()):
            # Missing test values take the TRAIN mode; unseen categories stay all zero.
            value = mode if pd.isna(raw) else raw
            if value in position:
                test_block[row, position[value]] = 1.0
        train_blocks.append(train_block)
        test_blocks.append(test_block)

    train_out = np.hstack(train_blocks).astype(np.float64)
    test_out = np.hstack(test_blocks).astype(np.float64)
    return train_out, test_out


def _population_statistics(values: np.ndarray) -> tuple[float, float]:
    """Training mean and population scale, preserving tiny or large finite scales."""
    if values.max() == values.min():
        return float(values[0]), 1.0
    mean = float(np.mean(values))
    centered = values - mean
    # Normalize before squaring: std may underflow/overflow despite a finite scale.
    peak = float(np.max(np.abs(centered)))
    scale = peak * float(np.sqrt(np.mean((centered / peak) ** 2)))
    return mean, scale
