"""Data loading utilities for advertisement datasets.

This module provides functionality for loading and preprocessing tabular
advertisement data for click-through rate prediction. It handles missing
values, encodes categorical variables, and normalizes numerical features.
PyTorch ``Dataset`` and ``DataLoader`` objects are created for convenient
training.

Example:
    >>> from .data_loader import load_tabular_data, save_preprocessor
    >>> train_loader, val_loader, test_loader, pre = load_tabular_data(
    ...     train_csv="train.csv", val_csv="val.csv", test_csv="test.csv",
    ...     target_col="clicked")
    >>> save_preprocessor(pre, "models/preprocessor.pkl")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


@dataclass
class TabularLoaders:
    """Container holding dataloaders for train/val/test sets."""

    train: DataLoader
    val: Optional[DataLoader]
    test: Optional[DataLoader]


class AdDataset(Dataset):
    """PyTorch ``Dataset`` wrapping preprocessed feature and label tensors."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def _build_preprocessor(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: Optional[List[str]] = None,
    numerical_cols: Optional[List[str]] = None,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Create a preprocessing pipeline for the dataset."""

    if categorical_cols is None:
        categorical_cols = (
            df.select_dtypes(include=["object", "category"]).columns.tolist()
        )
    if numerical_cols is None:
        numerical_cols = (
            df.select_dtypes(include=[np.number])
            .columns.difference([target_col])
            .tolist()
        )

    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", num_pipe, numerical_cols),
            ("cat", cat_pipe, categorical_cols),
        ]
    )
    return preprocessor, categorical_cols, numerical_cols


def load_tabular_data(
    train_csv: str,
    target_col: str,
    val_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    categorical_cols: Optional[List[str]] = None,
    numerical_cols: Optional[List[str]] = None,
    batch_size: int = 32,
    shuffle: bool = True,
) -> Tuple[TabularLoaders, ColumnTransformer]:
    """Load and preprocess tabular data from CSV files.

    Args:
        train_csv: Path to the training CSV file.
        target_col: Name of the target column.
        val_csv: Optional path to validation data.
        test_csv: Optional path to test data.
        categorical_cols: Optional list of categorical feature names.
        numerical_cols: Optional list of numerical feature names.
        batch_size: Batch size for the dataloaders.
        shuffle: Whether to shuffle the training data.

    Returns:
        A ``TabularLoaders`` object containing the train/val/test dataloaders
        and the fitted ``ColumnTransformer``.
    """

    train_df = pd.read_csv(train_csv)
    preprocessor, categorical_cols, numerical_cols = _build_preprocessor(
        train_df, target_col, categorical_cols, numerical_cols
    )

    X_train = preprocessor.fit_transform(train_df.drop(columns=[target_col]))
    y_train = train_df[target_col].values
    train_ds = AdDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)

    val_loader = None
    if val_csv:
        val_df = pd.read_csv(val_csv)
        X_val = preprocessor.transform(val_df.drop(columns=[target_col]))
        y_val = val_df[target_col].values
        val_ds = AdDataset(X_val, y_val)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    test_loader = None
    if test_csv:
        test_df = pd.read_csv(test_csv)
        X_test = preprocessor.transform(test_df.drop(columns=[target_col]))
        y_test = test_df[target_col].values
        test_ds = AdDataset(X_test, y_test)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    loaders = TabularLoaders(train_loader, val_loader, test_loader)
    return loaders, preprocessor


def save_preprocessor(preprocessor: ColumnTransformer, filepath: str) -> None:
    """Persist a fitted ``ColumnTransformer`` to disk."""

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(preprocessor, filepath)


def load_preprocessor(filepath: str) -> ColumnTransformer:
    """Load a previously saved ``ColumnTransformer``."""

    return joblib.load(filepath)
