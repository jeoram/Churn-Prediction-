from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["claim_to_premium"] = df["claim_amount"] / (df["monthly_premium"] + 1e-6)
    df["contact_per_tenure"] = df["support_contacts"] / (df["tenure_months"] + 1e-6)
    df["premium_per_age"] = df["monthly_premium"] / (df["age"] + 1e-6)
    df["claims_per_tenure"] = df["claims_count"] / (df["tenure_months"] + 1e-6)
    return df


def prepare_training_data(df: pd.DataFrame, target_col: str = "churn") -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    df = engineer_features(df)
    y = df[target_col]
    X = df.drop(columns=[target_col])

    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = []
    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = list(preprocessor.get_feature_names_out())

    return X_train_processed, X_test_processed, y_train, y_test, feature_names
