from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def train_models(X_train: Any, y_train: Any) -> dict[str, Any]:
    models: dict[str, Any] = {}

    logistic = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
    logistic.fit(X_train, y_train)
    models["logistic_regression"] = logistic

    rf = RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    models["random_forest"] = rf

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    models["xgboost"] = xgb

    return models
