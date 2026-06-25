from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def ensure_data_file(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "churn_synthetic.csv"
    if not data_path.exists():
        rng = np.random.default_rng(42)
        n_rows = 4000
        df = pd.DataFrame(
            {
                "age": rng.integers(22, 75, size=n_rows),
                "tenure_months": rng.integers(3, 96, size=n_rows),
                "monthly_premium": rng.normal(120, 35, size=n_rows).clip(40, 300),
                "claims_count": rng.poisson(0.8, size=n_rows),
                "claim_amount": rng.gamma(shape=2.0, scale=80.0, size=n_rows),
                "support_contacts": rng.poisson(1.3, size=n_rows),
                "satisfaction_score": rng.integers(1, 6, size=n_rows),
                "payment_delay_days": rng.poisson(2.0, size=n_rows),
                "has_dispute": rng.binomial(1, 0.12, size=n_rows),
                "online_engagement": rng.integers(0, 11, size=n_rows),
                "policy_type": rng.choice(["standard", "premium", "family"], size=n_rows, p=[0.55, 0.25, 0.20]),
                "region": rng.choice(["north", "south", "east", "west"], size=n_rows, p=[0.3, 0.25, 0.2, 0.25]),
            }
        )
        churn_prob = (
            0.02
            + 0.0006 * (df["age"] - 40)
            + 0.001 * df["support_contacts"]
            + 0.0008 * df["payment_delay_days"]
            + 0.03 * (df["satisfaction_score"] < 3)
            + 0.05 * df["has_dispute"]
            - 0.002 * np.clip(df["online_engagement"], 0, 10)
            + 0.01 * (df["policy_type"] == "standard")
            + 0.01 * (df["region"] == "south")
            + 0.003 * np.clip(df["claims_count"], 0, 4)
        )
        churn_prob = np.clip(churn_prob, 0.01, 0.9)
        df["churn"] = rng.binomial(1, churn_prob)
        df["claim_amount"] = df["claim_amount"].round(2)
        df["monthly_premium"] = df["monthly_premium"].round(2)
        df.to_csv(data_path, index=False)
    return data_path


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
