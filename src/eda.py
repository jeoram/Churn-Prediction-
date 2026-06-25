from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run_eda(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution de {col}")
        plt.tight_layout()
        plt.savefig(output_dir / f"hist_{col}.png", dpi=180)
        plt.close()

    plot_cols = [col for col in numeric_cols if col != "churn"]
    box_col = None
    for candidate in ["monthly_premium", "MonthlyCharges", "total_charges", "TotalCharges"]:
        if candidate in df.columns:
            box_col = candidate
            break
    if box_col is None and plot_cols:
        box_col = plot_cols[0]

    if box_col is not None and "churn" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[["churn", box_col]], x="churn", y=box_col)
        plt.title(f"{box_col} par statut de churn")
        plt.tight_layout()
        plt.savefig(output_dir / "boxplot_numeric_churn.png", dpi=180)
        plt.close()

    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Heatmap des variables numériques")
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_numeric.png", dpi=180)
    plt.close()
