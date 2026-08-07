import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from _paths import INDICES_PATH, OUTPUTS_DIR


def add_indices(df: pd.DataFrame, indices_file: str | Path = INDICES_PATH) -> pd.DataFrame:
    df = df.copy()

    with Path(indices_file).open("r", encoding="utf-8") as f:
        indices = json.load(f)

    for index_name, bands in indices.get("ND", {}).items():
        col_a, col_b = bands
        df[index_name] = (df[col_a] - df[col_b]) / (df[col_a] + df[col_b])

    for index_name, spec in indices.get("FORMULA", {}).items():
        expression = spec["expression"]
        namespace = {}

        for symbol, band_col in spec.get("bands", {}).items():
            namespace[symbol] = df[band_col]

        for const_name, const_value in spec.get("constants", {}).items():
            namespace[const_name] = const_value

        result = pd.eval(
            expression,
            local_dict=namespace,
            global_dict={"__builtins__": {}},
        )
        df[index_name] = result.mask(~np.isfinite(result))

    return df


def correlation_matrix(
    df: pd.DataFrame,
    font_size: int = 20,
    columns: list[str] | None = None,
    title = None
) -> pd.DataFrame:
    if columns:
        df = df[columns]

    corr_matrix = df.corr(numeric_only=True)

    fig = plt.figure(figsize=(12, 10))
    hm = sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Pearsons R-value"},
    )
    cbar = hm.collections[0].colorbar
    cbar.set_label("Pearsons R-value", labelpad=15, fontsize=font_size - 2)

    plt.tick_params(axis="x", labelsize=font_size - 8)
    plt.tick_params(axis="y", labelsize=font_size - 8)
    if title:
        plt.title(
            title,
            fontweight="bold",
            fontsize=20,
            y=1.02,
            x=0.55,
        )
    plt.show()

    save_path = OUTPUTS_DIR / "corr_matrix.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return corr_matrix
