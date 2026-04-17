import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


INPUT_CSV = Path("results/experiment_results_final.csv")
OUT_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_model_name(name: str) -> str:
    if "tiny" in name.lower():
        return "Tiny"
    if "phi-3" in name.lower() or "phi_3" in name.lower() or "phi" in name.lower():
        return "Phi"
    if "llama-2" in name.lower():
        return "LLaMA2"
    if "mistral" in name.lower():
        return "Mistral"
    if "llama-3" in name.lower():
        return "LLaMA3"
    return name


def main(input_csv: Path = INPUT_CSV, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load CSV
    df = pd.read_csv(input_csv)

    # Parse timestamp for stable dedup; fallback if missing/invalid
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Remove exact checkpoint duplicates from resume runs
    key_cols = ["id", "dataset", "model_name", "compressed"]
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").drop_duplicates(subset=key_cols, keep="last")
    else:
        df = df.drop_duplicates(subset=key_cols, keep="last")

    # Ensure numeric types
    numeric_cols = [
        "compressed",
        "orig_tokens",
        "compressed_tokens",
        "exact_match",
        "keyword_match",
        "self_consistency",
        "nli_support",
        "hallucination",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean rows required for analysis
    df = df.dropna(subset=["model_name", "dataset", "compressed", "nli_support", "hallucination"])

    # Helpful label for plots
    df["compression_label"] = np.where(df["compressed"].astype(int) == 1, "compressed", "uncompressed")

    # 2) Aggregated statistics
    mean_nli_model_comp = (
        df.groupby(["model_name", "compression_label"], as_index=False)["nli_support"]
        .mean()
        .rename(columns={"nli_support": "mean_nli_support"})
    )

    mean_hall_model_comp = (
        df.groupby(["model_name", "compression_label"], as_index=False)["hallucination"]
        .mean()
        .rename(columns={"hallucination": "mean_hallucination_rate"})
    )

    mean_hall_model = (
        df.groupby("model_name", as_index=False)["hallucination"]
        .mean()
        .rename(columns={"hallucination": "mean_hallucination_rate"})
    )

    mean_metrics_dataset = (
        df.groupby("dataset", as_index=False)[
            ["exact_match", "keyword_match", "self_consistency", "nli_support", "hallucination"]
        ]
        .mean()
        .rename(
            columns={
                "exact_match": "mean_exact_match",
                "keyword_match": "mean_keyword_match",
                "self_consistency": "mean_self_consistency",
                "nli_support": "mean_nli_support",
                "hallucination": "mean_hallucination_rate",
            }
        )
    )

    # 3) Delta metrics (compressed - uncompressed)
    has_uncompressed = (df["compressed"] == 0).any()
    has_compressed = (df["compressed"] == 1).any()

    if has_uncompressed and has_compressed:
        pair_means = (
            df.groupby(["model_name", "dataset", "compressed"], as_index=False)[["nli_support", "hallucination"]]
            .mean()
        )

        pivot = pair_means.pivot_table(
            index=["model_name", "dataset"],
            columns="compressed",
            values=["nli_support", "hallucination"],
            aggfunc="mean",
        )

        # Ensure both compression states exist before delta
        needed_cols = [
            ("nli_support", 0),
            ("nli_support", 1),
            ("hallucination", 0),
            ("hallucination", 1),
        ]
        for c in needed_cols:
            if c not in pivot.columns:
                pivot[c] = np.nan

        delta_df = pivot.reset_index()
        delta_df["delta_nli"] = delta_df[("nli_support", 1)] - delta_df[("nli_support", 0)]
        delta_df["delta_hallucination"] = delta_df[("hallucination", 1)] - delta_df[("hallucination", 0)]

        # Flatten columns for saving
        delta_df.columns = [
            c if isinstance(c, str) else (c[0] if c[1] == "" else f"{c[0]}_{int(c[1])}")
            for c in delta_df.columns
        ]

        delta_by_model = (
            delta_df.groupby("model_name", as_index=False)[["delta_nli", "delta_hallucination"]]
            .mean()
            .sort_values("delta_nli", ascending=False)
        )

        delta_by_dataset = (
            delta_df.groupby("dataset", as_index=False)[["delta_nli", "delta_hallucination"]]
            .mean()
            .sort_values("delta_nli", ascending=False)
        )
    else:
        delta_df = pd.DataFrame(columns=["model_name", "dataset", "delta_nli", "delta_hallucination"])
        delta_by_model = pd.DataFrame(columns=["model_name", "delta_nli", "delta_hallucination"])
        delta_by_dataset = pd.DataFrame(columns=["dataset", "delta_nli", "delta_hallucination"])

    # Save aggregate/delta CSVs
    mean_nli_model_comp.to_csv(out_dir / "agg_mean_nli_per_model_compression.csv", index=False)
    mean_hall_model_comp.to_csv(out_dir / "agg_mean_hallucination_per_model_compression.csv", index=False)
    mean_hall_model.to_csv(out_dir / "agg_mean_hallucination_per_model.csv", index=False)
    mean_metrics_dataset.to_csv(out_dir / "agg_mean_metrics_per_dataset.csv", index=False)
    delta_df.to_csv(out_dir / "delta_metrics_model_dataset.csv", index=False)
    delta_by_model.to_csv(out_dir / "delta_metrics_by_model.csv", index=False)
    delta_by_dataset.to_csv(out_dir / "delta_metrics_by_dataset.csv", index=False)

    # 4A) Bar plot: model vs hallucination rate, hue=compressed/uncompressed
    hall_plot_df = mean_hall_model_comp.copy()
    models = hall_plot_df["model_name"].unique().tolist()
    x = np.arange(len(models))
    width = 0.38

    y_un = []
    y_co = []
    for m in models:
        mdf = hall_plot_df[hall_plot_df["model_name"] == m]
        un = mdf.loc[mdf["compression_label"] == "uncompressed", "mean_hallucination_rate"]
        co = mdf.loc[mdf["compression_label"] == "compressed", "mean_hallucination_rate"]
        y_un.append(float(un.iloc[0]) if len(un) else np.nan)
        y_co.append(float(co.iloc[0]) if len(co) else np.nan)

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, y_un, width=width, label="uncompressed")
    plt.bar(x + width / 2, y_co, width=width, label="compressed")
    plt.xticks(x, models, rotation=20, ha="right")
    plt.ylabel("Hallucination Rate")
    plt.xlabel("Model")
    plt.title("Hallucination Rate by Model and Compression")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_a_hallucination_by_model_compression.png", dpi=180)
    plt.close()

    # 4B) Bar plot: dataset vs nli_support, hue=compressed/uncompressed
    nli_ds = (
        df.groupby(["dataset", "compression_label"], as_index=False)["nli_support"]
        .mean()
        .rename(columns={"nli_support": "mean_nli_support"})
    )
    datasets = nli_ds["dataset"].unique().tolist()
    x = np.arange(len(datasets))

    y_un = []
    y_co = []
    for d in datasets:
        ddf = nli_ds[nli_ds["dataset"] == d]
        un = ddf.loc[ddf["compression_label"] == "uncompressed", "mean_nli_support"]
        co = ddf.loc[ddf["compression_label"] == "compressed", "mean_nli_support"]
        y_un.append(float(un.iloc[0]) if len(un) else np.nan)
        y_co.append(float(co.iloc[0]) if len(co) else np.nan)

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, y_un, width=width, label="uncompressed")
    plt.bar(x + width / 2, y_co, width=width, label="compressed")
    plt.xticks(x, datasets, rotation=20, ha="right")
    plt.ylabel("Mean NLI Support")
    plt.xlabel("Dataset")
    plt.title("NLI Support by Dataset and Compression")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_b_nli_by_dataset_compression.png", dpi=180)
    plt.close()

    # 4C) Line plot: model size order vs hallucination rate
    order = ["Tiny", "Phi", "LLaMA2", "Mistral", "LLaMA3"]
    line_df = mean_hall_model.copy()
    line_df["model_family"] = line_df["model_name"].map(normalize_model_name)
    line_df = line_df.groupby("model_family", as_index=False)["mean_hallucination_rate"].mean()
    line_df["order_idx"] = line_df["model_family"].map({k: i for i, k in enumerate(order)})
    line_df = line_df.dropna(subset=["order_idx"]).sort_values("order_idx")

    plt.figure(figsize=(9, 5))
    plt.plot(line_df["model_family"], line_df["mean_hallucination_rate"], marker="o")
    plt.xlabel("Model Size Order")
    plt.ylabel("Mean Hallucination Rate")
    plt.title("Hallucination Rate Across Model Size Order")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_c_hallucination_model_order.png", dpi=180)
    plt.close()

    # 4D) Scatter plot: prompt token count vs hallucination rate
    plt.figure(figsize=(10, 6))
    un_df = df[df["compressed"].astype(int) == 0]
    co_df = df[df["compressed"].astype(int) == 1]

    plt.scatter(
        un_df["orig_tokens"],
        un_df["hallucination"],
        s=10,
        alpha=0.25,
        label="uncompressed",
    )
    plt.scatter(
        co_df["orig_tokens"],
        co_df["hallucination"],
        s=10,
        alpha=0.25,
        label="compressed",
    )
    plt.xlabel("Prompt Token Count")
    plt.ylabel("Hallucination Rate")
    plt.title("Prompt Length vs Hallucination")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plot_d_tokens_vs_hallucination_scatter.png", dpi=180)
    plt.close()

    # 6) Summary insights
    best_model = delta_by_model.iloc[0] if len(delta_by_model) else None
    best_dataset = delta_by_dataset.iloc[0] if len(delta_by_dataset) else None
    worst_model = (
        delta_by_model.sort_values("delta_nli", ascending=True).iloc[0]
        if len(delta_by_model)
        else None
    )
    best_overall_model = mean_hall_model.sort_values("mean_hallucination_rate", ascending=True).iloc[0]
    best_overall_dataset = mean_metrics_dataset.sort_values("mean_hallucination_rate", ascending=True).iloc[0]

    print("=== Aggregated Statistics (head) ===")
    print("\nMean NLI support per model (compressed vs uncompressed):")
    print(mean_nli_model_comp.sort_values(["model_name", "compression_label"]).to_string(index=False))

    print("\nMean hallucination rate per model:")
    print(mean_hall_model.sort_values("model_name").to_string(index=False))

    print("\nMean metrics per dataset:")
    print(mean_metrics_dataset.sort_values("dataset").to_string(index=False))

    print("\n=== Delta Metrics Summary ===")
    if len(delta_by_model) and len(delta_by_dataset):
        print("\nDelta by model (compressed - uncompressed):")
        print(delta_by_model.to_string(index=False))

        print("\nDelta by dataset (compressed - uncompressed):")
        print(delta_by_dataset.to_string(index=False))
    else:
        print("\nDelta metrics are not available in this CSV because only one compression state is present.")

    print("\n=== Insights ===")
    if best_model is not None:
        print(
            f"Model improved most with compression (highest ΔNLI): {best_model['model_name']} "
            f"(ΔNLI={best_model['delta_nli']:.4f}, ΔHall={best_model['delta_hallucination']:.4f})"
        )
    else:
        print("Model compression improvement cannot be determined from this file (compressed=1 rows are missing).")

    if best_dataset is not None:
        print(
            f"Dataset benefiting most (highest mean ΔNLI): {best_dataset['dataset']} "
            f"(ΔNLI={best_dataset['delta_nli']:.4f}, ΔHall={best_dataset['delta_hallucination']:.4f})"
        )
    else:
        print("Dataset compression benefit cannot be determined from this file (compressed=1 rows are missing).")

    if worst_model is not None:
        print(
            f"Largest negative model effect (lowest ΔNLI): {worst_model['model_name']} "
            f"(ΔNLI={worst_model['delta_nli']:.4f}, ΔHall={worst_model['delta_hallucination']:.4f})"
        )
    else:
        print("Negative compression effects cannot be determined from this file (compressed=1 rows are missing).")

    print(
        f"Best overall model by lowest hallucination rate (available data): {best_overall_model['model_name']} "
        f"({best_overall_model['mean_hallucination_rate']:.4f})"
    )
    print(
        f"Best overall dataset by lowest hallucination rate (available data): {best_overall_dataset['dataset']} "
        f"({best_overall_dataset['mean_hallucination_rate']:.4f})"
    )

    print("\nSaved plots:")
    print((out_dir / "plot_a_hallucination_by_model_compression.png").as_posix())
    print((out_dir / "plot_b_nli_by_dataset_compression.png").as_posix())
    print((out_dir / "plot_c_hallucination_model_order.png").as_posix())
    print((out_dir / "plot_d_tokens_vs_hallucination_scatter.png").as_posix())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment results CSV and produce aggregate tables/plots.")
    parser.add_argument("--input", default=str(INPUT_CSV), help="Input CSV path")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory for generated files")
    args = parser.parse_args()
    main(input_csv=Path(args.input), out_dir=Path(args.out_dir))
