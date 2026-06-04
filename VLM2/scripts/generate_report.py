import argparse
from pathlib import Path

import pandas as pd

from vlm_fewshot_config import METRICS_DIR, VLM2_ROOT, VLM_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final VLM few-shot markdown report.")
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=METRICS_DIR / "summary_metrics.csv",
        help="Path to summary metrics CSV.",
    )
    parser.add_argument(
        "--comparison_csv",
        type=Path,
        default=METRICS_DIR / "model_comparison.csv",
        help="Path to comparison CSV (optional).",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=VLM2_ROOT / "reports" / "vlm_fewshot_report.md",
        help="Output markdown report path.",
    )
    parser.add_argument("--print_only", action="store_true", help="Do not write report file.")
    return parser.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    cols = [str(c) for c in df.columns]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            if pd.isna(v):
                vals.append("")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def to_numeric_if_exists(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def pick_best_and_fastest(summary_df: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None]:
    perf = summary_df.dropna(subset=["macro_f1"]) if "macro_f1" in summary_df.columns else pd.DataFrame()
    best = None if perf.empty else perf.loc[perf["macro_f1"].idxmax()]

    speed = summary_df.dropna(subset=["avg_inference_time_sec"]) if "avg_inference_time_sec" in summary_df.columns else pd.DataFrame()
    speed = speed[speed["avg_inference_time_sec"] > 0] if not speed.empty else speed
    fastest = None if speed.empty else speed.loc[speed["avg_inference_time_sec"].idxmin()]
    return best, fastest


def format_float(x, digits: int = 4) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "-"


def model_display_name(model_key: str) -> str:
    for m in VLM_MODELS:
        if m["name"] == model_key:
            return f"{model_key} ({m['hf_id']})"
    return model_key


def build_report_md(summary_df: pd.DataFrame | None, comparison_df: pd.DataFrame | None) -> str:
    lines: list[str] = []
    lines.append("# VLM Few-Shot Comparison Report")
    lines.append("")
    lines.append("## Objective")
    lines.append("This report compares 6 Vision-Language Models on the same dataset for few-shot image classification.")
    lines.append("")

    lines.append("## Models")
    for i, m in enumerate(VLM_MODELS, start=1):
        lines.append(f"{i}. `{m['name']}` - `{m['hf_id']}`")
    lines.append("")

    lines.append("## Data and Hardware")
    lines.append("- Data root: `data/prepared-data`")
    lines.append("- GPU target: NVIDIA RTX 4060 (8 GB assumption)")
    lines.append("- Batch size: 1")
    lines.append("- Quantization: model-level setting (4-bit/FP16 from config)")
    lines.append("")

    lines.append("## Experiment Scope")
    lines.append("- Shot values: 0, 1, 3, 5")
    lines.append("- Prompt languages: EN (required), TR (optional)")
    lines.append("- Metrics: Accuracy, Macro F1, Weighted F1, Average inference time")
    lines.append("")

    if summary_df is None:
        lines.append("## Results")
        lines.append("No `summary_metrics.csv` found yet, so the metrics table is empty.")
        lines.append("")
        lines.append("## Best / Fastest Model")
        lines.append("- Best model: -")
        lines.append("- Fastest model: -")
        lines.append("")
        lines.append("## Note")
        lines.append("Run prediction first (`run_vlm_fewshot.py`), then evaluate (`evaluate_vlm_results.py`).")
        lines.append("")
        return "\n".join(lines)

    required = ["model", "shots", "accuracy", "macro_f1", "weighted_f1", "avg_inference_time_sec", "num_samples"]
    missing = [c for c in required if c not in summary_df.columns]
    if missing:
        lines.append("## Results")
        lines.append(f"`summary_metrics.csv` has missing columns: {missing}")
        lines.append("")
        return "\n".join(lines)

    summary_df = to_numeric_if_exists(
        summary_df,
        ["shots", "accuracy", "macro_f1", "weighted_f1", "avg_inference_time_sec", "num_samples"],
    )
    summary_df = summary_df.sort_values(by=["shots", "macro_f1"], ascending=[True, False], na_position="last")

    display_df = summary_df.copy()
    display_df["Model"] = display_df["model"].astype(str)
    display_df["Shot"] = display_df["shots"]
    display_df["Accuracy"] = display_df["accuracy"].map(lambda v: format_float(v, 4))
    display_df["Macro F1"] = display_df["macro_f1"].map(lambda v: format_float(v, 4))
    display_df["Weighted F1"] = display_df["weighted_f1"].map(lambda v: format_float(v, 4))
    display_df["Avg Time (s)"] = display_df["avg_inference_time_sec"].map(lambda v: format_float(v, 4))
    display_df["Samples"] = display_df["num_samples"].fillna(0).astype(int)
    display_df = display_df[["Model", "Shot", "Accuracy", "Macro F1", "Weighted F1", "Avg Time (s)", "Samples"]]

    lines.append("## Results")
    lines.append(dataframe_to_markdown(display_df))
    lines.append("")

    best, fastest = pick_best_and_fastest(summary_df)
    lines.append("## Best / Fastest Model")
    if best is None:
        lines.append("- Best model (Macro F1): -")
    else:
        lines.append(
            f"- Best model (Macro F1): `{best['model']}` | shot={int(best['shots']) if pd.notna(best['shots']) else '-'} | "
            f"macro_f1={format_float(best['macro_f1'])} | accuracy={format_float(best['accuracy'])}"
        )
    if fastest is None:
        lines.append("- Fastest model (Avg Time): -")
    else:
        lines.append(
            f"- Fastest model (Avg Time): `{fastest['model']}` | shot={int(fastest['shots']) if pd.notna(fastest['shots']) else '-'} | "
            f"avg_time={format_float(fastest['avg_inference_time_sec'])}s"
        )
    lines.append("")

    if comparison_df is not None:
        lines.append("## Comparison Summary")
        cols = [c for c in ["model", "shots", "accuracy", "macro_f1", "weighted_f1", "avg_inference_time_sec"] if c in comparison_df.columns]
        if cols:
            lines.append(dataframe_to_markdown(comparison_df[cols]))
        else:
            lines.append("Expected columns were not found in `model_comparison.csv`.")
        lines.append("")

    lines.append("## Notes")
    lines.append("- Track performance changes as shot count increases.")
    lines.append("- On RTX 4060, use 4-bit and low `max_new_tokens` for heavy models.")
    lines.append("")

    lines.append("## Conclusion")
    lines.append("- Performance-focused choice: highest Macro F1 model")
    lines.append("- Speed-focused choice: lowest average inference time model")
    lines.append("- RTX 4060 compatibility: model that finishes without OOM and keeps a good speed/quality balance")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary_df = safe_read_csv(args.summary_csv)
    comparison_df = safe_read_csv(args.comparison_csv)
    report_md = build_report_md(summary_df, comparison_df)

    if args.print_only:
        print(report_md)
        return

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report_md, encoding="utf-8")
    print(f"Saved report: {args.output_md}")


if __name__ == "__main__":
    main()
