import argparse
from pathlib import Path

import pandas as pd

from vlm_fewshot_config import METRICS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare VLM models from summary metrics CSV.")
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=METRICS_DIR / "summary_metrics.csv",
        help="Path to summary metrics CSV from evaluate_vlm_results.py",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=METRICS_DIR / "model_comparison.csv",
        help="Output comparison CSV.",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=METRICS_DIR / "model_comparison.md",
        help="Output markdown report.",
    )
    parser.add_argument("--print_only", action="store_true", help="Do not write files, only print results.")
    return parser.parse_args()


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


def validate_columns(df: pd.DataFrame) -> None:
    required = {"model", "shots", "accuracy", "macro_f1", "weighted_f1", "avg_inference_time_sec", "num_samples"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def best_row(df: pd.DataFrame, score_col: str, higher_is_better: bool) -> pd.Series:
    if higher_is_better:
        idx = df[score_col].idxmax()
    else:
        idx = df[score_col].idxmin()
    return df.loc[idx]


def build_overall_summary(df: pd.DataFrame) -> dict:
    valid_perf = df.dropna(subset=["macro_f1"])
    if valid_perf.empty:
        best_perf = None
    else:
        best_perf = best_row(valid_perf, "macro_f1", higher_is_better=True)

    valid_speed = df.dropna(subset=["avg_inference_time_sec"])
    valid_speed = valid_speed[valid_speed["avg_inference_time_sec"] > 0]
    if valid_speed.empty:
        fastest = None
    else:
        fastest = best_row(valid_speed, "avg_inference_time_sec", higher_is_better=False)

    return {
        "best_performance": best_perf,
        "fastest_model": fastest,
    }


def build_shot_leaders(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for shot, group in df.groupby("shots", dropna=False):
        perf_group = group.dropna(subset=["macro_f1"])
        speed_group = group.dropna(subset=["avg_inference_time_sec"])
        speed_group = speed_group[speed_group["avg_inference_time_sec"] > 0]

        best_perf_model = None
        best_perf_macro_f1 = None
        if not perf_group.empty:
            best_perf = best_row(perf_group, "macro_f1", higher_is_better=True)
            best_perf_model = best_perf["model"]
            best_perf_macro_f1 = best_perf["macro_f1"]

        fastest_model = None
        fastest_time = None
        if not speed_group.empty:
            fast = best_row(speed_group, "avg_inference_time_sec", higher_is_better=False)
            fastest_model = fast["model"]
            fastest_time = fast["avg_inference_time_sec"]

        rows.append(
            {
                "shots": shot,
                "best_macro_f1_model": best_perf_model,
                "best_macro_f1": best_perf_macro_f1,
                "fastest_model": fastest_model,
                "fastest_avg_inference_time_sec": fastest_time,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "shots",
                "best_macro_f1_model",
                "best_macro_f1",
                "fastest_model",
                "fastest_avg_inference_time_sec",
            ]
        )
    return pd.DataFrame(rows).sort_values(by=["shots"], na_position="last")


def to_markdown(overall: dict, shot_leaders: pd.DataFrame, full_df: pd.DataFrame) -> str:
    lines = ["# VLM Model Comparison", ""]

    best_perf = overall["best_performance"]
    fastest = overall["fastest_model"]

    lines.append("## Overall")
    if best_perf is None:
        lines.append("- Best performance (Macro F1): N/A")
    else:
        lines.append(
            f"- Best performance (Macro F1): {best_perf['model']} "
            f"(shot={best_perf['shots']}, macro_f1={best_perf['macro_f1']:.4f}, accuracy={best_perf['accuracy']:.4f})"
        )
    if fastest is None:
        lines.append("- Fastest model: N/A")
    else:
        lines.append(
            f"- Fastest model: {fastest['model']} "
            f"(shot={fastest['shots']}, avg_time={float(fastest['avg_inference_time_sec']):.4f}s)"
        )
    lines.append("")

    lines.append("## Shot Leaders")
    if shot_leaders.empty:
        lines.append("No shot-level data available.")
    else:
        lines.append(dataframe_to_markdown(shot_leaders))
    lines.append("")

    lines.append("## Full Table")
    lines.append(dataframe_to_markdown(full_df))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.summary_csv.exists():
        print(f"Summary CSV not found: {args.summary_csv}")
        return

    df = pd.read_csv(args.summary_csv)
    if df.empty:
        print(f"Summary CSV is empty: {args.summary_csv}")
        return

    validate_columns(df)

    # Numeric safety
    for col in ["shots", "accuracy", "macro_f1", "weighted_f1", "avg_inference_time_sec", "num_samples"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(by=["shots", "macro_f1", "accuracy"], ascending=[True, False, False], na_position="last")
    shot_leaders = build_shot_leaders(df)
    overall = build_overall_summary(df)
    markdown_text = to_markdown(overall, shot_leaders, df)

    if args.print_only:
        print("=== Overall ===")
        best_perf = overall["best_performance"]
        fastest = overall["fastest_model"]
        if best_perf is not None:
            print(
                f"Best Macro F1: {best_perf['model']} | shot={best_perf['shots']} | "
                f"macro_f1={best_perf['macro_f1']:.4f} | acc={best_perf['accuracy']:.4f}"
            )
        else:
            print("Best Macro F1: N/A")

        if fastest is not None:
            print(
                f"Fastest: {fastest['model']} | shot={fastest['shots']} | "
                f"avg_time={float(fastest['avg_inference_time_sec']):.4f}s"
            )
        else:
            print("Fastest: N/A")

        print("\n=== Shot Leaders ===")
        if shot_leaders.empty:
            print("No data.")
        else:
            print(shot_leaders.to_string(index=False))
        return

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8")
    args.output_md.write_text(markdown_text, encoding="utf-8")

    print(f"Saved comparison CSV: {args.output_csv}")
    print(f"Saved comparison MD:  {args.output_md}")


if __name__ == "__main__":
    main()
