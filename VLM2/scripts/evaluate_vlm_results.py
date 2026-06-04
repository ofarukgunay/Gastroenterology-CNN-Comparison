import argparse
import glob
import json
import time
from pathlib import Path

import pandas as pd

from vlm_fewshot_config import METRICS_DIR, PREDICTIONS_DIR


def parse_model_shot_from_name(path: Path) -> tuple[str, int | None]:
    stem = path.stem
    if "_shot" not in stem:
        return stem, None
    model_name, shot_part = stem.rsplit("_shot", 1)
    try:
        return model_name, int(shot_part)
    except ValueError:
        return model_name, None


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _compute_classification_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> tuple[dict, list[list[int]]]:
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    cm = [[0 for _ in range(n)] for _ in range(n)]

    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t]][label_to_idx[p]] += 1

    per_class = {}
    precision_list: list[float] = []
    recall_list: list[float] = []
    f1_list: list[float] = []
    supports: list[int] = []

    for i, label in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n) if r != i)
        fn = sum(cm[i][c] for c in range(n) if c != i)
        support = sum(cm[i])

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": int(support),
        }

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        supports.append(support)

    total_support = sum(supports)
    weighted_precision = _safe_div(sum(p * s for p, s in zip(precision_list, supports)), total_support)
    weighted_recall = _safe_div(sum(r * s for r, s in zip(recall_list, supports)), total_support)
    weighted_f1 = _safe_div(sum(f * s for f, s in zip(f1_list, supports)), total_support)

    macro_precision = _safe_div(sum(precision_list), len(labels))
    macro_recall = _safe_div(sum(recall_list), len(labels))
    macro_f1 = _safe_div(sum(f1_list), len(labels))

    report = dict(per_class)
    report["macro avg"] = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1-score": macro_f1,
        "support": int(total_support),
    }
    report["weighted avg"] = {
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1-score": weighted_f1,
        "support": int(total_support),
    }
    report["accuracy"] = _safe_div(sum(cm[i][i] for i in range(n)), total_support)
    return report, cm


def evaluate_single_csv(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    required_cols = {"true_label", "pred_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path.name}: {sorted(missing)}")

    y_true = df["true_label"].astype(str)
    y_pred = df["pred_label"].astype(str)
    y_true_list = y_true.tolist()
    y_pred_list = y_pred.tolist()

    model_name, shot = parse_model_shot_from_name(csv_path)
    if "model" in df.columns and not df["model"].empty:
        model_name = str(df["model"].iloc[0])
    if "shots" in df.columns and not df["shots"].empty:
        try:
            shot = int(df["shots"].iloc[0])
        except Exception:
            pass

    labels = sorted(set(y_true_list) | set(y_pred_list))
    report, cm = _compute_classification_metrics(y_true_list, y_pred_list, labels)

    accuracy = float(report["accuracy"])
    macro_f1 = float(report["macro avg"]["f1-score"])
    weighted_f1 = float(report["weighted avg"]["f1-score"])

    metrics = {
        "file": str(csv_path),
        "model": model_name,
        "shots": shot,
        "num_samples": int(len(df)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    if "inference_time_sec" in df.columns:
        try:
            metrics["avg_inference_time_sec"] = float(pd.to_numeric(df["inference_time_sec"], errors="coerce").mean())
        except Exception:
            metrics["avg_inference_time_sec"] = None
    else:
        metrics["avg_inference_time_sec"] = None

    return {
        "metrics": metrics,
        "labels": labels,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def find_prediction_files(input_glob: str) -> list[Path]:
    files = sorted(Path(p) for p in glob.glob(input_glob))
    return [p for p in files if p.is_file() and p.suffix.lower() == ".csv"]


def export_confusion_png(cm_path: Path, labels: list[str], cm_data: list[list[int]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.array(cm_data, dtype=int)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix - {cm_path.stem.replace('_confusion_matrix', '')}")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    out_png = cm_path.with_suffix(".png")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def write_eval_artifacts(result: dict, output_dir: Path, run_tag: str, export_png: bool = True) -> None:
    metrics = result["metrics"]
    model = metrics["model"]
    shots = metrics["shots"]
    source_stem = Path(metrics["file"]).stem
    stem = f"{source_stem}_{run_tag}" if run_tag else source_stem

    output_dir.mkdir(parents=True, exist_ok=True)
    cm_path = output_dir / f"{stem}_confusion_matrix.json"
    report_path = output_dir / f"{stem}_classification_report.json"

    with cm_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"labels": result["labels"], "confusion_matrix": result["confusion_matrix"]},
            f,
            ensure_ascii=False,
            indent=2,
        )
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(result["classification_report"], f, ensure_ascii=False, indent=2)

    if export_png:
        try:
            export_confusion_png(cm_path, result["labels"], result["confusion_matrix"])
        except Exception as ex:
            print(f"[WARN] PNG export failed for {cm_path.name}: {ex}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VLM prediction CSV files.")
    parser.add_argument(
        "--input_glob",
        type=str,
        default=str(PREDICTIONS_DIR / "*_shot*.csv"),
        help="Glob pattern for prediction CSV files.",
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=METRICS_DIR / "summary_metrics.csv",
        help="Output summary CSV path.",
    )
    parser.add_argument(
        "--details_dir",
        type=Path,
        default=METRICS_DIR / "details",
        help="Output directory for confusion matrix and classification reports.",
    )
    parser.add_argument(
        "--no_png",
        action="store_true",
        help="Disable confusion matrix PNG export.",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="Optional tag appended to details filenames. Default: current timestamp.",
    )
    parser.add_argument("--print_only", action="store_true", help="Print metrics without writing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_tag = args.run_tag or time.strftime("%Y%m%d_%H%M%S")
    files = find_prediction_files(args.input_glob)
    if not files:
        print(f"No prediction files found for pattern: {args.input_glob}")
        return

    rows: list[dict] = []
    evaluated = 0
    failed = 0

    for csv_path in files:
        try:
            result = evaluate_single_csv(csv_path)
            rows.append(result["metrics"])
            evaluated += 1
            print(
                f"[OK] {csv_path.name} | acc={result['metrics']['accuracy']:.4f} | "
                f"macro_f1={result['metrics']['macro_f1']:.4f} | weighted_f1={result['metrics']['weighted_f1']:.4f}"
            )
            if not args.print_only:
                write_eval_artifacts(result, args.details_dir, run_tag=run_tag, export_png=not args.no_png)
        except Exception as ex:
            failed += 1
            print(f"[FAIL] {csv_path.name}: {ex}")

    if not rows:
        print("No files evaluated successfully.")
        return

    summary_df = pd.DataFrame(rows).sort_values(by=["model", "shots"], na_position="last")

    if args.print_only:
        print("\nSummary:")
        print(summary_df.to_string(index=False))
        print(f"\nEvaluated: {evaluated} | Failed: {failed}")
        return

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.summary_csv, index=False, encoding="utf-8")
    print(f"\nSaved summary: {args.summary_csv}")
    print(f"Saved details to: {args.details_dir}")
    print(f"Evaluated: {evaluated} | Failed: {failed}")


if __name__ == "__main__":
    main()
