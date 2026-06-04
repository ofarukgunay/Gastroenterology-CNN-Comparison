import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vlm_fewshot_config import METRICS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export confusion matrix JSON files to PNG heatmaps.")
    parser.add_argument(
        "--input_glob",
        type=str,
        default=str(METRICS_DIR / "details" / "*_confusion_matrix.json"),
        help="Glob pattern for confusion matrix JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional output directory for PNG files. Default: same directory as each input JSON.",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Image DPI.")
    parser.add_argument("--fig_w", type=float, default=10.0, help="Figure width in inches.")
    parser.add_argument("--fig_h", type=float, default=8.0, help="Figure height in inches.")
    parser.add_argument("--dry_run", action="store_true", help="List planned outputs without writing files.")
    return parser.parse_args()


def load_confusion_json(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    labels = payload.get("labels", [])
    matrix = np.array(payload.get("confusion_matrix", []), dtype=int)
    if matrix.ndim != 2:
        raise ValueError("confusion_matrix must be a 2D array")
    if len(labels) != matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("labels count must match square confusion matrix size")
    return labels, matrix


def build_output_path(input_path: Path, output_dir: Path | None) -> Path:
    stem = input_path.stem.replace("_confusion_matrix", "")
    out_name = f"{stem}_confusion_matrix.png"
    if output_dir is None:
        return input_path.with_suffix(".png")
    return output_dir / out_name


def draw_and_save(labels: list[str], cm: np.ndarray, out_path: Path, title: str, dpi: int, fig_w: float, fig_h: float) -> None:
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    files = sorted(Path(p) for p in glob.glob(args.input_glob))
    files = [p for p in files if p.is_file()]

    if not files:
        print(f"No files found for pattern: {args.input_glob}")
        return

    generated = 0
    failed = 0
    for path in files:
        try:
            labels, cm = load_confusion_json(path)
            out_path = build_output_path(path, args.output_dir)
            title = f"Confusion Matrix - {path.stem.replace('_confusion_matrix', '')}"
            if args.dry_run:
                print(f"[DRY] {path} -> {out_path}")
                generated += 1
                continue
            draw_and_save(labels, cm, out_path, title, args.dpi, args.fig_w, args.fig_h)
            print(f"[OK] {out_path}")
            generated += 1
        except Exception as ex:
            print(f"[FAIL] {path}: {ex}")
            failed += 1

    print(f"Done. Generated: {generated} | Failed: {failed}")


if __name__ == "__main__":
    main()
