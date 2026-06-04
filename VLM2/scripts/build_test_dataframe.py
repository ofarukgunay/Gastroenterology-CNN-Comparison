import argparse
from pathlib import Path

import pandas as pd

from vlm_fewshot_config import IMAGE_EXTENSIONS, PREDICTIONS_DIR, TEST_DIR


def build_test_dataframe(test_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []

    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        for ext in IMAGE_EXTENSIONS:
            for image_path in sorted(class_dir.glob(ext)):
                rows.append(
                    {
                        "image_path": str(image_path),
                        "label": class_dir.name,
                    }
                )

    return pd.DataFrame(rows, columns=["image_path", "label"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build test dataframe CSV from test split.")
    parser.add_argument("--test_dir", type=Path, default=TEST_DIR, help="Path to test split directory.")
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=PREDICTIONS_DIR / "test_files.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--print_only",
        action="store_true",
        help="Do not write CSV, only print summary and first rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {args.test_dir}")

    df = build_test_dataframe(args.test_dir)
    print(f"Rows: {len(df)}")
    print(f"Classes: {df['label'].nunique() if len(df) else 0}")
    if len(df):
        print(df.head(5).to_string(index=False))

    if args.print_only:
        return

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
