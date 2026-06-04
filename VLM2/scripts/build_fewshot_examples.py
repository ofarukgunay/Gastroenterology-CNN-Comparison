import argparse
import json
import random
from pathlib import Path

from vlm_fewshot_config import IMAGE_EXTENSIONS, PREDICTIONS_DIR, SEED, TRAIN_DIR


def list_images(class_dir: Path) -> list[Path]:
    images: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(class_dir.glob(ext))
    return sorted(images)


def collect_classes(train_dir: Path) -> dict[str, list[Path]]:
    class_to_images: dict[str, list[Path]] = {}
    for class_dir in sorted(train_dir.iterdir()):
        if class_dir.is_dir():
            class_to_images[class_dir.name] = list_images(class_dir)
    return class_to_images


def build_fewshot_examples(train_dir: Path, shots_per_class: int, seed: int) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    class_to_images = collect_classes(train_dir)
    examples: list[dict] = []
    stats: dict[str, int] = {}

    for class_name, images in class_to_images.items():
        if not images:
            stats[class_name] = 0
            continue

        sample_count = min(shots_per_class, len(images))
        selected = rng.sample(images, sample_count)
        stats[class_name] = sample_count

        for image_path in selected:
            examples.append(
                {
                    "image_path": str(image_path),
                    "label": class_name,
                }
            )

    rng.shuffle(examples)
    return examples, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build balanced few-shot examples from train split.")
    parser.add_argument("--train_dir", type=Path, default=TRAIN_DIR, help="Path to train split directory.")
    parser.add_argument("--shots", type=int, default=3, help="Number of examples per class.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Output JSON path. Default: VLM2/outputs/vlm_fewshot/predictions/fewshot_shot{shots}_seed{seed}.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dir = args.train_dir

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    if args.shots < 0:
        raise ValueError("--shots must be >= 0")

    examples, stats = build_fewshot_examples(train_dir=train_dir, shots_per_class=args.shots, seed=args.seed)

    default_output = PREDICTIONS_DIR / f"fewshot_shot{args.shots}_seed{args.seed}.json"
    output_path = args.output_json or default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "train_dir": str(train_dir),
        "shots": args.shots,
        "seed": args.seed,
        "num_classes": len(stats),
        "num_examples": len(examples),
        "selected_per_class": stats,
        "examples": examples,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved: {output_path}")
    print(f"Classes: {len(stats)} | Total examples: {len(examples)}")
    for class_name, count in stats.items():
        print(f"- {class_name}: {count}")


if __name__ == "__main__":
    main()
