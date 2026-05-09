import argparse
import csv
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "prepared-data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "vlm"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_DESCRIPTIONS = {
    "dyed-lifted-polyps": "a chromoendoscopy image where dye highlights lifted polyp tissue before removal",
    "dyed-resection-margins": "a chromoendoscopy image showing the dyed margin or border after tissue resection",
    "esophagitis": "inflammation or irritation in the esophagus, often near the gastroesophageal junction",
    "normal-cecum": "a normal cecum view, often including the appendiceal orifice or ileocecal valve region",
    "normal-pylorus": "a normal pylorus view showing the gastric outlet opening",
    "normal-z-line": "a normal z-line view at the gastroesophageal junction",
    "polyps": "an endoscopy image containing one or more visible polyp lesions",
    "ulcerative-colitis": "inflamed colon mucosa with findings compatible with ulcerative colitis",
}


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: str


def collect_samples(data_dir: Path, split: str) -> Tuple[List[str], List[ImageSample]]:
    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    class_names = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
    samples: List[ImageSample] = []
    for class_name in class_names:
        class_dir = split_dir / class_name
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(ImageSample(path=image_path, label=class_name))
    return class_names, samples


def limit_samples_per_class(
    samples: Sequence[ImageSample],
    max_samples_per_class: Optional[int],
    seed: int,
) -> List[ImageSample]:
    if not max_samples_per_class:
        return list(samples)

    rng = random.Random(seed)
    grouped: Dict[str, List[ImageSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.label, []).append(sample)

    limited: List[ImageSample] = []
    for label in sorted(grouped):
        class_samples = grouped[label][:]
        rng.shuffle(class_samples)
        limited.extend(sorted(class_samples[:max_samples_per_class], key=lambda s: str(s.path)))
    return limited


def select_support_examples(
    data_dir: Path,
    split: str,
    class_names: Sequence[str],
    shots: int,
    seed: int,
) -> List[ImageSample]:
    if shots <= 0:
        return []

    support_dir = data_dir / split
    if not support_dir.is_dir():
        raise FileNotFoundError(f"Support split folder not found: {support_dir}")

    rng = random.Random(seed)
    examples: List[ImageSample] = []
    for class_name in class_names:
        class_dir = support_dir / class_name
        image_paths = [
            p for p in sorted(class_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if len(image_paths) < shots:
            raise ValueError(
                f"Class '{class_name}' has {len(image_paths)} images in {support_dir}, "
                f"but {shots} shots were requested."
            )
        rng.shuffle(image_paths)
        examples.extend(ImageSample(path=p, label=class_name) for p in image_paths[:shots])
    return examples


def build_classification_prompt(
    class_names: Sequence[str],
    has_support_images: bool,
    prompt_style: str = "standard",
) -> str:
    support_sentence = (
        "Use the labeled example images as references before classifying the final query image."
        if has_support_images
        else "Use the class list and the visual content of the image."
    )

    if prompt_style == "choice":
        choices = "\n".join(f"{index}. {name}" for index, name in enumerate(class_names, start=1))
        return (
            "You are a medical image classification assistant for gastrointestinal endoscopy images.\n"
            f"{support_sentence}\n"
            "Classify only the final query image by choosing one option.\n\n"
            "Options:\n"
            f"{choices}\n\n"
            "Return only the option number and class label, for example: 3. esophagitis"
        )

    if prompt_style == "medical_choice":
        choices = "\n".join(
            f"{index}. {name}: {CLASS_DESCRIPTIONS.get(name, 'gastrointestinal endoscopy class')}"
            for index, name in enumerate(class_names, start=1)
        )
        return (
            "You are a medical image classification assistant for gastrointestinal endoscopy images.\n"
            f"{support_sentence}\n"
            "The disease or landmark may depend on small mucosal texture, color, shape, or anatomical details.\n"
            "Classify only the final query image. Use any reference images only as visual examples of the labels.\n\n"
            "Valid options:\n"
            f"{choices}\n\n"
            "Return exactly one option number and exact class label from the list above, for example: "
            "3. esophagitis\n"
            "Do not explain. Do not invent a new label."
        )

    classes = "\n".join(f"- {name}" for name in class_names)

    if prompt_style == "strict":
        return (
            "You are a medical image classification assistant for gastrointestinal endoscopy images.\n"
            f"{support_sentence}\n"
            "Classify only the final query image. Do not classify the reference examples.\n\n"
            "Valid class labels:\n"
            f"{classes}\n\n"
            "Answer with exactly one valid class label from the list above. Do not use synonyms. Do not explain."
        )

    return (
        "You are a medical image classification assistant for gastrointestinal endoscopy images.\n"
        f"{support_sentence}\n\n"
        "Classify the query image into exactly one of these classes:\n"
        f"{classes}\n\n"
        "Return only one class name. Do not explain."
    )


def build_qwen_messages(
    image_path: Path,
    class_names: Sequence[str],
    support_examples: Sequence[ImageSample],
    few_shot_format: str,
    prompt_style: str,
) -> List[dict]:
    prompt = build_classification_prompt(class_names, bool(support_examples), prompt_style)

    if support_examples and few_shot_format == "conversation":
        messages: List[dict] = []
        for example in support_examples:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(example.path)},
                        {
                            "type": "text",
                            "text": "This is a labeled reference image. Return its class label only.",
                        },
                    ],
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": example.label}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        )
        return messages

    content = []
    if support_examples:
        content.append({"type": "text", "text": "Labeled reference examples:"})
        for example in support_examples:
            content.append({"type": "image", "image": str(example.path)})
            content.append({"type": "text", "text": f"Class: {example.label}"})
        content.append({"type": "text", "text": "Now classify the query image."})
    content.append({"type": "image", "image": str(image_path)})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def normalize_answer(raw_answer: str, class_names: Sequence[str]) -> str:
    answer = (raw_answer or "").strip().lower()
    if not answer:
        return "unknown"

    def normalize_text(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    normalized_answer = normalize_text(answer)
    matches = []
    for class_name in class_names:
        normalized_class = normalize_text(class_name)
        index = normalized_answer.find(normalized_class)
        if index >= 0:
            matches.append((index, class_name))
    if matches:
        return sorted(matches, key=lambda item: item[0])[0][1]

    option_match = re.search(r"\b(?:option|class|label|choice|answer)\s*[:#.-]?\s*([1-9][0-9]?)\b", normalized_answer)
    if not option_match:
        option_match = re.fullmatch(r"([1-9][0-9]?)", normalized_answer)
    if option_match:
        option_index = int(option_match.group(1))
        if 1 <= option_index <= len(class_names):
            return class_names[option_index - 1]

    aliases = {
        "dyed lifted polyps": "dyed-lifted-polyps",
        "dyed resection margins": "dyed-resection-margins",
        "z line": "normal-z-line",
        "normal z line": "normal-z-line",
        "cecum": "normal-cecum",
        "pylorus": "normal-pylorus",
        "ulcerative colitis": "ulcerative-colitis",
        "esophagitis": "esophagitis",
        "polyps": "polyps",
    }
    for alias, class_name in aliases.items():
        if alias in normalized_answer and class_name in class_names:
            return class_name

    return "unknown"


def open_rgb_image(path: Path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def _fit_image_on_canvas(image, size: Tuple[int, int], fill: str = "white"):
    from PIL import Image

    canvas = Image.new("RGB", size, fill)
    image = image.convert("RGB")
    image.thumbnail(size)
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def build_few_shot_montage(
    image_path: Path,
    support_examples: Sequence[ImageSample],
    cell_size: Tuple[int, int] = (224, 224),
    label_height: int = 34,
):
    from math import ceil, sqrt
    from PIL import Image, ImageDraw

    panels = list(support_examples) + [ImageSample(path=image_path, label="QUERY")]
    cols = max(3, ceil(sqrt(len(panels))))
    rows = ceil(len(panels) / cols)
    montage = Image.new("RGB", (cols * cell_size[0], rows * (cell_size[1] + label_height)), "white")
    draw = ImageDraw.Draw(montage)

    for index, panel in enumerate(panels):
        row = index // cols
        col = index % cols
        x = col * cell_size[0]
        y = row * (cell_size[1] + label_height)
        label = panel.label
        image = _fit_image_on_canvas(open_rgb_image(panel.path), cell_size)
        montage.paste(image, (x, y))
        draw.rectangle([x, y + cell_size[1], x + cell_size[0], y + cell_size[1] + label_height], fill="white")
        draw.rectangle([x, y, x + cell_size[0] - 1, y + cell_size[1] + label_height - 1], outline="black")
        draw.text((x + 6, y + cell_size[1] + 8), label, fill="black")

    return montage


def build_visual_input(image_path: Path, support_examples: Sequence[ImageSample], few_shot_format: str):
    if support_examples and few_shot_format == "montage":
        return build_few_shot_montage(image_path, support_examples)
    return open_rgb_image(image_path)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_predictions(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "true_label", "pred_label", "raw_answer", "correct"],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(path: Path, y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str], title: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig_width = max(9, len(labels) * 1.1)
    fig_height = max(7, len(labels) * 0.9)
    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def append_result(csv_path: Path, row: dict) -> None:
    import pandas as pd

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(csv_path, index=False)


def parse_common_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--support-split", default="train")
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt-style", choices=["standard", "strict", "choice", "medical_choice"], default="standard")
    parser.add_argument("--few-shot-format", choices=["inline", "conversation", "montage"], default="inline")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def torch_dtype_from_name(name: str):
    import torch

    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def resolve_runtime_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def evaluate_predictions(
    *,
    model_key: str,
    model_id: str,
    args: argparse.Namespace,
    predict_fn: Callable[[Path, Sequence[str], Sequence[ImageSample]], str],
    supports_support_images: bool,
) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
    from tqdm import tqdm

    class_names, samples = collect_samples(args.data_dir, args.split)
    samples = limit_samples_per_class(samples, args.max_samples_per_class, args.seed)
    support_examples = select_support_examples(args.data_dir, args.support_split, class_names, args.shots, args.seed)

    if args.dry_run:
        print(f"Model: {model_key} ({model_id})")
        print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
        print(f"Evaluation samples: {len(samples)}")
        print(f"Support examples: {len(support_examples)}")
        print(f"Image few-shot support: {supports_support_images}")
        return {}

    uses_montage = args.shots > 0 and args.few_shot_format == "montage"
    used_support = support_examples if supports_support_images or uses_montage else []
    if args.shots > 0 and support_examples and not supports_support_images and not uses_montage:
        print(
            f"[WARN] {model_key} does not support separate image examples; "
            "running without visual support. Use --few-shot-format montage for composite-image few-shot."
        )

    run_tag = f"{args.shots}shot_{args.split}_seed{args.seed}"
    if args.prompt_style != "standard":
        run_tag += f"_{args.prompt_style}"
    if args.shots > 0 and args.few_shot_format != "inline":
        run_tag += f"_{args.few_shot_format}"
    if args.max_samples_per_class:
        run_tag += f"_max{args.max_samples_per_class}"

    model_output_dir = args.output_root / model_key
    reports_dir = model_output_dir / "reports"
    plots_dir = model_output_dir / "plots"
    predictions_dir = model_output_dir / "predictions"

    rows = []
    y_true = []
    y_pred = []
    for sample in tqdm(samples, desc=f"{model_key} {run_tag}"):
        raw_answer = predict_fn(sample.path, class_names, used_support)
        pred_label = normalize_answer(raw_answer, class_names)
        y_true.append(sample.label)
        y_pred.append(pred_label)
        rows.append(
            {
                "image_path": display_path(sample.path),
                "true_label": sample.label,
                "pred_label": pred_label,
                "raw_answer": raw_answer,
                "correct": pred_label == sample.label,
            }
        )

    prediction_path = predictions_dir / f"predictions_{model_key}_{run_tag}.csv"
    write_predictions(prediction_path, rows)

    report_labels = list(class_names)
    if any(label not in class_names for label in y_pred):
        report_labels.append("unknown")

    report = classification_report(y_true, y_pred, labels=report_labels, zero_division=0, digits=4)
    report_path = reports_dir / f"classification_report_{model_key}_{run_tag}.txt"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    cm_path = plots_dir / f"confusion_matrix_{model_key}_{run_tag}.png"
    save_confusion_matrix(cm_path, y_true, y_pred, report_labels, f"{model_key} - {run_tag}")

    result_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_key": model_key,
        "model_id": model_id,
        "split": args.split,
        "shots": args.shots,
        "seed": args.seed,
        "support_split": args.support_split,
        "num_test_samples": len(samples),
        "num_support_examples": len(used_support),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=class_names, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0),
        "predictions_csv": display_path(prediction_path),
        "classification_report": display_path(report_path),
        "confusion_matrix": display_path(cm_path),
    }

    append_result(args.output_root / "vlm_results.csv", result_row)
    append_result(model_output_dir / f"vlm_results_{model_key}.csv", result_row)
    print(
        f"[DONE] {model_key}: accuracy={result_row['accuracy']:.4f}, "
        f"macro_f1={result_row['macro_f1']:.4f}, weighted_f1={result_row['weighted_f1']:.4f}"
    )
    return result_row
