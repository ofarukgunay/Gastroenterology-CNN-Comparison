import argparse
import csv
import json
import random
import time
from pathlib import Path

from build_fewshot_examples import build_fewshot_examples
from build_test_dataframe import build_test_dataframe
from vlm_inference import load_vlm_model, run_single_prediction_hf, unload_vlm_model
from vlm_fewshot_config import (
    DTYPE,
    IMAGE_SIZE,
    LOGS_DIR,
    MAX_NEW_TOKENS,
    PREDICTIONS_DIR,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
    VLM_MODELS,
    ensure_output_dirs,
    get_model_map,
)


def get_class_names(train_dir: Path) -> list[str]:
    return sorted([p.name for p in train_dir.iterdir() if p.is_dir()])


def build_prompt(class_names: list[str], shots: int, language: str) -> str:
    classes_line = ", ".join(class_names)
    if language == "tr":
        return (
            "Sen bir görsel sınıflandırma asistanısın.\n\n"
            "Görevin, verilen görseli aşağıdaki sınıflardan tam olarak birine atamaktır:\n"
            f"{classes_line}\n\n"
            f"Few-shot örnek sayısı: {shots}\n"
            "Sadece sınıf adını yaz.\n"
            "Açıklama yapma."
        )
    return (
        "You are a visual classification assistant.\n\n"
        "Your task is to classify the given image into exactly one of the following classes:\n"
        f"{classes_line}\n\n"
        f"Few-shot shot count: {shots}\n"
        "Return only the class name.\n"
        "Do not explain."
    )


def add_textual_fewshot_block(prompt: str, fewshot_examples: list[dict], limit: int = 24) -> str:
    if not fewshot_examples:
        return prompt

    lines = ["", "Few-shot textual references (image_file -> class):"]
    for idx, ex in enumerate(fewshot_examples[:limit], start=1):
        image_name = Path(ex["image_path"]).name
        lines.append(f"{idx}. {image_name} -> {ex['label']}")
    return prompt + "\n" + "\n".join(lines)


def normalize_prediction(raw_text: str, class_names: list[str]) -> str:
    text = (raw_text or "").strip().lower()
    for cls in class_names:
        if cls.lower() == text:
            return cls
    for cls in class_names:
        if cls.lower() in text:
            return cls
    return "UNKNOWN"


def sample_test_rows(test_rows: list[dict], max_samples_per_class: int | None, seed: int) -> list[dict]:
    if max_samples_per_class is None:
        return test_rows

    by_class: dict[str, list[dict]] = {}
    for row in test_rows:
        by_class.setdefault(row["label"], []).append(row)

    rng = random.Random(seed)
    sampled: list[dict] = []
    for label in sorted(by_class):
        rows = by_class[label]
        if len(rows) <= max_samples_per_class:
            sampled.extend(rows)
        else:
            sampled.extend(rng.sample(rows, max_samples_per_class))
    return sampled


def run_single_prediction_dummy(image_path: str, prompt: str) -> str:
    _ = image_path, prompt
    return "UNKNOWN"


def write_predictions_csv(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model",
        "shots",
        "image_path",
        "true_label",
        "raw_output",
        "pred_label",
        "is_correct",
        "inference_time_sec",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def log_error(model_name: str, message: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"{model_name}_{ts}.log"
    try:
        with log_path.open("w", encoding="utf-8") as f:
            f.write(message)
        print(f"[WARN] Logged error to: {log_path}")
    except Exception:
        print("[WARN] Could not write log file. Error message:")
        print(message.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run few-shot VLM inference.")
    parser.add_argument("--model", type=str, default="smolvlm_500m", help="'all' or model name from config.")
    parser.add_argument("--shots", type=int, default=1, help="Shots per class (0/1/3/5 recommended).")
    parser.add_argument("--language", type=str, default="en", choices=["en", "tr"], help="Prompt language.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--train_dir", type=Path, default=TRAIN_DIR, help="Train split dir.")
    parser.add_argument("--test_dir", type=Path, default=TEST_DIR, help="Test split dir.")
    parser.add_argument(
        "--max_samples_per_class",
        type=int,
        default=None,
        help="Optional test subset size per class.",
    )
    parser.add_argument("--print_only", action="store_true", help="Do not write CSV outputs.")
    parser.add_argument(
        "--backend",
        type=str,
        default="dummy",
        choices=["dummy", "hf"],
        help="Inference backend.",
    )
    parser.add_argument("--dtype", type=str, default=DTYPE, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--disable_4bit",
        action="store_true",
        help="Force disable 4-bit quantization for all models in this run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    if not args.train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {args.train_dir}")
    if not args.test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {args.test_dir}")
    if args.shots < 0:
        raise ValueError("--shots must be >= 0")

    class_names = get_class_names(args.train_dir)
    if not class_names:
        raise RuntimeError(f"No class directories found in {args.train_dir}")

    fewshot_examples, per_class_counts = build_fewshot_examples(
        train_dir=args.train_dir,
        shots_per_class=args.shots,
        seed=args.seed,
    )
    prompt = build_prompt(class_names=class_names, shots=args.shots, language=args.language)
    prompt = add_textual_fewshot_block(prompt=prompt, fewshot_examples=fewshot_examples)

    test_df = build_test_dataframe(args.test_dir)
    test_rows = test_df.to_dict(orient="records")
    test_rows = sample_test_rows(
        test_rows=test_rows,
        max_samples_per_class=args.max_samples_per_class,
        seed=args.seed,
    )

    model_map = get_model_map()
    if args.model == "all":
        model_names = [m["name"] for m in VLM_MODELS]
    else:
        if args.model not in model_map:
            raise ValueError(f"Unknown model: {args.model}")
        model_names = [args.model]

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Few-shot per class: {json.dumps(per_class_counts, ensure_ascii=False)}")
    print(f"Few-shot examples total: {len(fewshot_examples)}")
    print(f"Test rows to run: {len(test_rows)}")

    for model_name in model_names:
        print(f"\n[MODEL] {model_name} | backend={args.backend} | shots={args.shots}")
        predictions: list[dict] = []
        bundle = None
        try:
            if args.backend == "hf":
                model_cfg = dict(model_map[model_name])
                if args.disable_4bit:
                    model_cfg["load_in_4bit"] = False
                bundle = load_vlm_model(
                    model_cfg=model_cfg,
                    dtype_name=args.dtype,
                    device_preference="cuda",
                )

            for row in test_rows:
                image_path = row["image_path"]
                true_label = row["label"]

                t0 = time.perf_counter()
                if args.backend == "dummy":
                    raw_output = run_single_prediction_dummy(image_path=image_path, prompt=prompt)
                else:
                    raw_output = run_single_prediction_hf(
                        bundle=bundle,
                        image_path=image_path,
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                        image_size=args.image_size,
                    )
                inference_time_sec = round(time.perf_counter() - t0, 4)
                pred_label = normalize_prediction(raw_output, class_names)

                predictions.append(
                    {
                        "model": model_name,
                        "shots": args.shots,
                        "image_path": image_path,
                        "true_label": true_label,
                        "raw_output": raw_output,
                        "pred_label": pred_label,
                        "is_correct": int(pred_label == true_label),
                        "inference_time_sec": inference_time_sec,
                    }
                )
        except Exception as ex:  # pragma: no cover
            log_error(model_name, f"Error during inference for {model_name}\n{ex}\n")
            continue
        finally:
            unload_vlm_model(bundle)

        output_path = PREDICTIONS_DIR / f"{model_name}_shot{args.shots}.csv"
        if args.print_only:
            print(f"Predictions ready (not saved): {len(predictions)} rows")
            continue

        write_predictions_csv(output_path, predictions)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
