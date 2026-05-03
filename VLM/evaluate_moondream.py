from pathlib import Path
from typing import Sequence

from common import (
    ImageSample,
    build_classification_prompt,
    build_visual_input,
    evaluate_predictions,
    parse_common_args,
    resolve_runtime_device,
)


MODEL_KEY = "moondream"
MODEL_ID = "vikhyatk/moondream2"
REVISION = "2025-06-21"


def main() -> None:
    args = parse_common_args("Evaluate Moondream on Kvasir.")

    if args.dry_run:
        evaluate_predictions(
            model_key=MODEL_KEY,
            model_id=MODEL_ID,
            args=args,
            predict_fn=lambda *_: "",
            supports_support_images=False,
        )
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    runtime_device = resolve_runtime_device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    ).to(runtime_device)
    AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, local_files_only=args.local_files_only)

    def predict(image_path: Path, class_names: Sequence[str], support_examples: Sequence[ImageSample]) -> str:
        image = build_visual_input(image_path, support_examples, args.few_shot_format)
        prompt = build_classification_prompt(class_names, bool(support_examples), args.prompt_style)
        return model.query(image, prompt)["answer"]

    evaluate_predictions(
        model_key=MODEL_KEY,
        model_id=MODEL_ID,
        args=args,
        predict_fn=predict,
        supports_support_images=False,
    )


if __name__ == "__main__":
    main()
