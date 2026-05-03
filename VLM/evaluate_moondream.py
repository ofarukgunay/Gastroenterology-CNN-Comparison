from pathlib import Path
from typing import Sequence

from common import (
    ImageSample,
    build_classification_prompt,
    evaluate_predictions,
    open_rgb_image,
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
        image = open_rgb_image(image_path)
        prompt = build_classification_prompt(class_names, False)
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
