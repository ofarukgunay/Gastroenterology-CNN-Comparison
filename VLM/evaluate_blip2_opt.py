from pathlib import Path
from typing import Sequence

from common import (
    ImageSample,
    build_classification_prompt,
    evaluate_predictions,
    open_rgb_image,
    parse_common_args,
    torch_dtype_from_name,
)


MODEL_KEY = "blip2_opt"
MODEL_ID = "Salesforce/blip2-opt-2.7b"


def main() -> None:
    args = parse_common_args("Evaluate BLIP-2 OPT on Kvasir.")

    if args.dry_run:
        evaluate_predictions(
            model_key=MODEL_KEY,
            model_id=MODEL_ID,
            args=args,
            predict_fn=lambda *_: "",
            supports_support_images=False,
        )
        return

    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype_from_name(args.torch_dtype),
        device_map=args.device,
        local_files_only=args.local_files_only,
    )
    processor = Blip2Processor.from_pretrained(MODEL_ID, local_files_only=args.local_files_only)

    def predict(image_path: Path, class_names: Sequence[str], support_examples: Sequence[ImageSample]) -> str:
        image = open_rgb_image(image_path)
        prompt = build_classification_prompt(class_names, False)
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        model_dtype = next(model.parameters()).dtype
        inputs = {
            key: value.to(model.device, dtype=model_dtype) if key == "pixel_values" else value.to(model.device)
            for key, value in inputs.items()
        }
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        return processor.decode(output_ids[0], skip_special_tokens=True).strip()

    evaluate_predictions(
        model_key=MODEL_KEY,
        model_id=MODEL_ID,
        args=args,
        predict_fn=predict,
        supports_support_images=False,
    )


if __name__ == "__main__":
    main()
