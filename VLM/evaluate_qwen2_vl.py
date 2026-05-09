from pathlib import Path
from typing import Sequence

from common import (
    ImageSample,
    build_qwen_messages,
    evaluate_predictions,
    parse_common_args,
    torch_dtype_from_name,
)


MODEL_KEY = "qwen2_vl"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


def main() -> None:
    args = parse_common_args("Evaluate Qwen2-VL on Kvasir.")

    if args.dry_run:
        evaluate_predictions(
            model_key=MODEL_KEY,
            model_id=MODEL_ID,
            args=args,
            predict_fn=lambda *_: "",
            supports_support_images=True,
        )
        return

    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype_from_name(args.torch_dtype),
        device_map=args.device,
        local_files_only=args.local_files_only,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=args.local_files_only)

    def predict(image_path: Path, class_names: Sequence[str], support_examples: Sequence[ImageSample]) -> str:
        messages = build_qwen_messages(
            image_path=image_path,
            class_names=class_names,
            support_examples=support_examples,
            few_shot_format=args.few_shot_format,
            prompt_style=args.prompt_style,
        )
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    evaluate_predictions(
        model_key=MODEL_KEY,
        model_id=MODEL_ID,
        args=args,
        predict_fn=predict,
        supports_support_images=True,
    )


if __name__ == "__main__":
    main()
