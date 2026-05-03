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


MODEL_KEY = "llava_phi3"
MODEL_ID = "xtuner/llava-phi-3-mini-hf"


def main() -> None:
    args = parse_common_args("Evaluate LLaVA-Phi-3 on Kvasir.")

    if args.dry_run:
        evaluate_predictions(
            model_key=MODEL_KEY,
            model_id=MODEL_ID,
            args=args,
            predict_fn=lambda *_: "",
            supports_support_images=False,
        )
        return

    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype_from_name(args.torch_dtype),
        device_map=args.device,
        local_files_only=args.local_files_only,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=args.local_files_only)
    if getattr(processor, "patch_size", None) is None:
        processor.patch_size = getattr(model.config.vision_config, "patch_size", 14)
    processor.vision_feature_select_strategy = "full"

    def predict(image_path: Path, class_names: Sequence[str], support_examples: Sequence[ImageSample]) -> str:
        image = open_rgb_image(image_path)
        prompt_text = build_classification_prompt(class_names, False)
        if getattr(processor, "chat_template", None):
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        else:
            prompt = f"<image>\nUSER: {prompt_text}\nASSISTANT:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    evaluate_predictions(
        model_key=MODEL_KEY,
        model_id=MODEL_ID,
        args=args,
        predict_fn=predict,
        supports_support_images=False,
    )


if __name__ == "__main__":
    main()
