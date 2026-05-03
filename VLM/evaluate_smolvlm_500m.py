from pathlib import Path
from typing import Sequence

from common import (
    ImageSample,
    build_classification_prompt,
    evaluate_predictions,
    open_rgb_image,
    parse_common_args,
    resolve_runtime_device,
    torch_dtype_from_name,
)


MODEL_KEY = "smolvlm_500m"
MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"


def main() -> None:
    args = parse_common_args("Evaluate SmolVLM-500M on Kvasir.")

    if args.dry_run:
        evaluate_predictions(
            model_key=MODEL_KEY,
            model_id=MODEL_ID,
            args=args,
            predict_fn=lambda *_: "",
            supports_support_images=False,
        )
        return

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    runtime_device = resolve_runtime_device(args.device)
    dtype = torch_dtype_from_name(args.torch_dtype)
    if dtype == "auto":
        dtype = torch.float16 if runtime_device.startswith("cuda") else torch.float32
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        _attn_implementation="eager",
        local_files_only=args.local_files_only,
    ).to(runtime_device)
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=args.local_files_only)

    def predict(image_path: Path, class_names: Sequence[str], support_examples: Sequence[ImageSample]) -> str:
        image = open_rgb_image(image_path)
        prompt_text = build_classification_prompt(class_names, False)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(runtime_device)
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    evaluate_predictions(
        model_key=MODEL_KEY,
        model_id=MODEL_ID,
        args=args,
        predict_fn=predict,
        supports_support_images=False,
    )


if __name__ == "__main__":
    main()
