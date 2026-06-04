from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class LoadedVLM:
    model_name: str
    hf_id: str
    family: str
    model: Any
    processor: Any
    device: Any


def _get_torch_dtypes(dtype_name: str):
    import torch

    if dtype_name == "float16":
        return torch.float16, torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16, torch.float32
    return torch.float32, torch.float32


def _build_quant_config(load_in_4bit: bool, compute_dtype_name: str):
    if not load_in_4bit:
        return None
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except Exception:
        return None

    compute_dtype = torch.float16 if compute_dtype_name == "float16" else torch.bfloat16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_vlm_model(model_cfg: dict, dtype_name: str = "float16", device_preference: str = "cuda") -> LoadedVLM:
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, Idefics3ForConditionalGeneration

    hf_id = model_cfg["hf_id"]
    load_in_4bit = bool(model_cfg.get("load_in_4bit", False))
    preferred_dtype, cpu_dtype = _get_torch_dtypes(dtype_name)
    use_cuda = device_preference == "cuda" and torch.cuda.is_available()

    has_accelerate = False
    try:
        import accelerate  # noqa: F401

        has_accelerate = True
    except Exception:
        has_accelerate = False

    effective_4bit = load_in_4bit and use_cuda and has_accelerate
    if load_in_4bit and not effective_4bit:
        print("[WARN] 4-bit disabled (requires CUDA + accelerate). Falling back to non-quantized load.")

    quant_config = _build_quant_config(load_in_4bit=effective_4bit, compute_dtype_name=dtype_name)

    target_device = torch.device("cuda" if use_cuda else "cpu")
    load_kwargs = {
        "trust_remote_code": True,
        "dtype": preferred_dtype if use_cuda else cpu_dtype,
        "low_cpu_mem_usage": False,
    }
    if model_cfg.get("family") == "internvl":
        load_kwargs["use_flash_attn"] = False
        load_kwargs["attn_implementation"] = "eager"
    use_device_map = has_accelerate and model_cfg.get("family") != "internvl"
    if use_device_map:
        load_kwargs["device_map"] = "auto" if use_cuda else "cpu"
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config

    model = None
    last_error = None

    candidate_classes = []
    if model_cfg.get("family") == "smolvlm":
        candidate_classes.append(Idefics3ForConditionalGeneration)
    candidate_classes.extend([AutoModelForImageTextToText, AutoModelForCausalLM])

    for model_cls in candidate_classes:
        try:
            model = model_cls.from_pretrained(hf_id, **load_kwargs)
            break
        except Exception as ex:  # pragma: no cover
            last_error = ex

    if model is None:
        raise RuntimeError(f"Failed to load model: {hf_id}\nLast error: {last_error}")

    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
    if not use_device_map:
        model.to(target_device)
    model.eval()

    device = target_device
    return LoadedVLM(
        model_name=model_cfg["name"],
        hf_id=hf_id,
        family=model_cfg["family"],
        model=model,
        processor=processor,
        device=device,
    )


def _build_chat_prompt(processor: Any, prompt: str) -> str:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(conversation, add_generation_prompt=True)
    return prompt


def _prepare_inputs(bundle: LoadedVLM, image: Image.Image, prompt: str):
    processor = bundle.processor
    text_prompt = _build_chat_prompt(processor, prompt)

    try:
        inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    except Exception:
        # Fallback for processors that prefer plain prompt without chat template.
        inputs = processor(images=image, text=prompt, return_tensors="pt")

    # Move tensors to model device.
    for key, value in list(inputs.items()):
        if hasattr(value, "to"):
            inputs[key] = value.to(bundle.model.device)
    return inputs


def run_single_prediction_hf(
    bundle: LoadedVLM,
    image_path: str | Path,
    prompt: str,
    max_new_tokens: int = 32,
    image_size: int = 224,
) -> str:
    import torch

    path = Path(image_path)
    image = Image.open(path).convert("RGB")
    if image_size > 0:
        image = image.resize((image_size, image_size))

    inputs = _prepare_inputs(bundle=bundle, image=image, prompt=prompt)

    with torch.no_grad():
        output_ids = bundle.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    text = bundle.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return text.strip()


def unload_vlm_model(bundle: LoadedVLM | None) -> None:
    if bundle is None:
        return

    try:
        del bundle.model
        del bundle.processor
    except Exception:
        pass

    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
