# VLM Evaluation

This folder adds zero-shot and few-shot evaluation for the 6 selected vision-language models.
Each model has its own entrypoint so model-specific processor/loading problems do not block the others.

## Models

Default model ids:

| Key | Model id |
| --- | --- |
| `moondream` | `vikhyatk/moondream2` |
| `qwen2_vl` | `Qwen/Qwen2-VL-2B-Instruct` |
| `qwen25_vl` | `Qwen/Qwen2.5-VL-3B-Instruct` |
| `smolvlm_500m` | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| `blip2_opt` | `Salesforce/blip2-opt-2.7b` |
| `llava_phi3` | `xtuner/llava-phi-3-mini-hf` |

## Output Layout

The runner keeps the same idea as the CNN outputs:

```text
outputs/
  vlm/
    vlm_results.csv
    qwen2_vl/
      vlm_results_qwen2_vl.csv
      predictions/
      reports/
      plots/
```

Each model writes:

- prediction CSV
- classification report
- confusion matrix
- model-level result CSV
- global `outputs/vlm/vlm_results.csv`

## Smoke Test

Run a tiny test first, one image per class:

```bash
python VLM/evaluate_qwen2_vl.py --max-samples-per-class 1 --device auto
```

## Zero-Shot Evaluation

Run one model on the full test split:

```bash
python VLM/evaluate_qwen2_vl.py --device auto
```

Run the final models one by one:

```bash
python VLM/evaluate_moondream.py --device auto
python VLM/evaluate_qwen2_vl.py --device auto
python VLM/evaluate_qwen25_vl.py --device auto
python VLM/evaluate_smolvlm_500m.py --device auto
python VLM/evaluate_blip2_opt.py --device auto
python VLM/evaluate_llava_phi3.py --device auto
```

## Few-Shot Evaluation

Use `--shots K` to select K support images per class from `train`.

```bash
python VLM/evaluate_qwen2_vl.py --shots 5 --device auto
```

Important: image-based in-context few-shot is currently used for adapters that support multiple images cleanly:

- `qwen2_vl`
- `qwen25_vl`

The other adapters still run with the query image and class-list prompt. For those models, few-shot should be handled later with LoRA/fine-tuning if a fair few-shot training comparison is required.

Removed/problematic models from the first trial:

- `microsoft/Florence-2-base-ft`: runs technically, but returns `unanswerable` for this class-list classification prompt.
- `microsoft/Phi-3-vision-128k-instruct`: its model card recommends older `transformers==4.40.2`; that version is not practical in the current Python 3.13 environment because older `tokenizers` needs local compilation.
- `openbmb/MiniCPM-V-2`: model card targets older Transformers behavior and fails in the current environment with generation/cache errors. The newer `MiniCPM-V-2_6` is gated and requires Hugging Face access approval.

## Suggested Run Order

Start with `--max-samples-per-class 1` for each model, then remove that flag for the real test.

Recommended full zero-shot order on the current machine:

1. `qwen2_vl`
2. `moondream`
3. `qwen25_vl`
4. `blip2_opt`
5. `smolvlm_500m`
6. `llava_phi3`

Current first zero-shot results on the full test split (`1208` images, `seed=42`, `max_new_tokens=16`):

| Model | Accuracy | Macro F1 | Note |
| --- | ---: | ---: | --- |
| `qwen25_vl` | 0.2285 | 0.1888 | Best first zero-shot result. |
| `moondream` | 0.1772 | 0.0937 | Completed full test split. |
| `qwen2_vl` | 0.1614 | 0.0653 | Completed full test split. |
| `llava_phi3` | 0.1258 | 0.0309 | Completed, but very slow on this machine. |
| `blip2_opt` | 0.1250 | 0.0278 | Completed full test split. |
| `smolvlm_500m` | 0.0141 | 0.0191 | Completed full test split. |
