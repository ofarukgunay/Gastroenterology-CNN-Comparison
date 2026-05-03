# VLM Evaluation

This folder adds few-shot evaluation for the 6 selected vision-language models.
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

## Few-Shot Evaluation

Use `--shots K` to select K support images per class from `train`.

```bash
python VLM/evaluate_qwen2_vl.py --shots 5 --device auto
```

Important: image-based in-context few-shot is currently used for adapters that support multiple images cleanly:

- `qwen2_vl`
- `qwen25_vl`

The other adapters receive only one image through their current APIs. For those models, use `--few-shot-format montage` so support examples and the query image are presented in one labeled composite image.

Removed/problematic models from the first trial:

- `microsoft/Florence-2-base-ft`: runs technically, but returns `unanswerable` for this class-list classification prompt.
- `microsoft/Phi-3-vision-128k-instruct`: its model card recommends older `transformers==4.40.2`; that version is not practical in the current Python 3.13 environment because older `tokenizers` needs local compilation.
- `openbmb/MiniCPM-V-2`: model card targets older Transformers behavior and fails in the current environment with generation/cache errors. The newer `MiniCPM-V-2_6` is gated and requires Hugging Face access approval.

## Suggested Run Order

Start with `--max-samples-per-class 1` for each few-shot condition, then remove that flag for the real test.

Current first image-based 1-shot results on the full test split (`1208` images, `seed=42`, `max_new_tokens=16`):

| Model | Prompt | Shots | Support images | Accuracy | Macro F1 | Note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `qwen25_vl` | `choice` | 1 | 8 | 0.2541 | 0.1802 | Numbered-option prompt; best VLM accuracy so far. |
| `qwen2_vl` | `standard` | 1 | 8 | 0.1250 | 0.0278 | One support image per class from `train`. |
| `qwen25_vl` | `standard` | 1 | 8 | 0.1995 | 0.1258 | One support image per class from `train`. |

For the reportable VLM comparison, use only full-test few-shot rows. Zero-shot outputs were removed because this project phase focuses on few-shot prompting.

## Prompt Variants

The default few-shot format is the original inline prompt, so the first experiment outputs remain reproducible.
For the next controlled trials, use the prompt tags below. They are appended to output filenames and result rows.

```bash
python VLM/evaluate_qwen2_vl.py --shots 1 --prompt-style choice --device auto
python VLM/evaluate_qwen25_vl.py --shots 1 --prompt-style choice --device auto
```

Use `medical_choice` to apply the prompt engineering rules used in the second phase:

- fixed output format
- numbered options
- exact class-label constraint
- short medical descriptions for technical labels
- reminder that small mucosal texture/color/anatomical details matter

This prompt style can be used by all 6 VLM adapters. For `qwen2_vl` and `qwen25_vl`, combine it with `--shots 1` for native image-based few-shot. For the other adapters, combine `--shots 1` with `--few-shot-format montage` so support examples and the query are shown in one composite image.

```bash
python VLM/evaluate_moondream.py --shots 1 --few-shot-format montage --prompt-style choice --device auto
python VLM/evaluate_qwen2_vl.py --shots 1 --prompt-style medical_choice --device auto
python VLM/evaluate_qwen25_vl.py --shots 1 --prompt-style medical_choice --device auto
python VLM/evaluate_smolvlm_500m.py --shots 1 --few-shot-format montage --prompt-style choice --device auto
python VLM/evaluate_blip2_opt.py --shots 1 --few-shot-format montage --prompt-style choice --device auto
python VLM/evaluate_llava_phi3.py --shots 1 --few-shot-format montage --prompt-style choice --device auto
```

For Qwen models, labeled support images can also be provided as separate chat turns:

```bash
python VLM/evaluate_qwen2_vl.py --shots 1 --prompt-style choice --few-shot-format conversation --device auto
python VLM/evaluate_qwen25_vl.py --shots 1 --prompt-style choice --few-shot-format conversation --device auto
```

Use smoke tests first by adding `--max-samples-per-class 1`. Smoke outputs contain `_max1` and should not be included in final reported results.
