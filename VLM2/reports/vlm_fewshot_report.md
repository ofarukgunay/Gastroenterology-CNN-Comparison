# VLM Few-Shot Comparison Report

## Objective
This report compares 6 Vision-Language Models on the same dataset for few-shot image classification.

## Models
1. `smolvlm_500m` - `HuggingFaceTB/SmolVLM-500M-Instruct`
2. `smolvlm_2b` - `HuggingFaceTB/SmolVLM-Instruct`
3. `internvl2_5_2b` - `OpenGVLab/InternVL2_5-2B`
4. `paligemma_3b_224` - `google/paligemma-3b-mix-224`
5. `qwen2_5_vl_3b` - `Qwen/Qwen2.5-VL-3B-Instruct`
6. `llava_1_5_7b` - `llava-hf/llava-1.5-7b-hf`

## Data and Hardware
- Data root: `data/prepared-data`
- GPU target: NVIDIA RTX 4060 (8 GB assumption)
- Batch size: 1
- Quantization: model-level setting (4-bit/FP16 from config)

## Experiment Scope
- Shot values: 0, 1, 3, 5
- Prompt languages: EN (required), TR (optional)
- Metrics: Accuracy, Macro F1, Weighted F1, Average inference time

## Results
| Model | Shot | Accuracy | Macro F1 | Weighted F1 | Avg Time (s) | Samples |
| --- | --- | --- | --- | --- | --- | --- |
| smolvlm_500m | 1 | 0.1250 | 0.0278 | 0.0278 | 1.8395 | 24 |

## Best / Fastest Model
- Best model (Macro F1): `smolvlm_500m` | shot=1 | macro_f1=0.0278 | accuracy=0.1250
- Fastest model (Avg Time): `smolvlm_500m` | shot=1 | avg_time=1.8395s

## Comparison Summary
| model | shots | accuracy | macro_f1 | weighted_f1 | avg_inference_time_sec |
| --- | --- | --- | --- | --- | --- |
| smolvlm_500m | 1 | 0.125 | 0.0277777777777777 | 0.0277777777777777 | 1.839475 |

## Notes
- Track performance changes as shot count increases.
- On RTX 4060, use 4-bit and low `max_new_tokens` for heavy models.

## Conclusion
- Performance-focused choice: highest Macro F1 model
- Speed-focused choice: lowest average inference time model
- RTX 4060 compatibility: model that finishes without OOM and keeps a good speed/quality balance
