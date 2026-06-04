from pathlib import Path

# Project roots
PROJECT_ROOT = Path(__file__).resolve().parents[2]
VLM2_ROOT = Path(__file__).resolve().parents[1]

# Data
DATA_ROOT = PROJECT_ROOT / "data" / "prepared-data"
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"

# Outputs
OUTPUT_ROOT = VLM2_ROOT / "outputs" / "vlm_fewshot"
PREDICTIONS_DIR = OUTPUT_ROOT / "predictions"
METRICS_DIR = OUTPUT_ROOT / "metrics"
LOGS_DIR = OUTPUT_ROOT / "logs"

# Runtime defaults (RTX 4060 friendly)
IMAGE_SIZE = 224
BATCH_SIZE = 1
MAX_NEW_TOKENS = 32
DTYPE = "float16"
DEVICE = "cuda"
SEED = 42

# File extensions to scan
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.tif", "*.tiff"]

# Supported VLM models
VLM_MODELS = [
    {
        "name": "smolvlm_500m",
        "hf_id": "HuggingFaceTB/SmolVLM-500M-Instruct",
        "family": "smolvlm",
        "load_in_4bit": False,
    },
    {
        "name": "smolvlm_2b",
        "hf_id": "HuggingFaceTB/SmolVLM-Instruct",
        "family": "smolvlm",
        "load_in_4bit": False,
    },
    {
        "name": "internvl2_5_2b",
        "hf_id": "OpenGVLab/InternVL2_5-2B",
        "family": "internvl",
        "load_in_4bit": True,
    },
    {
        "name": "paligemma_3b_224",
        "hf_id": "google/paligemma-3b-mix-224",
        "family": "paligemma",
        "load_in_4bit": True,
    },
    {
        "name": "qwen2_5_vl_3b",
        "hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "family": "qwen2_5_vl",
        "load_in_4bit": True,
    },
    {
        "name": "llava_1_5_7b",
        "hf_id": "llava-hf/llava-1.5-7b-hf",
        "family": "llava",
        "load_in_4bit": True,
    },
]


def ensure_output_dirs() -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def get_model_map() -> dict:
    return {m["name"]: m for m in VLM_MODELS}
