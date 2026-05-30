"""Manage Hugging Face model downloads, caching, and inference."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, list_models, hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Path to app state file
APP_STATE_PATH = Path("data/app_state.json")

# Maximum parameter count allowed (3B parameters = reasonable for 12GB with 4-bit)
MAX_PARAMETERS = 3_000_000_000

# Pre-approved models list
CURATED_MODELS = {
    "meta-llama/Llama-3.2-1B-Instruct": {
        "name": "Meta LLaMA 3.2 1B Instruct",
        "param_count": 1_000_000_000,
        "max_tokens": 512,
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "name": "Meta LLaMA 3.1 8B Instruct",
        "param_count": 8_000_000_000,
        "max_tokens": 256,
        "note": "7B+ models may need gradient_checkpointing and smaller batch sizes",
    },
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "name": "Qwen 2.5 0.5B Instruct",
        "param_count": 500_000_000,
        "max_tokens": 512,
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "name": "Qwen 2.5 1.5B Instruct",
        "param_count": 1_500_000_000,
        "max_tokens": 512,
    },
}


@dataclass
class ModelInfo:
    """Information about a model."""
    hf_id: str
    name: str
    param_count: int
    max_tokens: int = 512
    note: str = ""
    is_cached: bool = False
    status: str = "pending"  # pending, downloading, cached, failed


def load_app_state() -> dict:
    """Load app state from disk."""
    if APP_STATE_PATH.exists():
        with open(APP_STATE_PATH) as f:
            return json.load(f)
    return {
        "available_models": [],
        "available_datasets": [],
        "registered_adapters": [],
        "selected_model": None,
        "selected_dataset": None,
        "training_runs": [],
    }


def save_app_state(state: dict) -> None:
    """Save app state to disk."""
    with open(APP_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def get_model_cache_path() -> Path:
    """Get the default Hugging Face cache directory."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def list_models_by_param(max_params: int) -> list[ModelInfo]:
    """List models from Hugging Face filtered by parameter count."""
    api = HfApi()
    try:
        models = api.list_models(
            search="model_family:transformers",
            sort="downloads",
            limit=200,
        )
    except Exception:
        return []

    result = []
    seen = set()

    for model in models:
        if model.id in seen or model.id in CURATED_MODELS:
            continue
        seen.add(model.id)

        # Skip if we can't determine param count
        if not model.card_data or not model.card_data.get("model_index"):
            continue

        if "Parameters" not in str(model.card_data.get("model_index", {})):
            continue

        try:
            param_count = int(model.card_data["model_index"]["Parameters"])
        except (TypeError, ValueError):
            continue

        if param_count <= max_params:
            result.append(ModelInfo(
                hf_id=model.id,
                name=model.id.split("/")[-1],
                param_count=param_count,
                is_cached=False,
            ))

        if len(result) >= 20:
            break

    return result


def get_cached_models() -> list[dict]:
    """Get list of models cached locally."""
    cache_path = get_model_cache_path()
    cached = []

    if not cache_path.exists():
        return cached

    for entry in cache_path.iterdir():
        if entry.is_dir() and entry.name.startswith("models--"):
            # Parse model ID from HF cache directory name
            cached.append({
                "hf_id": entry.name.replace("models--", "").replace("--", "/"),
                "name": entry.name.replace("models--", "").replace("--", "/").split("/")[-1],
                "is_cached": True,
                "status": "cached",
            })

    # Also load from app_state
    state = load_app_state()
    for model_id in state.get("available_models", []):
        if model_id and model_id not in [c["hf_id"] for c in cached]:
            cached.append({
                "hf_id": model_id,
                "is_cached": True,
                "status": "cached",
            })

    return cached


def list_models() -> list[dict]:
    """Get list of available models, including cached ones."""
    available = []
    cached = get_cached_models()
    cached_ids = {c["hf_id"] for c in cached}

    # Add curated models
    for hf_id, info in CURATED_MODELS.items():
        available.append({
            "hf_id": hf_id,
            "name": info["name"],
            "param_count": info["param_count"],
            "is_cached": hf_id in cached_ids,
            "status": "cached" if hf_id in cached_ids else "pending",
        })

    # Add local HF list (filtered by params)
    try:
        local_models = list_models_by_param(MAX_PARAMETERS)
    except Exception:
        local_models = []
    for model in local_models:
        if model.hf_id not in cached_ids:
            available.append({
                "hf_id": model.hf_id,
                "name": model.name,
                "param_count": model.param_count,
                "is_cached": False,
                "status": "pending",
            })

    return available


def download_model(hf_id: str) -> str:
    """
    Download a model from Hugging Face Hub.

    Returns the local cache path on success.
    Raises exception on failure.
    """
    state = load_app_state()

    if hf_id not in CURATED_MODELS:
        raise ValueError(f"Model {hf_id} is not in curated list. Please select from the dropdown.")

    logger.info(f"Downloading model {hf_id}...")

    try:
        # Download the model (Transformers Auto classes handle caching)
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        config = AutoConfig.from_pretrained(hf_id)
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModelForCausalLM.from_pretrained(hf_id)

        # Get cache path from the config's cache_dir if available
        cache_path = None
        if hasattr(config, "pretrained_model_name_or_path"):
            cache_path = str(config.pretrained_model_name_or_path)
        else:
            # HF caches in ~/.cache/huggingface/hub/models--{hf_id}
            cache_path = str(get_model_cache_path() / f"models--{hf_id.replace('/', '--')}")

        # Update app state
        if hf_id not in state["available_models"]:
            state["available_models"].append(hf_id)
            save_app_state(state)

        logger.info(f"Model {hf_id} downloaded to {cache_path}")
        return cache_path

    except Exception as e:
        logger.error(f"Failed to download model {hf_id}: {e}")
        raise


def register_model(hf_id: str) -> str:
    """Register a model in app state."""
    state = load_app_state()
    if hf_id not in state["available_models"]:
        state["available_models"].append(hf_id)
        save_app_state(state)
    return hf_id


def get_model_info(hf_id: str) -> dict:
    """Get metadata for a specific model."""
    if hf_id in CURATED_MODELS:
        info = CURATED_MODELS[hf_id].copy()
        info["hf_id"] = hf_id
        info["is_cached"] = any(
            cached["hf_id"] == hf_id for cached in get_cached_models()
        )
        return info

    raise ValueError(f"Model {hf_id} is not in curated list.")
