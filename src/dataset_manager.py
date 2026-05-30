"""Manage Hugging Face dataset downloads and preprocessing."""

import logging
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

# Pre-approved reasoning datasets
CURATED_DATASETS = {
    "openai/gsm8k": {
        "name": "GSM8K",
        "description": "Grade school math word problems",
        "column_names": ["question", "answer"],
        "config": "main",
    },
    "rajpurkar/squad": {
        "name": "SQuAD",
        "description": "Stanford Question Answering Dataset",
        "column_names": ["id", "title", "context", "question", "answers"],
    },
    "Dahoas/cot_gsm8k": {
        "name": "CoT GSM8K",
        "description": "GSM8K with chain-of-thought reasoning",
        "column_names": ["question", "answer", "prompt", "response"],
    },
    "HuggingFaceH4/no_robots": {
        "name": "No Robots",
        "description": "Human-written instruction data",
        "column_names": ["prompt", "prompt_id", "messages", "category"],
    },
    "OpenAssistant/oasst1": {
        "name": "OpenAssistant",
        "description": "Multi-turn conversation dataset",
        "column_names": ["message_id", "parent_id", "text", "role", "lang"],
    },
    "HuggingFaceH4/ultrachat_200k": {
        "name": "UltraChat 200K",
        "description": "High-quality multi-turn conversations",
        "column_names": ["prompt", "prompt_id", "messages"],
    },
}


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    hf_id: str
    name: str
    description: str = ""
    train_size: int = 0
    test_size: int = 0
    column_names: list[str] = field(default_factory=list)
    is_cached: bool = False
    status: str = "pending"


def list_datasets() -> list[dict]:
    """Get list of available datasets."""
    datasets = []

    for hf_id, info in CURATED_DATASETS.items():
        datasets.append({
            "hf_id": hf_id,
            "name": info["name"],
            "description": info["description"],
            "column_names": info["column_names"],
            "config": info.get("config"),
            "is_cached": False,
            "status": "pending",
        })

    return datasets


def download_dataset(hf_id: str, config: str | None = None) -> str:
    """Download a dataset from Hugging Face Hub."""
    logger.info(f"Downloading dataset {hf_id} (config={config})...")

    try:
        if config:
            dataset = load_dataset(hf_id, name=config, download_mode="reuse_cache_if_exists")
        else:
            dataset = load_dataset(hf_id, download_mode="reuse_cache_if_exists")
        logger.info(f"Dataset {hf_id} downloaded successfully.")
        return hf_id
    except Exception as e:
        logger.error(f"Failed to download dataset {hf_id}: {e}")
        raise


def get_dataset_summary(hf_id: str, config: str | None = None) -> dict:
    """Get summary statistics for a dataset."""
    logger.info(f"Loading dataset summary for {hf_id} (config={config})...")

    try:
        if config:
            dataset = load_dataset(hf_id, name=config, download_mode="reuse_cache_if_exists")
        else:
            dataset = load_dataset(hf_id, download_mode="reuse_cache_if_exists")
        summary = {
            "hf_id": hf_id,
            "name": CURATED_DATASETS.get(hf_id, {}).get("name", hf_id),
            "description": CURATED_DATASETS.get(hf_id, {}).get("description", ""),
        }

        for split_name, split_data in dataset.items():
            if isinstance(split_data, Dataset):
                summary[f"{split_name}_size"] = len(split_data)
                summary[f"{split_name}_columns"] = split_data.column_names

        return summary
    except Exception as e:
        logger.error(f"Failed to load dataset summary for {hf_id}: {e}")
        raise


def format_for_finetuning(dataset, source_format: str = "question/answer") -> list[str]:
    """
    Convert dataset to Alpaca-style prompts for fine-tuning.

    For GSM8K-style data (question/answer):
        Input: {"question": "...", "answer": "..."}
        Output: "### Question:\n{q}\n\n### Answer:\n{a}"

    For general text format:
        Input: {"text": "..."}
        Output: "### Instruction:\n{t}\n\n### Response:\n{text}"
    """
    prompts = []

    if source_format == "question/answer":
        for item in dataset:
            q = item.get("question", "")
            a = item.get("answer", "")
            prompt = f"### Question:\n{q}\n\n### Answer:\n{a}"
            prompts.append(prompt)
    elif source_format == "text":
        for item in dataset:
            text = item.get("text", "")
            prompt = f"### Instruction:\n{text}\n\n### Response:\n{text}"
            prompts.append(prompt)

    return prompts


def get_available_datasets() -> list[dict]:
    """Get list of datasets with cached status."""
    available = []

    for hf_id, info in CURATED_DATASETS.items():
        available.append({
            "hf_id": hf_id,
            "name": info["name"],
            "description": info["description"],
            "column_names": info["column_names"],
            "config": info.get("config"),
            "is_cached": False,
            "status": "pending",
        })

    return available
