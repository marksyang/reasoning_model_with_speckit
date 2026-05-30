"""Unsloth + QLoRA fine-tuning with streaming logs (MLX backend on macOS)."""

import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Generator

from unsloth import FastLanguageModel
from datasets import DatasetDict

from .dataset_manager import format_for_finetuning

logger = logging.getLogger(__name__)


# ---- Training configuration ----

@dataclass
class TrainingConfig:
    """Configuration for fine-tuning."""
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    num_train_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    max_seq_length: int = 2048
    logging_steps: int = 10


class _ProgressCallback:
    """Bridge MLX TrainingCallback to a queue for progress streaming."""

    def __init__(self, q: queue.Queue):
        self.q = q

    def on_train_loss_report(self, train_info: dict):
        try:
            self.q.put({
                "type": "progress",
                "iteration": train_info.get("iteration", 0),
                "train_loss": train_info.get("train_loss", 0),
                "learning_rate": train_info.get("learning_rate", 0),
                "tokens_per_second": train_info.get("tokens_per_second", 0),
            }, block=False)
        except queue.Full:
            pass

    def on_val_loss_report(self, val_info: dict):
        try:
            self.q.put({
                "type": "val",
                "val_loss": val_info.get("val_loss", 0),
            }, block=False)
        except queue.Full:
            pass


def validate_training_inputs(base_model_path: str, dataset_path: str) -> bool:
    """Validate that both model and dataset paths exist."""
    if not Path(base_model_path).exists():
        logger.error(f"Model path does not exist: {base_model_path}")
        return False
    if not Path(dataset_path).exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return False
    return True


def train(
    base_model_id: str,
    dataset,
    adapter_id: str,
    config: TrainingConfig,
    progress_callback=None,
    logging_callback=None,
) -> Generator[dict, None, None]:
    """
    Full fine-tuning with Unsloth + QLoRA (MLX backend).

    Yields progress updates: {epoch, loss, perplexity, lr, message}
    On completion: {type: "completed", adapter_path: str}
    On failure: {type: "failed", error: str}
    """
    adapter_dir = Path(f"data/adapters/{adapter_id}")
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_id,
            max_seq_length=config.max_seq_length,
            load_in_4bit=True,
            fast_inference=False,
        )
    except Exception as e:
        error_msg = f"Failed to load model: {e}"
        logger.error(error_msg)
        yield {"type": "failed", "error": error_msg}
        return

    # Prepare dataset — pick train split from DatasetDict
    if isinstance(dataset, DatasetDict):
        train_split = dataset.get("train", dataset[list(dataset.keys())[0]])
    else:
        train_split = dataset

    # Format as alpaca-style prompts
    try:
        formatted_texts = format_for_finetuning(train_split)
    except Exception as e:
        import traceback
        error_msg = f"Failed to format dataset: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        yield {"type": "failed", "error": error_msg}
        return

    # Create MLX dataset
    from mlx_lm.tuner.datasets import TextDataset
    train_set = TextDataset([{"text": t} for t in formatted_texts], tokenizer, text_key="text")

    # Dummy empty validation set
    val_set = TextDataset([], tokenizer, text_key="text")

    # Progress queue
    q: queue.Queue = queue.Queue(maxsize=100)
    callback = _ProgressCallback(q)

    # Build args namespace for mlx_lm.lora.train_model
    from argparse import Namespace
    num_layers = len(model.layers)
    args = Namespace(
        seed=42,
        num_layers=num_layers,
        fine_tune_type="lora",
        lora_parameters={"rank": config.lora_r, "scale": config.lora_alpha, "dropout": config.lora_dropout},
        resume_adapter_file=None,
        adapter_path=str(adapter_dir),
        batch_size=config.batch_size,
        iters=max(len(formatted_texts) // config.batch_size * config.num_train_epochs, 1),
        val_batches=0,
        steps_per_report=max(config.logging_steps, 1),
        steps_per_eval=0,
        save_every=max(len(formatted_texts) // config.batch_size, 1),
        max_seq_length=config.max_seq_length,
        grad_checkpoint=False,
        grad_accumulation_steps=config.gradient_accumulation_steps,
        lr_schedule=None,
        learning_rate=config.learning_rate,
        optimizer="adamw",
        optimizer_config={"adamw": {}},
    )

    yield {
        "epoch": 0,
        "loss": 0.0,
        "perplexity": 1.0,
        "lr": config.learning_rate,
        "message": "Starting training...",
    }

    # Run training — poll queue until done
    def _run():
        try:
            from mlx_lm.lora import train_model
            train_model(args, model, train_set, val_set, callback)
            q.put({"type": "done", "adapter_path": str(adapter_dir)})
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            q.put({"type": "error", "error": f"{e}\n{tb}"})

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Yield progress updates
    while True:
        try:
            msg = q.get(timeout=2.0)
        except queue.Empty:
            continue

        if msg["type"] == "done":
            yield {
                "type": "completed",
                "adapter_path": msg["adapter_path"],
                "message": "Training complete!",
            }
            return
        elif msg["type"] == "error":
            error_msg = f"Training failed: {msg['error']}"
            logger.error(error_msg)
            yield {"type": "failed", "error": error_msg}
            return
        elif msg["type"] == "progress":
            from math import exp
            loss = msg["train_loss"]
            ppl = round(float(exp(loss)), 2) if loss > 0 else 1.0
            yield {
                "epoch": 0,
                "loss": loss,
                "perplexity": ppl,
                "lr": msg["learning_rate"],
                "message": (
                    f"Iter {msg['iteration']}: loss={loss:.4f}, "
                    f"ppl={ppl:.2f}, lr={msg['learning_rate']:.2e}, "
                    f"tok/s={msg['tokens_per_second']:.0f}"
                ),
            }


def register_adapter(adapter_id: str, base_model_hf_id: str) -> dict:
    """Register a trained adapter in app state."""
    adapter_dir = Path(f"data/adapters/{adapter_id}")

    model_metadata = {
        "adapter_id": adapter_id,
        "base_model_hf_id": base_model_hf_id,
        "adapter_path": str(adapter_dir),
        "created_at": "2026-05-30",
    }

    from .model_manager import load_app_state, save_app_state
    state = load_app_state()

    state["registered_adapters"].append(adapter_id)
    state["available_models"].append(base_model_hf_id)
    if "training_runs" not in state:
        state["training_runs"] = []
    state["training_runs"].append({
        "adapter_id": adapter_id,
        "base_model": base_model_hf_id,
        "adapter_path": str(adapter_dir),
    })

    save_app_state(state)

    return model_metadata


def load_adapter(base_model_path: str, adapter_path: str) -> object:
    """Load a fine-tuned model from base model + adapter."""
    import mlx.core as mx
    from mlx_lm import load

    model, tokenizer = load(base_model_path)
    adapter_weights = mx.load(
        str(Path(adapter_path) / "adapters.safetensors"),
        format="safetensors",
    )
    model.update(adapter_weights)
    return model, tokenizer
