"""Unsloth + QLoRA fine-tuning with streaming logs."""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Generator

from unsloth import FastLanguageModel
from transformers import TrainingArguments, DataCollatorForSeq2Seq, Trainer
from trl import SFTTrainer
from datasets import Dataset

from .dataset_manager import format_for_finetuning

logger = logging.getLogger(__name__)


# ---- Training configuration ----

@dataclass
class TrainingConfig:
    """Configuration for fine-tuning."""
    # LoRA parameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"]
    )
    # Training hyperparameters
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    logging_steps: int = 1
    # Training flags
    use_gradient_checkpointing: bool = True
    use_rslora: bool = False
    fp16: bool = True
    bf16: bool = False


# ---- Core functions ----

def validate_training_inputs(base_model_path: str, dataset_path: str) -> bool:
    """Validate that both model and dataset paths exist."""
    if not Path(base_model_path).exists():
        logger.error(f"Model path does not exist: {base_model_path}")
        return False
    if not Path(dataset_path).exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return False
    return True


def load_base_model(model_id: str, config: TrainingConfig) -> FastLanguageModel:
    """Load the base model with Unsloth FastLanguageModel and 4-bit quantization."""
    logger.info(f"Loading base model: {model_id} with Unsloth...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=config.max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
    )

    return model, tokenizer


def format_dataset(dataset, tokenizer: object, config: TrainingConfig) -> Dataset:
    """Format the dataset for SFT training."""
    formatted = format_for_finetuning(dataset)
    # Convert to SFT format
    return Dataset.from_dict({"text": formatted})


def train(
    base_model_id: str,
    dataset,
    adapter_id: str,
    config: TrainingConfig,
    progress_callback=None,
    logging_callback=None,
) -> Generator[dict, None, None]:
    """
    Full fine-tuning with Unsloth + QLoRA.

    Yields progress updates: {epoch, loss, perplexity, lr, message}
    On completion: {type: "completed", adapter_path: str}
    On failure: {type: "failed", error: str}
    """
    adapter_dir = Path(f"data/adapters/{adapter_id}/lora_adapter")
    log_dir = Path(f"data/logs/{adapter_id}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Validate model exists
    try:
        model, tokenizer = load_base_model(base_model_id, config)
    except Exception as e:
        error_msg = f"Failed to load model: {e}"
        logger.error(error_msg)
        yield {"type": "failed", "error": error_msg}
        return

    # Save model locally first for adapter saving
    local_model_path = f"data/adapters/{adapter_id}/base_model"
    model.save_pretrained(local_model_path)
    tokenizer.save_pretrained(local_model_path)

    try:
        # Apply LoRA adapter
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            use_gradient_checkpointing="unsloth" if config.use_gradient_checkpointing else True,
            use_rslora=config.use_rslora,
            random_state=42,
        )

        # Format dataset
        formatted = format_for_finetuning(dataset)

        # Setup training
        training_args = TrainingArguments(
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_ratio=0.1,
            num_train_epochs=config.num_train_epochs,
            learning_rate=config.learning_rate,
            fp16=not config.bf16,
            bf16=config.bf16,
            logging_steps=config.logging_steps,
            save_strategy="epoch",
            output_dir=adapter_dir,
            report_to="none",
            max_grad_norm=1.0,
        )

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=Dataset.from_dict({"text": formatted}),
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
            args=training_args,
        )

        logger.info("Starting training...")

        # Wrap train call in generator for streaming
        yield {"epoch": 0, "loss": 0.0, "perplexity": 1.0, "lr": config.learning_rate, "message": "Starting training..."}

        for epoch in range(config.num_train_epochs):
            training_result = trainer.train()
            metrics = training_result.metrics
            metrics["epoch"] = epoch + 1

            # Calculate loss and perplexity
            train_loss = metrics.get("train_loss", 0.0)
            perplexity = round(float(exp(train_loss)), 2)

            yield {
                "epoch": epoch + 1,
                "loss": train_loss,
                "perplexity": perplexity,
                "lr": config.learning_rate * (1 - epoch / config.num_train_epochs),
                "message": f"Epoch {epoch + 1} complete",
            }

            if logging_callback:
                logging_callback(self)

        # Save the fine-tuned adapter
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        logger.info(f"Adapter saved to {adapter_dir}")

        yield {
            "type": "completed",
            "adapter_path": str(adapter_dir),
            "message": "Training complete!",
        }

    except Exception as e:
        error_msg = f"Training failed: {e}"
        logger.error(error_msg)
        yield {"type": "failed", "error": error_msg}
        raise


def register_adapter(adapter_id: str, base_model_hf_id: str) -> dict:
    """Register a trained adapter in app state."""
    adapter_dir = Path(f"data/adapters/{adapter_id}/lora_adapter")
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_dir}")

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
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_path,
            max_seq_length=2048,
            load_in_4bit=True,
        )

        model.load_adapter(adapter_path)
        return model, tokenizer
    except ImportError:
        # Fall back to standard PEFT
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        return model, tokenizer
