"""Run inference with base or fine-tuned models."""

import logging
from typing import Generator

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def base_inference(
    model_path: str,
    prompt: str,
    generation_kwargs: dict,
) -> Generator[str, None, None]:
    """
    Run inference on a base model with streaming output.

    Uses 4-bit quantization (nf4 + double quantization) to fit models
    within 12GB VRAM constraints.
    """
    logger.info(f"Loading base model from {model_path}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Apply LoRA adapter if available
    adapter_path = f"{model_path}/lora_adapter"
    try:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        logger.info(f"Merged LoRA adapter from {adapter_path}")
    except (ValueError, FileNotFoundError):
        logger.info("No adapter found, using base model only.")

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    generation_kwargs.setdefault("max_new_tokens", 512)
    generation_kwargs.setdefault("temperature", 0.7)
    generation_kwargs.setdefault("top_p", 0.9)
    generation_kwargs.setdefault("do_sample", True)

    generated = model.generate(
        inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        pad_token_id=tokenizer.eos_token_id,
        **generation_kwargs,
    )

    generated_tokens = generated[0][inputs["input_ids"].shape[1]:]
    output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    yield output


def finetuned_inference(
    base_model_path: str,
    adapter_path: str,
    prompt: str,
    generation_kwargs: dict,
) -> Generator[str, None, None]:
    """Run inference with a fine-tuned model (base model + LoRA adapter)."""
    logger.info(f"Loading fine-tuned model: base={base_model_path}, adapter={adapter_path}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    generation_kwargs.setdefault("max_new_tokens", 512)
    generation_kwargs.setdefault("temperature", 0.7)
    generation_kwargs.setdefault("top_p", 0.9)
    generation_kwargs.setdefault("do_sample", True)

    generated = model.generate(
        inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        pad_token_id=tokenizer.eos_token_id,
        **generation_kwargs,
    )

    generated_tokens = generated[0][inputs["input_ids"].shape[1]:]
    output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    yield output
