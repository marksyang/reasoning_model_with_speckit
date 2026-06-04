"""Run inference with base or fine-tuned models (MLX backend)."""

import logging
from typing import Generator

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

logger = logging.getLogger(__name__)


def base_inference(
    model_path: str,
    prompt: str,
    generation_kwargs: dict,
) -> Generator[str, None, None]:
    """
    Run inference on a base model with streaming output.

    Uses mlx_lm for MLX-accelerated inference on Apple Silicon.
    """
    logger.info(f"Loading base model from {model_path}...")

    model, tokenizer = load(model_path)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    max_tokens = int(generation_kwargs.get("max_new_tokens", 512))
    temp = float(generation_kwargs.get("temperature", 0.7))
    top_p = float(generation_kwargs.get("top_p", 0.9))
    sampler = make_sampler(temp=temp, top_p=top_p)

    output = generate(
        model,
        tokenizer,
        prompt=text,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    )

    yield output


def finetuned_inference(
    base_model_path: str,
    adapter_path: str,
    prompt: str,
    generation_kwargs: dict,
) -> Generator[str, None, None]:
    """Run inference with a fine-tuned model (base model + LoRA adapter).

    Uses ``mlx_lm.load(adapter_path=...)`` to load the adapter weights
    produced by mlx_lm training.
    """
    logger.info(
        f"Loading fine-tuned model: base={base_model_path}, adapter={adapter_path}"
    )

    # mlx_lm.load supports adapter_path natively
    model, tokenizer = load(base_model_path, adapter_path=adapter_path)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    max_tokens = int(generation_kwargs.get("max_new_tokens", 512))
    temp = float(generation_kwargs.get("temperature", 0.7))
    top_p = float(generation_kwargs.get("top_p", 0.9))
    sampler = make_sampler(temp=temp, top_p=top_p)

    output = generate(
        model,
        tokenizer,
        prompt=text,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    )

    yield output
