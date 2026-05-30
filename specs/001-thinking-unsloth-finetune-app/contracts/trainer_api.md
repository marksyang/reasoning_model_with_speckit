# Trainer API Contract

## Interface

The trainer is a single-file module (`src/trainer.py`) that provides these public methods:

### `validate_inputs(base_model_path: str, dataset_path: str) -> bool`
- **Precondition**: Both paths exist and point to valid models/datasets
- **Returns**: `True` if validation passed, `False` otherwise
- **Side effects**: None

### `train(base_model_path: str, dataset_path: str, adapter_id: str, lora_params: dict, progress_callback=None) -> Generator[dict, None, None]`
- **Precondition**: `validate_inputs()` returned True
- **Lora params**: `{r: int, alpha: int, dropout: float}` (defaults: 8, 16, 0.05)
- **Returns**: Generator yielding `{"epoch": int, "loss": float, "perplexity": float, "lr": float, "message": str}` per step
- **Side effects**: Creates `data/adapters/<adapter_id>/lora_adapter/` on completion
- **Raises**: `OutOfMemoryError` on OOM, `ValueError` on invalid params

### `register_adapter(adapter_id: str, base_model_hf_id: str) -> str`
- **Precondition**: Adapter files exist at `data/adapters/<adapter_id>/lora_adapter/`
- **Returns**: `model_entry_id` (new model ID for UI registration)
- **Side effects**: Creates `data/adapters/<adapter_id>/model_metadata.json`, updates `data/app_state.json`

### `load_adapter(base_model_path: str, adapter_path: str) -> AutoModelForCausalLM`
- **Precondition**: Both paths exist
- **Returns**: PEFT model ready for inference (loaded on GPU)
- **Side effects**: None (caller responsible for model lifecycle)

## Error Handling

All public methods MUST:
- Log errors to `data/logs/<adapter_id>/training.log` via Python `logging` module
- Raise descriptive exceptions (not bare `Exception`)
- Preserve previous state on failure (no partial writes)

## Streaming Contract

The `train()` generator:
- Yields metrics every `logging_frequency` steps (default: every step)
- Yields epoch summaries after each epoch (`{"epoch": N, "type": "epoch_summary", ...}`)
- On completion, yields `{"type": "completed", "adapter_path": str}`
- On failure, yields `{"type": "failed", "error": str}` and raises the exception after yield
- Generator MUST be catchable in Gradio UI without crashing the app
