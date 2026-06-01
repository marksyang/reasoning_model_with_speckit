# Project: Thinking Model Fine-Tuning Demo App

Local Gradio web app for demonstrating Unsloth thinking model fine-tuning on Apple Silicon (MLX) or NVIDIA GPU (PyTorch).

## Tech Stack

- **Python** 3.10+ (< 3.14, Unsloth incompatibility)
- **Backend**: Unsloth QLoRA + `mlx_lm` (macOS MLX) or `trl.SFTTrainer` (PyTorch)
- **UI**: Gradio Blocks API
- **Package management**: `uv`

## Project Structure

```
src/
  app.py             # Gradio Blocks UI + event handlers
  model_manager.py   # Model download, caching, inference
  dataset_manager.py # Dataset download, formatting for fine-tuning
  trainer.py         # Unsloth QLoRA training with MLX/PyTorch backend
  inference.py       # Base + fine-tuned model inference
data/
  app_state.json     # App state persistence
  adapters/          # Fine-tuned LoRA adapters (generated)
  logs/              # Training logs (generated)
tests/               # Pytest test suite
specs/               # Spec Kit feature specs
```

## Common Commands

```bash
# Setup
uv venv .venv --python 3.10 && source .venv/bin/activate
uv pip install -r requirements.txt

# Run app
python -m src.app

# Run tests
pytest tests/ -v
```

## Key Patterns

- Use **relative imports** within `src/` package (`from .model_manager import ...`)
- All dataset names use full HuggingFace namespace (e.g., `openai/gsm8k`)
- Training progress streams via `queue.Queue` callback pattern to Gradio
- LoRA adapters saved as `adapters.safetensors` in `data/adapters/{id}/`

## Documentation

- All generated documentation, comments, Spec Kit specs, and project files must be written in **Traditional Chinese (繁體中文)**.
- Technical terms, code identifiers, and English proper nouns may remain in their original form.
