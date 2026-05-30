# Quickstart: Thinking Model Fine-Tuning Demo App

**Setup time**: 10 minutes | **Hardware**: Ubuntu 22.04, Python 3.10+, NVIDIA GPU (12GB VRAM)

## Prerequisites

- Ubuntu 22.04 LTS or later
- NVIDIA GPU with ≥12GB VRAM (tested on RTX 3080 Ti)
- CUDA 12.x installed (`nvidia-smi` shows driver ≥535)
- Git

## One-Command Setup

```bash
# 1. Install uv (~30s)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repo
git clone <repo-url> && cd reasoning_model

# 3. Create virtual environment (Python 3.10 — Unsloth incompatible with 3.14)
uv venv .venv --python 3.10.12

# 4. Activate
source .venv/bin/activate

# 5. Install dependencies (~3 min)
uv pip install -r requirements.txt

# 6. Run the app
python -m src.app
```

## Verify It Works

Navigate to `http://127.0.0.1:7860` in your browser. The Gradio UI should load. Click "Download" on a model (e.g., `Qwen/Qwen2.5-0.5B-Instruct`) to verify download works.

## Run Tests

```bash
uv pip install pytest pytest-cov
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Common Issues

| Problem | Fix |
|---------|-----|
| `cuda:0 out of memory` | Select a smaller model (1B instead of 7B), or reduce LoRA rank to 4 |
| `HF Hub download fails` | Check internet; set `HF_ENDPOINT=https://hf-mirror.com` for China |
| `Unsloth import error` | Verify Python version is 3.10–3.13 (not 3.14+) |
| `ImportError: cannot import name 'FastLanguageModel'` | Upgrade: `pip install --upgrade unsloth` |
| Gradio shows "Connection refused" | Check if port 7860 is occupied; try `python -m src.app --port 7861` |
