# Thinking Model Fine-Tuning Demo App

A local Gradio web app for demonstrating unsloth thinking model fine-tuning on a single 12GB GPU. Select reasoning models and datasets from Hugging Face Hub, run inference before and after fine-tuning, and observe real-time training progress.

## Features

- **Model Download**: Curated list of GPU-appropriate thinking/reasoning models with automatic download and caching
- **Dataset Download**: Pre-selected public reasoning datasets (GSM8K, SQuAD, etc.) with streaming download
- **Base Model Inference**: Run inference on selected model with adjustable parameters (temperature, top-p, max tokens)
- **QLoRA Fine-Tuning**: Unsloth-powered 4-bit fine-tuning with real-time progress bar and streaming logs
- **Model Comparison**: Evaluate and compare base model vs fine-tuned model on validation examples

## Prerequisites

- **OS**: Ubuntu 22.04 LTS or later
- **GPU**: NVIDIA GPU with ≥12GB VRAM (tested on RTX 3080 Ti)
- **CUDA**: CUDA 12.x installed (`nvidia-smi` shows driver ≥535)
- **Python**: 3.10.12 (Unsloth does not support 3.14+)
- **Git**
- **uv** (recommended for faster installs)

## Quick Start

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repo
git clone https://github.com/marksyang/reasoning_model_with_speckit.git
cd reasoning_model_with_speckit

# 3. Create virtual environment
uv venv .venv --python 3.10.12

# 4. Activate
source .venv/bin/activate

# 5. Install dependencies
uv pip install -r requirements.txt

# 6. Run app
python -m src.app
# → Open http://127.0.0.1:7860
```

## Project Structure

```
.
├── src/
│   ├── app.py             # Gradio Blocks UI + event handlers
│   ├── model_manager.py   # Model download, caching, inference
│   ├── dataset_manager.py # Dataset download, formatting for fine-tuning
│   ├── trainer.py         # Unsloth QLoRA training with streaming logs
│   └── inference.py       # Base + fine-tuned model inference
├── tests/
│   ├── conftest.py        # Shared test fixtures
│   ├── test_model_manager.py
│   ├── test_dataset_manager.py
│   ├── test_trainer.py
│   └── test_inference.py
├── data/
│   ├── app_state.json     # App state (models, datasets, runs)
│   ├── adapters/          # Fine-tuned LoRA adapters (generated)
│   └── logs/              # Training logs (generated)
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Usage Guide

### 1. Preparation Tab

Select a model and dataset, then download:

**Models** (curated for 12GB VRAM):
| Model | Params | 4-bit VRAM | Notes |
|-------|--------|------------|-------|
| Qwen 2.5 0.5B | 500M | ~1 GB | Fastest, good for demos |
| LLaMA 3.2 1B | 1B | ~2 GB | Strong reasoning |
| LLaMA 3.1 8B | 8B | ~6 GB | Best quality (may need gradient_checkpointing) |

**Datasets**:
| Dataset | Description | Use Case |
|---------|-------------|----------|
| GSM8K | Grade school math word problems | Math reasoning (recommended) |
| SQuAD | Question answering | QA tasks |
| OpenAssistant | Multi-turn conversations | Dialogue |

### 2. Inference Tab

Select a downloaded model, enter a reasoning prompt (e.g., a math word problem), adjust generation parameters, and run inference. Output streams token-by-token in real time.

### 3. Training Tab

1. Select your model and dataset
2. Configure LoRA parameters (r, alpha, dropout)
3. Set training hyperparameters (epochs, learning rate)
4. Click "Start Training"
5. Watch progress bar and streaming logs
6. On completion, the fine-tuned model auto-registers in the dropdown

**Recommended settings**:
- `LoRA Rank`: 8
- `LoRA Alpha`: 16
- `LoRA Dropout`: 0.05
- `Epochs`: 1 (for quick demos)
- `Learning Rate`: 2e-4

### 4. Evaluate Tab

1. Select the fine-tuned model
2. Run inference on 3 sample math prompts
3. View accuracy comparison with expected answers

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `cuda:0 out of memory` | Select a smaller model (1B instead of 7B), or reduce LoRA rank |
| `HF Hub download fails` | Check internet; set `HF_ENDPOINT=https://hf-mirror.com` for China |
| `Unsloth import error` | Verify Python version is 3.10–3.13 (not 3.14+) |
| `ImportError: cannot import name 'FastLanguageModel'` | Upgrade: `pip install --upgrade unsloth` |
| Gradio shows "Connection refused" | Check port 7860; try `python -m src.app --port 7861` |

## Testing

```bash
uv pip install pytest pytest-cov
pytest tests/ -v --cov=src --cov-report=term-missing
```

## License

MIT
