# Thinking Model Fine-Tuning Demo App

A local Gradio web app for demonstrating Unsloth thinking model fine-tuning on Apple Silicon (MLX) or NVIDIA GPU (PyTorch). Select reasoning models and datasets from Hugging Face Hub, run inference before and after fine-tuning, and observe real-time training progress.

## Features

- **Model Download**: Curated list of GPU-appropriate thinking/reasoning models with automatic download and caching
- **Dataset Download**: Pre-selected public reasoning datasets with streaming download and cache detection
- **Base Model Inference**: Run inference on selected model with adjustable parameters (temperature, top-p, max tokens)
- **LoRA Fine-Tuning**: Unsloth-powered 4-bit fine-tuning with real-time progress bar, streaming logs, and Stop Training support
- **Model Evaluation**: Evaluate fine-tuned model on validation examples with accuracy tracking

## Prerequisites

- **OS**: macOS (Apple Silicon, MLX backend) or Ubuntu 22.04+ with NVIDIA GPU (PyTorch backend)
- **GPU**: NVIDIA GPU with >=12GB VRAM or Apple Silicon with >=16GB unified memory
- **Python**: 3.10+ (Unsloth does not support 3.14+)
- **Git**
- **uv** (recommended for faster installs)

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/marksyang/reasoning_model_with_speckit.git
cd reasoning_model_with_speckit

# 2. Create virtual environment
uv venv .venv --python 3.10
source .venv/bin/activate

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Run app
python -m src.app
# -> Open http://127.0.0.1:7860
```

## Project Structure

```
.
|-- src/
|   |-- app.py             # Gradio Blocks UI + event handlers
|   |-- model_manager.py   # Model download, caching, inference
|   |-- dataset_manager.py # Dataset download, formatting for fine-tuning
|   |-- trainer.py         # Unsloth QLoRA training with MLX/PyTorch backend
|   `-- inference.py       # Base + fine-tuned model inference
`-- data/
    |-- app_state.json     # App state (models, datasets, runs)
    |-- adapters/          # Fine-tuned LoRA adapters (generated)
    `-- logs/              # Training logs (generated)
```

## Usage Guide

### 1. Preparation Tab

Select a model and dataset, then download:

**Models** (curated for 12GB+ VRAM):

| Model | Params | 4-bit VRAM | Notes |
|-------|--------|------------|-------|
| Qwen 2.5 0.5B | 500M | ~1 GB | Fastest, good for demos |
| Qwen 2.5 1.5B | 1.5B | ~1.5 GB | Balanced performance |
| LLaMA 3.2 1B | 1B | ~2 GB | Strong reasoning |
| LLaMA 3.1 8B | 8B | ~6 GB | Best quality (needs gradient checkpointing) |

**Datasets**:

| Dataset | Description | Use Case |
|---------|-------------|----------|
| GSM8K | Grade school math word problems | Math reasoning (recommended) |
| SQuAD | Stanford Question Answering Dataset | QA tasks |
| CoT GSM8K | GSM8K with chain-of-thought reasoning | Chain-of-thought training |
| No Robots | Human-written instruction data | Instruction tuning |
| OpenAssistant | Multi-turn conversation dataset | Dialogue |
| UltraChat 200K | High-quality multi-turn conversations | Large-scale dialogue |

### 2. Inference Tab

Select a downloaded model, enter a reasoning prompt (e.g., a math word problem), adjust generation parameters, and run inference. Output streams token-by-token in real time.

### 3. Training Tab

1. Select your model and dataset
2. Configure LoRA parameters (r, alpha, dropout)
3. Set training hyperparameters (epochs, learning rate)
4. Click "Start Training"
5. Watch progress bar, streaming logs, and iteration count
6. Click "Stop Training" to interrupt training mid-way
7. On completion, the fine-tuned model auto-registers in the dropdown

**Recommended settings**:
- `LoRA Rank`: 8
- `LoRA Alpha`: 16
- `LoRA Dropout`: 0.05
- `Epochs`: 1 (for quick demos)
- `Learning Rate`: 2e-4

### 4. Evaluate Tab

1. Select a model (base or fine-tuned)
2. Enter 3 prompt-answer pairs (pre-populated with sample math problems)
3. Click "Run Evaluation"
4. View per-question accuracy and overall accuracy percentage

## Architecture

### Backend Selection

The app uses Unsloth for optimized model loading and fine-tuning. Unsloth automatically selects the appropriate backend:

| Platform | Backend | Framework | Training Engine |
|----------|---------|-----------|-----------------|
| macOS (Apple Silicon) | MLX | `mlx_lm` | `mlx_lm.lora.train_model` |
| Linux (NVIDIA GPU) | PyTorch | `transformers` | `trl.SFTTrainer` / `Trainer` |

### Progress Streaming

Training progress is streamed from the MLX/PyTorch training loop to the Gradio UI via a queue-based callback pattern:

```
Training Loop -> Callback.on_train_loss_report()
  -> queue.Queue.put(progress_dict)
  -> train() generator yields from queue
  -> Gradio handler yields tuple to (progress_bar, progress_textbox, log_textbox, dropdown, buttons)
```

### Dataset Format

All datasets use full HuggingFace namespace (e.g., `openai/gsm8k` instead of `gsm8k`) with the `config` parameter for multi-config datasets. Data is formatted as Alpaca-style prompts:

```
### Question:
{question}

### Answer:
{answer}
```

### Adapter Storage

Fine-tuned LoRA adapters are saved as `adapters.safetensors` in `data/adapters/{adapter_id}/`. Adapter metadata is registered in `data/app_state.json` for persistence across app restarts.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `cuda:0 out of memory` | Select a smaller model (0.5B instead of 8B), or reduce LoRA rank |
| `HF Hub download fails` | Check internet; set `HF_ENDPOINT=https://hf-mirror.com` for China |
| `Unsloth import error` | Verify Python version is 3.10-3.13 (not 3.14+) |
| `ImportError: cannot import name 'FastLanguageModel'` | Upgrade: `pip install --upgrade unsloth` |
| `ModuleNotFoundError: No module named 'mlx'` | Install MLX: `pip install mlx mlx-lm` (macOS only) |
| Gradio shows "Connection refused" | Check port 7860; try `python -m src.app --port 7861` |
| Training shows `local variable referenced before assignment` | This was fixed in recent commits — pull latest changes |

## License

MIT
