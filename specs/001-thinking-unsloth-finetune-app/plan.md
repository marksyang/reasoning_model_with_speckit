# Implementation Plan: Thinking Model Fine-Tuning Demo App

**Branch**: `001-thinking-unsloth-finetune-app` | **Date**: 2026-05-30 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-thinking-unsloth-finetune-app/spec.md`

## Summary

A Gradio web application that demonstrates unsloth thinking model fine-tuning on a single 12GB GPU. The app lets data scientists select reasoning models and datasets from Hugging Face Hub with automatic download, perform inference before and after fine-tuning, and observe real-time training progress. Built with QLoRA (4-bit quantization) + unsloth, Python 3.10+, and uv for dependency isolation.

## Technical Context

**Language/Version**: Python 3.10.12 (minimum — Unsloth does not support 3.14)

**Primary Dependencies**:

- `unsloth` (🔥 unsloth — memory-efficient 2x faster fine-tuning)
- `gradio` (interactive UI framework with streaming support)
- `transformers` (model loading, pipeline, AutoModelForCausalLM)
- `datasets` (Hugging Face dataset loading and caching)
- `bitsandbytes` (4-bit quantization viabnb+bitsandbytes)
- `peft` (LoRA adapters — PEFT library)
- `accelerate` (distributed training utilities)
- `huggingface_hub` (model/dataset download and caching)
- `sentencepiece` / `tiktoken` (tokenizer — model dependent)

**Storage**:

- **Model cache**: `$HOME/.cache/huggingface/hub` (Transformers/HF Datasets default)
- **Adapter output**: `data/adapters/<run_id>/lora_adapter/`
- **Logs**: `data/logs/<run_id>/training.log`
- **Project state**: `data/app_state.json` (selected models, datasets, runs)

**Testing**: `pytest` (unit tests for model_manager, dataset_manager, trainer; integration tests for Gradio UI)

**Target Platform**: Ubuntu 22.04 LTS, single NVIDIA GPU (RTX 3080 Ti, 12GB VRAM)

**Project Type**: Single-user local web app (Gradio Blocks)

**Performance Goals**:

- Base model inference: < 5s time-to-first-token on 7B 4-bit model
- Training throughput: > 50 examples/sec on 12GB GPU with QLoRA
- Overall demo flow (download → train → compare): < 15 minutes for 1B model, 100 examples

**Constraints**:

- **12GB VRAM hard limit** — all models and training must fit in 4-bit quantization
- **Single GPU** — no multi-GPU parallelism
- **Python 3.10–3.13** — Unsloth blocks 3.14
- **Single-user** — no authentication, no concurrency
- **All work is local** — no cloud training, no external API calls (except HF Hub for downloads)
- **240 min long timeout** for training — sessions are ephemeral (terminal or Gradio restart kills process)

**Scale/Scope**:

- **Models**: 1–7B parameter models (4-bit quantized)
- **Datasets**: 10–1,000 training examples for quick iteration
- **Runs**: 1 training run at a time; cumulative `data/adapters/` keeps history

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I — Code Quality

- **Check**: Unsloth/gradio boilerplate is external — our code (model_manager, dataset_manager, trainer, inference, app.py) MUST follow Clean Code.
- **Status**: ✅ Compliant — our code is the only part that needs Clean Code treatment.

### Principle II — Testing

- **Check**: Every module (model_manager, dataset_manager, trainer, inference) MUST have unit tests. Gradio UI integration test REQUIRED.
- **Status**: ✅ Compliant — test file planned for Phase 1 setup.

### Principle III — UX Consistency

- **Check**: Gradio must provide loading → error → success states for all operations. Training progress + logs streaming is the core UX.
- **Status**: ✅ Compliant — user stories define loading, error, and success states.

### Principle IV — Performance

- **Check**: 12GB VRAM constraint is the key performance budget. Unsloth + QLoRA 4-bit is explicitly selected to meet it.
- **Status**: ✅ Compliant — architectural choice (Unsloth + QLoRA) directly addresses this.

**Overall**: All 4 principles pass. No violations.

## Project Structure

### Documentation (this feature)

```text
specs/001-thinking-unsloth-finetune-app/
├── plan.md              # This file (implementation plan)
├── spec.md              # Feature specification
├── research.md          # Phase 0: Unsloth, QLoRA, Gradio research
├── data-model.md        # Phase 1: Model/Dataset/Training entity design
├── quickstart.md        # Phase 1: Dev machine setup (uv + Gradio)
├── contracts/
│   └── trainer_api.md   # Trainer interface contract
└── checklists/
    └── requirements.md  # Spec quality checklist
```

### Source Code (repository root)

```text
.
├── data/
│   ├── adapters/          # Finetuned LoRA adapters: <run_id>/lora_adapter/
│   ├── logs/              # Training logs: <run_id>/training.log
│   └── app_state.json     # App state (selected model, dataset, run history)
├── src/
│   ├── app.py             # Gradio Blocks UI + event handlers
│   ├── model_manager.py   # Hugging Face model download, caching, inference
│   ├── dataset_manager.py # Hugging Face dataset download, preprocessing
│   ├── trainer.py         # Unsloth QLoRA training with streaming logs
│   └── inference.py       # Base model inference + fine-tuned inference
└── tests/
    ├── conftest.py
    ├── test_model_manager.py
    ├── test_dataset_manager.py
    ├── test_trainer.py
    ├── test_inference.py
    └── test_app_ui.py
```

**Structure Decision**: Single-project layout. `src/` contains all application logic; `data/` holds runtime artifacts (adapters, logs). `tests/` at repo root for easy `pytest` discovery. No frontend/backend split — Gradio is both UI and server in one process.

## Research

### Unsloth — 2x Faster Fine-Tuning, 60% Less Memory

Unsloth patches the attention and feed-forward layers of Transformers to eliminate unnecessary operations during fine-tuning. Benchmarks from the Unsloth team show:

- 2x faster training vs vanilla Transformers
- 60% less GPU memory usage
- Full compatibility with Hugging Face PEFT + Transformers

**Key APIs we'll use**:
- `FastLanguageModel.from_pretrained()` — loads base model with quantization (CRITICAL: use `gradient_checkpointing=True` to fit 7B models into 12GB GPU)
- `FastLanguageModel.get_peft_model()` — applies LoRA adapters
- `FastLanguageModel.save_pretrained()` — saves adapter weights

### QLoRA (Quantized LoRA) — Training on 12GB GPU

QLoRA uses 4-bit NormalFloat (NF4) quantization to fit large models into limited VRAM while preserving fine-tuning quality. Combined with Double Quantization and paged optimizers, a 7B model + QLoRA adapter fits in ~6–8GB VRAM, leaving headroom for training.

**Recommended parameters (from spec, verified)**:
- `r = 8` (LoRA rank)
- `alpha = 16` (LoRA scaling)
- `dropout = 0.05`
- Load in 4-bit (`load_in_4bit=True`)
- Use double quantization + paged optimizers

### Hugging Face Hub Models

The app presents a curated dropdown of GPU-appropriate models. Two tiers:

| Tier | Examples | 4-bit VRAM (estimate) | Notes |
|------|----------|------------------------|-------|
| 1B | `Qwen/Qwen2.5-0.5B-Instruct`, `meta-llama/Llama-3.2-1B-Instruct` | ~1–2 GB | Fast, good for demos |
| 7B | `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` | ~6–8 GB | Stronger reasoning |

Selection criteria: 4-bit quantization compatible, good reasoning performance, 1–7B parameters.

### Hugging Face Datasets

Pre-curated dataset list:

| Dataset | Split | Examples | Domain |
|---------|-------|----------|--------|
| `gsm8k` | train/test | 7.4K / 1.3K | Math word problems |
| `open-compass/long-form-math` | train | ~1K | Long reasoning |
| `WizardLM/vicuna_dpo` | train | ~63K | Instruction following |
| `tatsu-lab/alpaca` | train | 52K | Instruction tuning |

For quick demo (P1 scope), use `gsm8k` or a small Alpaca subset (100–500 examples).

### Gradio Blocks — Event-Driven UI

Gradio Blocks provides the UI framework with:
- **Generators** for streaming training progress (yield → Gradio progress bar)
- **Progress bar** (`gr.Progress`) for real-time training visibility
- **Tabbed layout**: Preparation (US1+US2) → Inference (US3) → Training (US4) → Evaluate (US5) → Compare (US6)
- **Dynamic UI control** (enable/disable buttons, update dropdowns)
- **Auto-reload** during development (press Shift+Enter)

### uv — Fast Python Package Manager

`uv` provides:
- ~10–100x faster pip install for dependencies
- `uv venv` for creating isolated virtual environments
- `uv run` for executing scripts in the environment
- `pyproject.toml` for project metadata and dependencies

This replaces the traditional pip + venv workflow and makes setup ~2x faster.

### Data Format for Fine-Tuning

The spec requires GSM8K-style question/answer format:

```python
# Source: GSM8K Hugging Face dataset
dataset["train"][0] = {
    "question": "If there are 3 cars in the parking lot...",
    "answer": "There are 24 cars in the parking lot.",
}

# For fine-tuning, convert to Alpaca-style prompts:
prompt = f"### Question:\n{q}\n\n### Answer:\n{a}"
```

For future extensibility, the system supports both `question/answer` (GSM8K) and `input/output` (Alpaca) formats.

## Quickstart

**Prerequisites**: Ubuntu 22.04, Python 3.10.12, NVIDIA GPU with 12GB VRAM, CUDA 12.x.

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone repo
git clone <repo-url>
cd reasonung_model

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

**Testing**:
```bash
uv pip install pytest pytest-cov
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Phase-by-Phase Implementation

### Phase 0: Setup & Dependencies (1 day)

Set up the project structure, environment, and basic scaffolding.

1. Initialize project with `pyproject.toml` (metadata, dependencies)
2. Create `src/` and `tests/` directories
3. Create `data/` directories (`adapters/`, `logs/`)
4. Install core dependencies (unsloth, gradio, transformers, datasets, peft, bitsandbytes)
5. Write `app.py` skeleton (Gradio Blocks layout with 4 tabs, no logic)
6. Write `model_manager.py` skeleton (download + inference stubs)
7. Write `dataset_manager.py` skeleton (download + preview stubs)
8. Write `trainer.py` skeleton (start_training stub)
9. Write `inference.py` skeleton (base + fine-tuned inference stubs)
10. Write `tests/` setup with conftest.py

### Phase 1: Core Module Implementation (2 days)

Implement model_manager, dataset_manager, and inference logic. **Note**: for 7B models on 12GB GPU, training uses `gradient_checkpointing=True` to fit within memory.

1. **model_manager.py** — Implement HF model download with progress callback, cache management (check cached models on startup), download-in-progress state
2. **dataset_manager.py** — Implement HF dataset download with progress callback, summary stats display, retry on failure
3. **inference.py** — Implement base model inference (4-bit loading + pipeline), streaming output (token-by-token generation), fine-tuned inference (base model + LoRA adapter)
4. Write unit tests for model_manager (test downloads, cache hits)
5. Write unit tests for dataset_manager (test download, preview)
6. Write unit tests for inference (test pipeline, adapter loading)
7. Integration test: download a small model (1B), run inference, verify output

### Phase 2: Training Pipeline (2 days)

Implement Unsloth + QLoRA training with streaming logs.

1. **trainer.py** — Implement:
   - Model + dataset validation (both must be downloaded)
   - Unsloth FastLanguageModel loading with 4-bit quantization
   - LoRA adapter configuration (default params: r=8, alpha=16, dropout=0.05)
   - Trainer initialization (SFTTrainer)
   - Training loop wrapped in a **generator** that yields progress (loss, perplexity, epoch metrics)
   - `save_pretrained()` on completion → registers LoRA adapter in `data/adapters/<run_id>/lora_adapter/`
2. **Gradio training UI** — Wire `trainer.py` to Gradio UI:
   - Progress bar (tied to generator)
   - Log viewer (auto-scrolling text, pause/resume buttons)
   - Final model dropdown update (add new adapter)
3. Write unit tests for trainer (validation, adapter saving)
4. Integration test: train on GSM8K (100 examples, 1 epoch), verify adapter created

### Phase 3: UI Integration (1 day)

Connect Gradio UI to all modules, add events and state management.

1. **app.py** — Wire up all events:
   - Tab 1 (Preparation): Model select → download → dataset select → download
   - Tab 2 (Inference): Select model → enter prompt → run → display output
   - Tab 3 (Training): Select model + dataset → LoRA params → start training → progress + logs → registered model
   - Tab 4 (Evaluate): Select fine-tuned model → 5 examples → side-by-side comparison + accuracy
2. Shared state object (global dict in app.py) holds: `available_models`, `selected_model`, `available_datasets`, `selected_dataset`, `training_runs`, `registered_adapters`
3. UI state: training disables download/model selectors, "Evaluate" only active after training completes
4. Write integration test for Gradio UI (app.py functional smoke test)

### Phase 4: Polish & Verification (1 day)

Final quality pass, edge case handling, and verification.

1. Error handling for all modules (network errors, OOM, file system errors)
2. Logging setup (`logging` module — structured logs to `data/logs/`)
3. Progress lockout during training (FR-020)
4. "Compare Fine-Tuned Inference" tab (US6) — free-form prompt comparison
5. Verification against spec:
   - SC-001: Can complete workflow in 15 min?
   - SC-002: 7B model trains without OOM?
   - SC-003: Progress updates every batch?
   - SC-004: Downloads work without manual intervention?
   - SC-005: Training shows measurable improvement?
6. Final smoke test end-to-end

## Estimated Effort: 6–7 days total (1 developer)

## Dependencies & Execution Order

1. **Phase 0** (no dependencies) → establishes project structure
2. **Phase 1** (depends on Phase 0) → model and dataset loading + inference
3. **Phase 2** (depends on Phase 1, because training needs a working model loader) → training pipeline
4. **Phase 3** (depends on Phases 1+2) → connects all modules to Gradio UI
5. **Phase 4** (depends on Phase 3) → polish and verification

**Parallel Opportunities**: Within Phase 0, directory creation and stub file writing can be parallelized. Within Phase 1, model_manager and dataset_manager can be implemented in parallel (different files, no dependencies).
