# Data Model: Thinking Model Fine-Tuning Demo App

## Entities

### Model

Represents a Hugging Face model (base or fine-tuned).

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `hf_id` | string | Yes | — |
| `name` | string | Yes | Auto from HF |
| `param_count` | int | Yes | — |
| `is_finetuned` | bool | Yes | `False` |
| `adapter_path` | string \| null | Conditional | `None` (if not finetuned) |
| `cache_path` | string | Yes | Auto from HF |
| `status` | enum | Yes | `pending`, `downloading`, `cached`, `failed` |
| `created_at` | datetime | Yes | Current time |

**Relations**:
- A Model can have one Training Run (the run that fine-tuned it)
- A Model can have multiple Inference Results

### Dataset

Represents a Hugging Face dataset for fine-tuning.

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `hf_id` | string | Yes | — |
| `name` | string | Yes | Auto from HF |
| `train_size` | int | Yes | From HF |
| `test_size` | int | Yes | From HF |
| `column_names` | list[string] | Yes | From HF |
| `formatted_for_finetuning` | bool | Yes | `False` |
| `cache_path` | string | Yes | Auto from HF |
| `status` | enum | Yes | `pending`, `downloading`, `cached`, `failed` |

**Relations**:
- A Dataset is associated with many Training Runs

### Training Run

Represents one fine-tuning execution.

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `id` | string | Yes | UUID |
| `model_id` | string | Yes | FK → Model.hf_id |
| `dataset_id` | string | Yes | FK → Dataset.hf_id |
| `lora_params` | dict | Yes | `{r: 8, alpha: 16, dropout: 0.05}` |
| `status` | enum | Yes | `pending`, `running`, `completed`, `failed` |
| `metrics` | dict | Conditional | `None` (filled on completion) |
| `adapter_path` | string | Conditional | `None` (created on completion) |
| `started_at` | datetime | Conditional | `None` |
| `completed_at` | datetime | Conditional | `None` |
| `error_message` | string \| null | Conditional | `None` |

**Relations**:
- A Training Run produces exactly one LoRA Adapter (output file)
- The resulting model is a new Model entry (fine-tuned) with the adapter_path set

### Inference Result

Represents one model response to a prompt.

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `id` | string | Yes | UUID |
| `model_id` | string | Yes | FK → Model.hf_id |
| `prompt` | string | Yes | — |
| `output` | string | Yes | Generated response |
| `generated_at` | datetime | Yes | Current time |
| `training_run_id` | string \| null | Conditional | `None` |

**Relations**:
- An Inference Result is associated with exactly one Model
- If the model is fine-tuned, it has a training_run_id reference

## Persistence

All data persists to:

- **`data/app_state.json`**: Current selection (selected model, selected dataset, adapter list)
- **`data/adapters/<run_id>/lora_adapter/`**: PEFT adapter output
- **`data/logs/<run_id>/training.log`**: Training log output
- **`~/.cache/huggingface/hub/`**: HF model and dataset cache (managed by Transformers/Datasets)

No database — the app uses JSON + filesystem for all persistence.
