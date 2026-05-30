---

description: "Task list for Thinking Model Fine-Tuning Demo App"
---

# Tasks: Thinking Model Fine-Tuning Demo App

**Input**: Design documents from `/specs/001-thinking-unsloth-finetune-app/`

**Prerequisites**: plan.md (tech stack, libraries, structure), spec.md (user stories P1–P6), data-model.md (4 entities), contracts/trainer_api.md (trainer interface), research.md (Unsloth, QLoRA, Gradio)

**Tests**: Unit tests for every module + integration test for Gradio UI

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Path Conventions

- **Source Code**: `src/` at repository root
- **Tests**: `tests/` at repository root
- **Data**: `data/` at repository root (runtime artifacts)

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create project directories: `src/`, `tests/`, `data/adapters/`, `data/logs/`
- [ ] T002 Create `pyproject.toml` with project metadata, Python 3.10.12 minimum, dependencies (unsloth, gradio, transformers, datasets, peft, bitsandbytes, accelerate, huggingface_hub, pytest, pytest-cov)
- [ ] T003 Create `data/app_state.json` skeleton with keys: `available_models`, `available_datasets`, `registered_adapters`, `selected_model`, `selected_dataset`, `training_runs`
- [ ] T004 Create `src/app.py` skeleton with Gradio Blocks layout (5 tabs: Preparation, Inference, Training, Evaluate, Compare) — all widgets defined, no event logic
- [ ] T005 [P] Create `src/model_manager.py` stub with functions: `list_models()`, `download_model()`, `get_model_status()`
- [ ] T006 [P] Create `src/dataset_manager.py` stub with functions: `list_datasets()`, `download_dataset()`, `get_dataset_summary()`
- [ ] T007 [P] Create `src/trainer.py` stub with functions: `validate_training_inputs()`, `train()`, `register_adapter()`, `load_adapter()`
- [ ] T008 [P] Create `src/inference.py` stub with functions: `base_inference()`, `finetuned_inference()`
- [ ] T009 Create `tests/conftest.py` with fixtures: `tmp_dir`, `mock_model_path`, `mock_dataset_path`
- [ ] T010 Create `tests/test_model_manager.py` stub
- [ ] T011 Create `tests/test_dataset_manager.py` stub
- [ ] T012 Create `tests/test_trainer.py` stub
- [ ] T013 Create `tests/test_inference.py` stub

**Checkpoint**: Project structure complete, all stubs in place, pytest discovery works (`pytest tests/ --collect-only`).

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core model/dataset loading infrastructure that MUST be complete before any user story can run inference or training

**⚠️ CRITICAL**: No inference or training work can begin until Phase 2 is complete.

- [ ] T014 Implement model manager — `model_manager.py`:
  - `list_models()`: Query Hugging Face Hub for curated model list (`Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`), return metadata (name, param_count)
  - `download_model(hf_id)`: Use `huggingface_hub.hf_hub_download` or `AutoModel.from_pretrained()` with progress callback
  - `get_cached_models()`: Scan `~/.cache/huggingface/hub` for downloaded models, update `data/app_state.json`
  - `register_model(hf_id)`: Add model to `app_state.json` with status `cached`
- [ ] T015 Implement dataset manager — `dataset_manager.py`:
  - `list_datasets()`: Return curated dataset list (`gsm8k`, `open-compass/long-form-math`, `tatsu-lab/alpaca`) with metadata
  - `download_dataset(hf_id)`: Use `datasets.load_dataset()` with progress tracking
  - `get_dataset_summary(hf_id)`: Return split sizes, column names for the dataset
  - `format_for_finetuning(dataset, source_format="question/answer")`: Convert to Alpaca-style prompts: `"### Question:\n{q}\n\n### Answer:\n{a}"`
- [ ] T016 Implement inference module — `inference.py`:
  - `base_inference(model_path, prompt, generation_kwargs)`: Load model 4-bit via `transformers.Pipeline`, generate response, stream tokens one-by-one using `yield`
  - `finetuned_inference(base_model_path, adapter_path, prompt, generation_kwargs)`: Load base model + `PeftModel.from_pretrained(base_model, adapter_path)`, generate response
  - Both functions MUST support streaming via generator pattern (yield token → Gradio displays)
- [ ] T017 Write unit tests for model manager — `tests/test_model_manager.py`:
  - Test `list_models()` returns expected model list
  - Test `get_cached_models()` correctly identifies cached models
  - Test `register_model()` updates `app_state.json`
- [ ] T018 Write unit tests for dataset manager — `tests/test_dataset_manager.py`:
  - Test `list_datasets()` returns expected dataset list
  - Test `get_dataset_summary()` returns correct structure
  - Test `format_for_finetuning()` converts GSM8K format to Alpaca format

**Checkpoint**: Foundation ready — `model_manager`, `dataset_manager`, `inference.py` all working. US1–US3 can begin.

---

## Phase 3: User Story 1 - Select and Download Base Model (Priority: P1) 🎯 MVP

**Goal**: User can browse a curated list of thinking models, download one, and use it for inference.

**Independent Test**: User selects a model from dropdown, clicks download, and the model becomes available for inference without fine-tuning.

### Tests for User Story 1

- [ ] T019 [P] [US1] Contract test: Model dropdown returns models with `hf_id`, `name`, `param_count` fields in `tests/test_model_manager.py`
- [ ] T020 [P] [US1] Integration test: Download a small model (1B) and verify it appears as cached in `tests/test_app_ui.py`

### Implementation for User Story 1

- [ ] T021 [US1] Wire `model_manager.list_models()` to Gradio UI dropdown in Tab 1 (Preparation) — `src/app.py`
- [ ] T022 [US1] Wire `model_manager.download_model()` to "Download" button click in `src/app.py` — connect to progress bar
- [ ] T023 [US1] Wire `model_manager.get_cached_models()` to app startup — populate dropdowns on load in `src/app.py`
- [ ] T024 [US1] Add download state tracking — show progress during download, disable dropdown during download, handle network errors with toast message in `src/app.py`

**Checkpoint**: At this point, User Story 1 is fully functional — user can select and download a model, and it's ready for inference.

---

## Phase 4: User Story 2 - Select and Download Dataset (Priority: P1)

**Goal**: User can browse, download, and preview datasets.

**Independent Test**: User selects a dataset, downloads it, and sees summary statistics (split sizes, column names).

### Tests for User Story 2

- [ ] T025 [P] [US2] Contract test: Dataset dropdown returns datasets with `hf_id`, `name`, `train_size` fields in `tests/test_dataset_manager.py`
- [ ] T026 [P] [US2] Integration test: Download GSM8K dataset and verify summary display in `tests/test_app_ui.py`

### Implementation for User Story 2

- [ ] T027 [US2] Wire `dataset_manager.list_datasets()` to Gradio UI dropdown in Tab 1 — `src/app.py`
- [ ] T028 [US2] Wire `dataset_manager.download_dataset()` to "Download" button click — connect to progress bar
- [ ] T029 [US2] Wire `dataset_manager.get_dataset_summary()` to display dataset summary (sample count, splits, columns) below the dropdown in `src/app.py`
- [ ] T030 [US2] Wire `dataset_manager.format_for_finetuning()` — called automatically when user starts training with a dataset in `src/app.py`

**Checkpoint**: At this point, User Stories 1 AND 2 are both functional — user can prepare both a model and a dataset.

---

## Phase 5: User Story 3 - Infer Using Base Model (Priority: P1)

**Goal**: User can enter prompts and get reasoning model responses.

**Independent Test**: User enters a prompt, runs inference, and receives a valid model response displayed on screen.

### Tests for User Story 3

- [ ] T031 [P] [US3] Contract test: `base_inference()` accepts model_path, prompt, kwargs and yields tokens in `tests/test_inference.py`
- [ ] T032 [P] [US3] Contract test: `finetuned_inference()` accepts base path + adapter path and yields tokens in `tests/test_inference.py`

### Implementation for User Story 3

- [ ] T033 [US3] Wire inference module to Tab 2 (Inference) in `src/app.py`: model dropdown → prompt textarea → generation parameter inputs → "Run Inference" button → output text area
- [ ] T034 [US3] Wire streaming output: inference generator yields → Gradio displays progressively in real time in `src/app.py`
- [ ] T035 [US3] Wire generation parameters: max_tokens, temperature, top_p inputs update inference call in `src/app.py`
- [ ] T036 [US3] Error handling for inference: display network error, model loading error, and generation error messages in `src/app.py`

**Checkpoint**: At this point, User Stories 1, 2, AND 3 are functional — user can select model + dataset, download them, and run inference on both.

---

## Phase 6: User Story 4 - Fine-Tune Model (Priority: P2)

**Goal**: User can initiate QLoRA training with real-time progress and streaming logs.

**Independent Test**: User can initiate training, observe progress, and obtain a fine-tuned model that can run inference.

### Tests for User Story 4

- [ ] T037 [P] [US4] Contract test: `trainer.validate_training_inputs()` rejects missing model/dataset paths in `tests/test_trainer.py`
- [ ] T038 [P] [US4] Integration test: Train on 100 GSM8K examples (1 epoch) with `Qwen/Qwen2.5-7B-Instruct` and verify adapter created in `tests/test_app_ui.py`

### Implementation for User Story 4

- [ ] T039 [US4] Implement trainer module — `src/trainer.py`:
  - `validate_training_inputs(base_model_path, dataset_path)`: Check both paths exist
  - `train(base_model_path, dataset_path, adapter_id, lora_params, progress_callback)`: Full Unsloth QLoRA training wrapped in generator that yields `{"epoch", "loss", "perplexity", "lr", "message"}` per step
  - Use `Unsloth FastLanguageModel.from_pretrained(load_in_4bit=True, gradient_checkpointing=True)`, `get_peft_model()` with r=8, alpha=16, dropout=0.05
  - Training loop yields: epoch-level summaries + batch-level loss
  - On completion: `save_pretrained(adapter_path)` → writes to `data/adapters/<adapter_id>/lora_adapter/`
  - On failure: logs error to `data/logs/<adapter_id>/training.log`
  - Verify training fits in 12GB GPU before committing to large runs
- [ ] T040 [US4] Wire training UI to Tab 3 (Training) in `src/app.py`:
  - Also implement FR-020: disable all non-training UI elements during training
  - Model dropdown (required) + dataset dropdown (required) + LoRA param inputs (r, alpha, dropout with defaults)
  - "Start Training" button wired to `trainer.train()` generator
  - Progress bar tied to training generator
  - Log viewer with auto-scroll + pause/resume buttons
  - Training log written to `data/logs/<adapter_id>/training.log`
- [ ] T041 [US4] On training completion in `src/app.py`:
  - Auto-register adapter: call `trainer.register_adapter(adapter_id, base_model_hf_id)`
  - Add new fine-tuned model to model selector dropdown
  - Enable "Evaluate" button
  - Update `data/app_state.json`
- [ ] T042 [US4] Implement training termination: "Stop" button shows confirmation dialog, gracefully stops training loop in `src/app.py`
- [ ] T043 [US4] Implement training error handling: display error message, show error log content, disable "Evaluate" button in `src/app.py`

**Checkpoint**: At this point, User Stories 1–4 are functional — complete flow from model selection to fine-tuning with streaming logs.

---

## Phase 7: User Story 5 - Evaluate Fine-Tuned Model (Priority: P3)

**Goal**: User can evaluate the fine-tuned model on 5 validation examples and see accuracy comparison.

**Independent Test**: User selects fine-tuned model, runs 5 validation examples, and sees a summary of per-question and overall accuracy.

### Tests for User Story 5

- [ ] T044 [P] [US5] Contract test: Evaluation flow loads 5 examples, runs inference on both models, and computes accuracy in `tests/test_app_ui.py`

### Implementation for User Story 5

- [ ] T057 [US5] Implement eval metric in `src/eval.py`:
  - `compute_accuracy(predictions, targets)` function
  - Support GSM8K format (question/answer) and Alpaca format (input/output)
  - Return per-example and total accuracy
- [ ] T045 [US5] Wire Tab 4 (Evaluate) in `src/app.py`:
  - Model dropdown for fine-tuned model
  - "Run Evaluation" button loads 5 validation examples from the dataset
  - For each example: display prompt, run inference on both models, show side-by-side output
- [ ] T046 [US5] Implement accuracy tracking in `src/app.py`:
  - After inference on 5 validation examples, call `compute_accuracy(predictions, targets)` from `src/eval.py`
  - On completion: display summary table (per-question + total accuracy %)
- [ ] T047 [US5] Guard "Evaluate" tab: only active when at least one fine-tuned model is registered in `src/app.py`

**Checkpoint**: At this point, User Stories 1–5 are functional — full workflow from download to evaluation.

---

## Phase 8: User Story 6 - Compare Fine-Tuned Inference (Priority: P3)

**Goal**: User can explore custom prompts on both models side by side.

**Independent Test**: User enters a custom prompt, runs inference on both models, and observes the output.

### Tests for User Story 6

- [ ] T048 [P] [US6] Contract test: Custom prompt comparison shows both model outputs side by side in `tests/test_app_ui.py`

### Implementation for User Story 6

- [ ] T049 [US6] Wire "Compare" tab in `src/app.py`:
  - Prompt textarea
  - Model selector dropdown (all models: base + fine-tuned)
  - "Compare" button runs inference on both selected models
  - Output: side-by-side display of both model responses
  - Note on VRAM: when simultaneously loading both models exceeds GPU memory
    (typical on 12GB GPUs with >3B models), unload the base model before
    loading the fine-tuned model
- [ ] T050 [US6] Add "Run Inference" button enable/disable logic: only active when a model is selected and downloaded in `src/app.py`

**Checkpoint**: All 6 user stories are functional.

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T051 [P] Add error handling across all modules: network timeout errors, HF Hub rate limiting, GPU OOM (fallback message with advice)
- [ ] T052 [P] Add structured logging: `logging` module in each module, logs to `data/logs/app.log`
- [ ] T053 [US4] Lock downloads and model selection during active training (FR-020): disable all non-training UI elements in `src/app.py`; verify lockout works correctly
- [ ] T054 [P] Add reset/retry flow: allow user to retry failed downloads and failed training runs in `src/app.py`; also verify FR-020 (training lockout) works correctly
- [ ] T055 [P] Integration smoke test: end-to-end test downloading a small model, downloading GSM8K (10 examples), running base inference, verifying output in `tests/test_app_ui.py`
- [ ] T056 Verify against spec:
  - SC-001: Can complete workflow in 15 minutes? (test with 1B model, 100 examples)
  - SC-002: 7B model trains without OOM? (verify VRAM usage with `Qwen/Qwen2.5-7B-Instruct`)
  - SC-003: Progress updates every batch? (verify log frequency)
  - SC-004: Downloads work without manual intervention? (verify download success rate)
  - SC-005: Training shows measurable improvement? (via `src/eval.py` accuracy metric from T057)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3–8)**: All depend on Foundational phase completion
  - US1 + US2 (P1) can proceed in parallel (different files: model_manager vs dataset_manager)
  - US3 (P1) depends on US1 completion (needs working model for inference)
  - US4 (P2) depends on US1 + US2 completion (needs both model and dataset)
  - US5 (P3) depends on US4 completion (needs fine-tuned model)
  - US6 (P3) depends on US4 completion (needs fine-tuned model)
- **Polish (Final Phase)**: Depends on all user stories being complete

### User Story Dependencies

- **US1 (P1) / US2 (P1)**: Can start in parallel after Foundational phase
- **US3 (P1)**: Can start after US1 (inference depends on model loaded)
- **US4 (P2)**: Can start after US1 + US2 (training needs both model + dataset)
- **US5 (P3)**: Can start after US4 (evaluation needs fine-tuned model)
- **US6 (P3)**: Can start after US4 (comparison needs fine-tuned model)

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Models before services
- Services before UI wiring
- UI wiring before polish
- Story complete before moving to next priority

### Parallel Opportunities

- Setup phase (Phase 1): T005–T009 can run in parallel (different files)
- Foundational phase (Phase 2): T014–T018 can run in parallel (model_manager and dataset_manager are independent)
- US1 + US2: Can run in parallel by different developers
- US5 + US6: Can run in parallel (different UI tabs)

---

## Implementation Strategy

### MVP First (User Story 1 + 2 + 3 only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. Complete Phase 4: User Story 2
5. Complete Phase 5: User Story 3
6. **STOP and VALIDATE**: User can select model, download, download dataset, and run inference
7. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add US1 + US2 → **Test independently → Deploy/Demo** (download + model available)
3. Add US3 → **Test independently → Deploy/Demo** (MVP — inference works)
4. Add US4 → **Test independently → Deploy/Demo** (full training + streaming logs)
5. Add US5 + US6 → **Test independently → Deploy/Demo** (complete demo)
6. Polish → Final verification

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: US1 (model download)
   - Developer B: US2 (dataset download)
   - Developer C: US3 (inference)
3. After P1 done:
   - Developer A: US4 (training pipeline)
   - Developer B: US5 (evaluation)
   - Developer C: US6 (comparison)
4. All polish together

---

## Notes

- **58 total tasks** across 8 phases
- **Tests included**: Yes (unit tests for every module, integration tests for Gradio UI)
- `[P]` tasks = different files, no dependencies
- `[Story]` label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
