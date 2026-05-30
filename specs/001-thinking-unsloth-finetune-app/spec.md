# Feature Specification: Thinking Model Fine-Tuning Demo App

**Feature Branch**: `[001-thinking-unsloth-finetune-app]`

**Created**: 2026-05-30

**Status**: Draft

**Input**: User description: Web application for demonstrating unsloth thinking model fine-tuning on a 12GB GPU. The app should help users understand how fine-tuning improves reasoning models and enable complete experimentation (from data download to inference comparison) in a single session. Users need to be able to choose base model and dataset with automatic download, perform inference before and after fine-tuning, and observe real-time training progress.

## Clarifications

### Session 2026-05-30

- Q: What is the training target type? → A: Mathematics reasoning (GSM8K) — step-by-step word problem solutions
- Q: What LoRA parameters should be used? → A: Default r=8, alpha=16, dropout=0.05 with user-overridable fields
- Q: How to verify training improvement? → A: Interactive evaluation — system presents 5 validation examples, user runs inference on both models, tracks accuracy
- Q: What training log granularity? → A: Epoch-level summaries plus batch-level loss values (best balance of real-time feel and UI density)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Select and Download Base Model (Priority: P1)

A data scientist wants to experiment with fine-tuning a thinking/reasoning model on their local machine with only a 12GB GPU. They navigate to a simple web interface, pick a pre-approved base model from a list of GPU-appropriate options, and start downloading without touching the command line or managing file paths. The download shows progress and completes automatically.

**Why this priority**: Users cannot proceed to inference or fine-tuning without a base model. This is the first step in the workflow and is essential for any demo to work.

**Independent Test**: The user can select a model from the dropdown, click download, and the model becomes available for inference without fine-tuning.

**Acceptance Scenarios**:

1. **Given** the app has loaded with a list of 1B and 7B parameter thinking models suitable for 12GB VRAM, **When** the user selects one and clicks "Download," **Then** the system displays a progress indicator and automatically caches the model locally.
2. **Given** a model is successfully downloaded and cached, **When** the user selects another model, **Then** the previous model remains cached and selectable for future inference.
3. **Given** the user restarts the app, **When** they open the model dropdown, **Then** previously downloaded models appear in the list as available.

---

### User Story 2 - Select and Download Dataset (Priority: P1)

A data scientist wants to fine-tune their model on reasoning data without spending time searching for and formatting datasets. They open a dropdown of pre-approved public datasets, select one, and the system downloads and caches it. The interface shows how many examples are available in each split.

**Why this priority**: The model needs training data to fine-tune. Without data, fine-tuning cannot proceed.

**Independent Test**: The user can select a dataset, download it, and see summary statistics (split sizes, column names).

**Acceptance Scenarios**:

1. **Given** the app has loaded a curated list of reasoning datasets (math, logic, instruction-following), **When** the user selects one and clicks "Download," **Then** the dataset downloads and a summary appears showing sample count and structure.
2. **Given** a dataset is fully downloaded, **When** the user navigates away and returns, **Then** the dataset appears in the dropdown as already available (no re-download).
3. **Given** a dataset download fails, **When** the user retries, **Then** the system resumes from where it left off rather than starting over.

---

### User Story 3 - Infer Using Base Model (Priority: P1)

A data scientist has downloaded a base model and wants to see how it reasons before fine-tuning. They type a reasoning prompt (math word problem, logic puzzle, or complex instruction), review generation parameters, and click "Run Inference." The model's response appears in the output area.

**Why this priority**: Without seeing the base model's output, users cannot compare it against the fine-tuned version. This establishes the baseline for all later improvement comparisons.

**Independent Test**: The user enters a prompt, runs inference, and receives a valid model response displayed on screen.

**Acceptance Scenarios**:

1. **Given** a base model is downloaded and cached, **When** the user enters a prompt and clicks "Run Inference," **Then** the model generates a response that appears in the output panel.
2. **Given** the user is generating inference output, **When** they want to adjust parameters, **Then** they can change them before the next inference run.
3. **Given** inference is running, **When** the model generates output, **Then** results appear progressively so the user can see what the model is thinking in real time.

---

### User Story 4 - Fine-Tune Model (Priority: P2)

A data scientist wants to improve their model on reasoning tasks by fine-tuning with QLoRA and unsloth. They select a dataset and review optional fine-tuning parameters. They click "Start Training" and the app shows a progress bar with streaming logs. Training completes and the fine-tuned model automatically registers in the app's model selector.

**Why this priority**: This is the core value proposition. Users want to see that fine-tuning actually improves model reasoning capabilities. The entire demo hinges on this step working.

**Independent Test**: The user can initiate training, observe progress, and obtain a fine-tuned model that can run inference.

**Acceptance Scenarios**:

1. **Given** a base model and dataset are available, **When** the user clicks "Start Training," **Then** training begins with a progress bar and live log output.
2. **Given** training is running, **When** the user views the log panel, **Then** they see continuous scrolling updates (loss, metrics) streamed in real time.
3. **Given** training completes successfully, **When** it finishes, **Then** the fine-tuned model appears in the model selector dropdown as a new option.
4. **Given** training encounters an error, **When** it fails, **Then** the UI displays a clear error message and the error logs remain visible for review.
5. **Given** training is running, **When** the user clicks "Terminate," **Then** the system displays a confirmation warning and stops training safely.

---

### User Story 5 - Evaluate Fine-Tuned Model (Priority: P3)

A data scientist has finished fine-tuning and wants to verify the model improved on specific examples. The system presents 5 validation examples from the dataset. For each example, the user runs inference on both the base model and fine-tuned model, and tracks whether the fine-tuned model answered correctly. A summary screen shows accuracy percentages for both models.

**Why this priority**: This validates the entire experiment with measurable evidence. Without structured comparison on the same examples, users cannot confirm fine-tuning was effective.

**Independent Test**: The user selects the fine-tuned model, runs the system's 5 validation examples, and sees a summary of per-question and overall accuracy.

**Acceptance Scenarios**:

1. **Given** a fine-tuned model has been registered from a completed training run, **When** the user clicks "Evaluate," **Then** the system displays 5 validation examples one at a time.
2. **Given** a validation example is shown, **When** the user runs inference on both models, **Then** both responses appear side by side.
3. **Given** all 5 examples are evaluated, **When** the user finishes, **Then** a summary screen shows per-question accuracy and total accuracy for both models.

---

### User Story 6 - Compare Fine-Tuned Inference (Priority: P3)

A data scientist has finished evaluation and wants to explore the fine-tuned model's behavior on their own custom prompts. They enter custom prompts and compare the fine-tuned model's responses against the base model's responses.

**Why this priority**: Allows deeper exploration of model improvements beyond the structured evaluation set.

**Independent Test**: The user enters a custom prompt, runs inference on both models, and observes the output.

**Acceptance Scenarios**:

1. **Given** a fine-tuned model has been registered from a completed training run, **When** the user selects it in the dropdown, **Then** the "Run Inference" button becomes active.
2. **Given** the user selects a fine-tuned model and runs inference, **When** the output appears, **Then** it displays the model's reasoning response for the entered prompt.
3. **Given** the user has both base model and fine-tuned model outputs, **When** they compare the two, **Then** the difference in response quality is visible.

---

### Edge Cases

- What happens when the 12GB GPU runs out of memory during inference?
- How does the system handle interrupted downloads for models or datasets?
- What does the user see if training runs out of memory or fails partway through?
- How are partial training checkpoints handled if the system crashes?
- What occurs when the user switches between models mid-inference?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Users MUST be able to browse a curated list of pre-selected thinking/reasoning models from Hugging Face Hub
- **FR-002**: Users MUST be able to select a base model from the curated list and initiate an automated download
- **FR-003**: The app MUST display a real-time progress indicator during downloads
- **FR-004**: Downloaded models MUST be automatically cached and persist across app restarts
- **FR-005**: Users MUST be able to browse a curated list of pre-selected public reasoning datasets from Hugging Face Hub
- **FR-006**: Users MUST be able to select a dataset and initiate an automated download
- **FR-007**: The app MUST display summary statistics (sample count, splits, columns) after downloading a dataset
- **FR-008**: Users MUST be able to enter text prompts for model inference
- **FR-009**: Users MUST be able to run inference using the currently selected base model
- **FR-010**: Users MUST be able to run inference using a fine-tuned model once it is available
- **FR-011**: Inference output MUST appear progressively so users can see generation in real time
- **FR-012**: Users MUST be able to initiate model fine-tuning with a base model and a dataset; default LoRA parameters MUST be configurable before training; default values: r=8, alpha=16, dropout=0.05
- **FR-013**: Fine-tuning MUST use QLoRA (4-bit quantization) with unsloth to fit within 12GB GPU memory
- **FR-014**: Training MUST display a real-time progress bar
- **FR-015**: Training MUST stream live logs: epoch-level summaries (learning rate, time per epoch, metrics) plus batch-level loss values to keep users aware of training progress
- **FR-016**: Upon successful training completion, the fine-tuned model MUST automatically register in the model selector
- **FR-017**: The UI MUST visually reflect the fine-tuned model's availability (e.g., button state, dropdown options)
- **FR-018**: Training MUST be safely interruptible with a confirmation warning
- **FR-019**: Training failures MUST display a clear error message with diagnostic logs
- **FR-020**: Downloads and model selection MUST be locked during active training to prevent conflicts
- **FR-021**: Users MUST be able to compare base model inference results with fine-tuned model inference results for the same prompt; when simultaneously loading both models exceeds GPU memory (typical on 12GB GPUs with >3B models), the app should unload the base model before loading the fine-tuned model
- **FR-022**: After training completes, the user MUST be able to evaluate the fine-tuned model on a set of 5 validation examples, with per-question accuracy tracking

### Key Entities

- **Model**: Represents a base or fine-tuned model from Hugging Face. Attributes include Hugging Face repo ID, parameter count, download status, cache location, and whether it has been fine-tuned.
- **Dataset**: Represents a reasoning dataset from Hugging Face. Attributes include Hugging Face repo ID, training split size, validation split size, column names, and whether training data has been formatted.
- **Training Run**: Represents a single fine-tuning execution. Attributes include model and dataset associations, LoRA parameters (default r=8, alpha=16, dropout=0.05, all editable), training metrics, status (pending, running, completed, failed), and the resulting fine-tuned model output.
- **Inference Result**: Represents a model response to a prompt. Attributes include the model used, the prompt, the generated response, and generation parameters.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can complete the entire workflow (select model → select dataset → run base inference → fine-tune → run fine-tuned inference) within 15 minutes for a 1B parameter model with 100 training examples
- **SC-002**: The system successfully trains on a 7B parameter model with 100 training examples on a single 12GB GPU without running out of memory
- **SC-003**: Training output (progress bar and logs) updates at least every batch (typically 1-5 seconds depending on batch size), with epoch summaries displayed after each epoch
- **SC-004**: A user completing the demo successfully downloads both a model and a dataset without manual intervention in 90% of attempts
- **SC-005**: After fine-tuning, the fine-tuned model demonstrates measurable reasoning accuracy improvement compared to the base model on a small test set (at least 5 examples)

## Assumptions

- Users have a single NVIDIA GPU with at least 12GB of VRAM and are running Ubuntu Linux
- Python 3.10+ is available with access to pip/uv for package installation
- The user has a stable internet connection capable of downloading multi-gigabyte model weights and datasets
- Pre-approved reasoning datasets are drawn from Hugging Face Hub and available for public download
- Base models are restricted to parameter counts suitable for 12GB VRAM (e.g., 1B to 7B parameters with 4-bit quantization)
- QLoRA with 4-bit quantization is the standard fine-tuning approach (no full fine-tuning available)
- The fine-tuning target is mathematics reasoning (GSM8K dataset): converting natural-language word problems into Chain-of-Thought formatted examples, each consisting of a "question" and "answer" field containing both the problem and step-by-step solution
- Training examples will typically be small datasets (10 to 1,000 examples) for rapid demonstration
- The UI will use Gradio Blocks, which supports event handling and progress streaming natively
