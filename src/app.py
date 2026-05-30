"""Thinking Model Fine-Tuning Demo App — Gradio UI."""

import json
import logging
import os
import threading
import time
from pathlib import Path

import gradio as gr

from .model_manager import (
    CURATED_MODELS,
    list_models,
    download_model,
    get_model_info,
    register_model,
    get_model_cache_path,
)
from .dataset_manager import (
    CURATED_DATASETS,
    list_datasets,
    download_dataset,
    get_dataset_summary,
    format_for_finetuning,
)
from .trainer import (
    validate_training_inputs,
    load_base_model,
    train,
    TrainingConfig,
    register_adapter as register_trained_adapter,
)
from .inference import base_inference, finetuned_inference

logger = logging.getLogger(__name__)

# Disable HF warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


# ---- App state ----

app_state = {
    "available_models": [],
    "available_datasets": [],
    "registered_adapters": [],
    "selected_model": None,
    "selected_dataset": None,
    "training_runs": [],
}


def init_app():
    """Initialize the app state."""
    global app_state
    if Path("data/app_state.json").exists():
        with open("data/app_state.json") as f:
            app_state = json.load(f)
    else:
        with open("data/app_state.json", "w") as f:
            json.dump(app_state, f, indent=2)


init_app()


def get_cached_models():
    """Get list of cached models."""
    return list_models()


def get_available_datasets():
    """Get list of available datasets."""
    return list_datasets()


# ---- Handlers ----

def on_model_select(model_hf_id):
    """Handle model selection."""
    if model_hf_id:
        try:
            if model_hf_id in CURATED_MODELS:
                info = CURATED_MODELS[model_hf_id].copy()
                info["hf_id"] = model_hf_id
                display = json.dumps(info, indent=2, default=str)
                return display
        except Exception:
            return "No additional info available."
    else:
        return "Select a model to see details."


def on_dataset_select(dataset_hf_id):
    """Handle dataset selection."""
    if not dataset_hf_id:
        return "Select a dataset first."
    try:
        summary = get_dataset_summary(dataset_hf_id)
        return json.dumps(summary, indent=2, default=str)
    except Exception as e:
        return f"Error loading dataset: {e}"


def on_model_download(model_hf_id):
    """Download a model."""
    if not model_hf_id:
        raise gr.Error("Please select a model to download.")

    # Check if already cached
    import os
    cache_pattern = f"models--{model_hf_id.replace('/', '--')}"
    cache_path = os.path.expanduser("~/.cache/huggingface/hub")
    cached_dirs = [d for d in os.listdir(cache_path) if d.startswith(cache_pattern) and os.path.isdir(os.path.join(cache_path, d))] if os.path.exists(cache_path) else []

    if cached_dirs:
        register_model(model_hf_id)
        return f"Model {model_hf_id} is already cached locally. Ready for inference!"

    # Try downloading
    try:
        result = download_model(model_hf_id)
        app_state["selected_model"] = model_hf_id
        register_model(model_hf_id)
        return f"Model {model_hf_id} downloaded successfully.\nCache path: {result}\nYou can now use it for inference."
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        msg = f"Download failed: {e}"
        if "gated" in str(e).lower() or "authentication" in str(e).lower():
            msg += "\n\nThis model requires Hub approval. Visit https://huggingface.co/" + model_hf_id + " and accept the terms."
        return msg


def on_dataset_download(dataset_hf_id):
    """Download a dataset."""
    if not dataset_hf_id:
        raise gr.Error("Please select a dataset to download.")

    # Check if already downloaded
    try:
        from datasets import load_dataset
        load_dataset(dataset_hf_id, download_mode="reuse_cache_if_exists")
        app_state["selected_dataset"] = dataset_hf_id
        return f"Dataset {dataset_hf_id} is already cached locally.\nSelect it in Training tab to start fine-tuning."
    except Exception:
        pass

    try:
        result = download_dataset(dataset_hf_id)
        app_state["selected_dataset"] = dataset_hf_id
        return f"Dataset {dataset_hf_id} downloaded successfully."
    except Exception as e:
        raise gr.Error(f"Download failed: {e}")


def on_run_inference(model_hf_id, prompt, max_tokens, temperature, top_p, do_sample):
    """Run inference on the selected model."""
    if not model_hf_id:
        raise gr.Error("Please select a model first.")
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    generation_kwargs = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample,
    }

    try:
        cache_path = get_model_cache_path()
        # Find the actual model path
        model_path = None
        if model_hf_id in CURATED_MODELS:
            # Build model path from HF cache
            model_path = str(cache_path / f"models--{model_hf_id.replace('/', '--')}--blobs"
                            for blob_path in cache_path.glob(f"models--{model_hf_id.replace('/', '--')}--blobs/*"))

        # Try loading from cache directly
        model_path = f"{cache_path}/models--{model_hf_id.replace('/', '--')}--blobs"

        # If we can't find it, try the hf_id directly as a hint
        model_path = model_hf_id

        if model_hf_id and prompt.strip():
            # Check if it's a fine-tuned model (registered adapter)
            is_finetuned = model_hf_id in app_state.get("registered_adapters", [])

            if is_finetuned:
                # Find adapter path
                adapter_path = None
                for run in app_state.get("training_runs", []):
                    if run.get("adapter_id") == model_hf_id:
                        adapter_path = run.get("adapter_path")
                        break

                if adapter_path:
                    base_model = CURATED_MODELS.get(
                        next((k for k, v in CURATED_MODELS.items()
                              if v.get("adapter_id") == model_hf_id), None),
                        {}).get("hf_id", "")

                    if base_model:
                        result = finetuned_inference(base_model, adapter_path, prompt, generation_kwargs)
                        return "".join(result)

        # Default: use HF as model path hint
        from inference import base_inference
        result = base_inference(model_hf_id, prompt, generation_kwargs)
        return "".join(result)

    except Exception as e:
        raise gr.Error(f"Inference failed: {e}")


def on_start_training(model_hf_id, dataset_hf_id, epochs, learning_rate,
                      lora_r, lora_alpha, lora_dropout):
    """Start fine-tuning with streaming progress."""
    if not model_hf_id or not dataset_hf_id:
        raise gr.Error("Please select both a model and dataset to train.")

    adapter_id = f"adapter_{int(time.time())}"
    progress_log = ""

    def training_generator():
        config = TrainingConfig(
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            lora_r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
        )

        try:
            from dataset_manager import load_dataset
            dataset = load_dataset(dataset_hf_id)

            for chunk in train(model_hf_id, dataset, adapter_id, config):
                if chunk.get("type") == "completed":
                    # Register the adapter
                    register_trained_adapter(adapter_id, model_hf_id)
                    yield f"✅ Training complete!\nAdapter saved to: {chunk['adapter_path']}"
                    return
                elif chunk.get("type") == "failed":
                    yield f"❌ Training failed: {chunk.get('error', 'Unknown error')}"
                    return

                log_entry = (
                    f"Epoch {chunk.get('epoch', '?')}: "
                    f"loss={chunk.get('loss', '?'):.4f}, "
                    f"ppl={chunk.get('perplexity', '?'):.2f}, "
                    f"lr={chunk.get('lr', '?'):.2e}"
                )
                yield log_entry + "\n"

        except Exception as e:
            yield f"❌ Training failed: {e}"

    return gr.Progress(generator=training_generator)


def on_evaluate(model_hf_id, prompts, answers):
    """Evaluate the fine-tuned model on validation examples."""
    if not model_hf_id:
        raise gr.Error("Please select a model to evaluate.")

    results = []
    for prompt, answer in zip(prompts, answers):
        try:
            result = on_run_inference(model_hf_id, prompt, 512, 0.7, 0.9, True)
            results.append({"prompt": prompt, "answer": answer, "result": result})
        except Exception as e:
            results.append({"prompt": prompt, "answer": answer, "error": str(e)})

    return json.dumps(results, indent=2)


def on_compute_accuracy(prompts, answers, predictions):
    """Compute accuracy for evaluation."""
    correct = 0
    total = len(prompts)
    details = []

    for i, (prompt, answer, prediction) in enumerate(zip(prompts, answers, predictions)):
        if i < len(prompts):
            p = prediction.get("result", "") if isinstance(prediction, dict) else str(prediction)
            a = answer.get("answer", "") if isinstance(answer, dict) else str(answer)
            # Simple exact match for now (real implementation would parse reasoning)
            is_correct = p.strip() == a.strip()
            if is_correct:
                correct += 1
            details.append({"prompt": prompts[i], "expected": a, "got": p, "correct": is_correct})

    accuracy = correct / total if total > 0 else 0
    return {
        "correct": correct,
        "total": total,
        "accuracy": f"{accuracy * 100:.1f}%",
        "details": json.dumps(details, indent=2),
    }


# ---- UI Setup ----

css = """
.gradio-container { max-width: 100% !important; padding: 20px !important; }
.progress-bar { height: 8px !important; background: linear-gradient(90deg, #4285f4, #34a853) !important; }
"""

with gr.Blocks() as app:
    gr.Markdown("# 🚀 Thinking Model Fine-Tuning Demo")

    with gr.Tabs():
        # Tab 1: Preparation
        with gr.Tab("Preparation"):
            gr.Markdown("## Download Models & Datasets")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Model")
                    model_dropdown = gr.Dropdown(
                        choices=list(CURATED_MODELS.keys()),
                        label="Select Model",
                        value=None,
                    )
                    download_model_btn = gr.Button("Download Model", variant="primary")
                                download_model_result = gr.Textbox(label="Model Status", lines=8, interactive=False)
                    model_info = gr.Textbox(label="Model Details", interactive=False)

                with gr.Column(scale=1):
                    gr.Markdown("### Dataset")
                    dataset_dropdown = gr.Dropdown(
                        choices=list(CURATED_DATASETS.keys()),
                        label="Select Dataset",
                        value=None,
                    )
                    download_dataset_btn = gr.Button("Download Dataset", variant="primary")
                    download_dataset_result = gr.Textbox(label="Dataset Status", lines=3)
                    dataset_summary = gr.Textbox(label="Dataset Summary", interactive=False)

        # Tab 2: Inference
        with gr.Tab("Inference"):
            gr.Markdown("## Run Inference on Selected Model")

            model_dropdown2 = gr.Dropdown(
                choices=list(CURATED_MODELS.keys()),
                label="Select Model",
                value=None,
            )

            with gr.Row():
                prompt_textbox = gr.Textbox(
                    lines=3,
                    label="Prompt",
                    placeholder="Enter a reasoning prompt (e.g., a math word problem)...",
                )

            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=64, maximum=1024, value=512, step=64,
                    label="Max New Tokens",
                )
                temperature = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.7, step=0.1,
                    label="Temperature",
                )
                top_p = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.9, step=0.1,
                    label="Top-p",
                )
                do_sample = gr.Checkbox(label="Sample", value=True)

            run_btn = gr.Button("Run Inference", variant="primary")
            inference_output = gr.Textbox(
                lines=20, label="Output", interactive=False,
            )

        # Tab 3: Training
        with gr.Tab("Training"):
            gr.Markdown("## Fine-Tune Model with Unsloth + QLoRA")

            with gr.Row():
                model_dropdown3 = gr.Dropdown(
                    choices=list(CURATED_MODELS.keys()),
                    label="Select Model",
                    value=None,
                )
                dataset_dropdown3 = gr.Dropdown(
                    choices=list(CURATED_DATASETS.keys()),
                    label="Select Dataset",
                    value=None,
                )

            with gr.Row():
                epochs = gr.Slider(
                    minimum=1, maximum=10, value=1, step=1,
                    label="Epochs",
                )
                learning_rate = gr.Slider(
                    minimum=1e-5, maximum=1e-3, value=2e-4, step=1e-5,
                    label="Learning Rate",
                )
                lora_r = gr.Slider(
                    minimum=4, maximum=32, value=8, step=4,
                    label="LoRA Rank",
                )
                lora_alpha = gr.Slider(
                    minimum=4, maximum=64, value=16, step=4,
                    label="LoRA Alpha",
                )
                lora_dropout = gr.Slider(
                    minimum=0.01, maximum=0.1, value=0.05, step=0.01,
                    label="LoRA Dropout",
                )

            train_btn = gr.Button("Start Training", variant="primary")
            training_progress = gr.Textbox(
                lines=15, label="Training Progress", interactive=False,
            )
            training_log = gr.Textbox(
                lines=10, label="Training Log", interactive=False,
            )
            registered_models_dropdown = gr.Dropdown(
                choices=[],
                label="Registered Fine-Tuned Models",
                value=None,
            )
            evaluate_btn = gr.Button("Evaluate", variant="secondary", interactive=False)

        # Tab 4: Evaluate
        with gr.Tab("Evaluate"):
            gr.Markdown("## Evaluate Fine-Tuned Model")

            evaluate_model_dropdown = gr.Dropdown(
                choices=list(CURATED_MODELS.keys()),
                label="Select Fine-Tuned Model",
                value=None,
            )

            gr.Markdown("### Sample Prompts")
            eval_prompt1 = gr.Textbox(label="Prompt 1", value="If I have 5 apples and give 2 to my friend, how many do I have left?")
            eval_answer1 = gr.Textbox(label="Expected Answer 1", value="3")

            eval_prompt2 = gr.Textbox(label="Prompt 2", value="What is the sum of the first 10 positive integers?")
            eval_answer2 = gr.Textbox(label="Expected Answer 2", value="55")

            eval_prompt3 = gr.Textbox(label="Prompt 3", value="If a train travels at 60 mph for 2 hours, how far does it go?")
            eval_answer3 = gr.Textbox(label="Expected Answer 3", value="120 miles")

            eval_run_btn = gr.Button("Run Evaluation", variant="primary")
            eval_result = gr.JSON(label="Evaluation Results")
            eval_accuracy = gr.Textbox(label="Accuracy", lines=3)


# ---- Event handlers ----

def setup_events():
    """Wire up all events."""

    # Preparation tab
    model_dropdown.select(on_model_select, model_dropdown, model_info)
    download_model_btn.click(on_model_download, model_dropdown, download_model_result)
    dataset_dropdown.select(on_dataset_select, dataset_dropdown, dataset_summary)
    download_dataset_btn.click(on_dataset_download, dataset_dropdown, download_dataset_result)

    # Inference tab
    run_btn.click(
        on_run_inference,
        [model_dropdown2, prompt_textbox, max_tokens, temperature, top_p, do_sample],
        inference_output,
    )

    # Training tab
    train_btn.click(
        on_start_training,
        [model_dropdown3, dataset_dropdown3, epochs, learning_rate, lora_r, lora_alpha, lora_dropout],
        training_progress,
    )

    # Evaluate tab
    eval_run_btn.click(
        on_compute_accuracy,
        [eval_prompt1, eval_answer1, eval_prompt2, eval_answer2, eval_prompt3, eval_answer3, eval_result],
        eval_accuracy,
    )


    setup_events()

if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="blue"),
        css=css,
    )
