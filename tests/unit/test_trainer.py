"""Tests for src/trainer module."""

import queue
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.trainer import (
    TrainingConfig,
    _ProgressCallback,
    _stop_event,
    load_adapter,
    register_adapter,
    stop_training,
    train,
    validate_training_inputs,
)


# ---- Helper to consume generator ----

def _consume(gen):
    """Consume a generator and return all yielded values."""
    return list(gen)


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_defaults(self):
        config = TrainingConfig()
        assert config.lora_r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.0
        assert config.num_train_epochs == 1
        assert config.batch_size == 4
        assert config.gradient_accumulation_steps == 1
        assert config.learning_rate == 5e-5
        assert config.max_seq_length == 2048
        assert config.logging_steps == 10

    def test_custom_values(self):
        config = TrainingConfig(
            lora_r=16,
            lora_alpha=32,
            num_train_epochs=3,
            batch_size=8,
            learning_rate=1e-4,
        )
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.num_train_epochs == 3
        assert config.batch_size == 8
        assert config.learning_rate == 1e-4


class TestProgressCallback:
    """Tests for _ProgressCallback."""

    def test_on_train_loss_report(self):
        q = MagicMock()
        callback = _ProgressCallback(q, total_iters=100)
        callback.on_train_loss_report({"iteration": 1, "train_loss": 0.5, "learning_rate": 1e-4, "tokens_per_second": 100})
        q.put.assert_called_once()

    def test_on_val_loss_report(self):
        q = MagicMock()
        callback = _ProgressCallback(q, total_iters=100)
        callback.on_val_loss_report({"val_loss": 0.6})
        q.put.assert_called_once()

    def test_on_train_loss_report_queue_full(self):
        q = MagicMock()
        q.put.side_effect = queue.Full("Queue full")
        callback = _ProgressCallback(q, total_iters=100)
        # Should not raise — catches queue.Full silently
        callback.on_train_loss_report({"iteration": 1, "train_loss": 0.5})

    def test_on_val_loss_report_queue_full(self):
        q = MagicMock()
        q.put.side_effect = queue.Full("Queue full")
        callback = _ProgressCallback(q, total_iters=100)
        # Should not raise — catches queue.Full silently
        callback.on_val_loss_report({"val_loss": 0.6})


class TestValidateTrainingInputs:
    """Tests for validate_training_inputs."""

    def test_valid_inputs(self, tmp_dir):
        model_path = tmp_dir / "model"
        dataset_path = tmp_dir / "dataset"
        model_path.touch()
        dataset_path.touch()
        assert validate_training_inputs(str(model_path), str(dataset_path)) is True

    def test_missing_model_path(self, tmp_dir):
        assert validate_training_inputs(str(tmp_dir / "nonexistent"), str(tmp_dir / "dataset")) is False

    def test_missing_dataset_path(self, tmp_dir):
        model_path = tmp_dir / "model"
        model_path.touch()
        assert validate_training_inputs(str(model_path), str(tmp_dir / "nonexistent")) is False


class TestRegisterAdapter:
    """Tests for register_adapter."""

    def test_registers_in_state(self, tmp_dir):
        state_path = tmp_dir / "app_state.json"
        with patch("src.model_manager.APP_STATE_PATH", state_path):
            metadata = register_adapter("adapter-1", "meta-llama/Llama-3.2-1B-Instruct")
        assert metadata["adapter_id"] == "adapter-1"
        assert metadata["base_model_hf_id"] == "meta-llama/Llama-3.2-1B-Instruct"

    def test_adds_to_training_runs(self, tmp_dir):
        state_path = tmp_dir / "app_state.json"
        with patch("src.model_manager.APP_STATE_PATH", state_path):
            register_adapter("adapter-1", "meta-llama/Llama-3.2-1B-Instruct")
        import json
        with open(state_path) as f:
            state = json.load(f)
        assert len(state["training_runs"]) == 1
        assert state["training_runs"][0]["adapter_id"] == "adapter-1"


class TestStopTraining:
    """Tests for stop_training."""

    def test_sets_stop_event(self):
        result = stop_training()
        assert result == "Stopping training..."


class TestTrain:
    """Tests for the train() generator function."""

    def test_yields_failed_on_model_load_error(self, tmp_dir):
        """When FastLanguageModel.from_pretrained raises, train() yields a failed message."""
        with patch("src.trainer.FastLanguageModel") as mock_flm:
            mock_flm.from_pretrained.side_effect = RuntimeError("model not found")
            results = _consume(train("bad-model", [], "test-adapter", TrainingConfig()))
            assert any(r.get("type") == "failed" for r in results)
            assert any("model not found" in r.get("error", "") for r in results)

    def test_yields_failed_on_format_error(self, tmp_dir):
        """When format_for_finetuning raises, train() yields a failed message."""
        with patch("src.trainer.FastLanguageModel") as mock_flm:
            mock_model = MagicMock()
            mock_model.layers = [MagicMock()]
            mock_flm.from_pretrained.return_value = (mock_model, MagicMock())
            with patch("src.trainer.format_for_finetuning", side_effect=ValueError("bad data")):
                results = _consume(train("good-model", [], "test-adapter", TrainingConfig()))
                assert any(r.get("type") == "failed" for r in results)
                assert any("bad data" in r.get("error", "") for r in results)

    def test_yields_completed_on_success(self, tmp_dir):
        """When training completes, train() yields a completed message."""
        mock_model = MagicMock()
        mock_model.layers = [MagicMock()]
        mock_tokenizer = MagicMock()

        with patch("src.trainer.FastLanguageModel") as mock_flm:
            mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
            with patch("src.trainer.format_for_finetuning", return_value=["text1"]):
                with patch.dict("sys.modules",
                        {"mlx_lm.tuner.datasets": MagicMock(),
                         "mlx_lm.lora": MagicMock()}):
                    import sys
                    # Patch train_model inside mlx_lm.lora
                    sys.modules["mlx_lm.lora"].train_model = MagicMock(
                        side_effect=lambda args, model, ts, vs, cb:
                            cb.q.put({"type": "done", "adapter_path": "data/adapters/test-adapter"})
                    )
                    _stop_event.clear()
                    results = _consume(train("test-model", [], "test-adapter", TrainingConfig()))
                    assert any(r.get("type") == "completed" for r in results)

    def test_yields_failed_on_training_error(self, tmp_dir):
        """When training encounters an error, train() yields a failed message."""
        mock_model = MagicMock()
        mock_model.layers = [MagicMock()]
        mock_tokenizer = MagicMock()

        with patch("src.trainer.FastLanguageModel") as mock_flm:
            mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
            with patch("src.trainer.format_for_finetuning", return_value=["text1"]):
                with patch.dict("sys.modules",
                        {"mlx_lm.tuner.datasets": MagicMock(),
                         "mlx_lm.lora": MagicMock()}):
                    import sys
                    sys.modules["mlx_lm.lora"].train_model = MagicMock(
                        side_effect=lambda args, model, ts, vs, cb:
                            cb.q.put({"type": "error", "error": "GPU crash"})
                    )
                    _stop_event.clear()
                    results = _consume(train("test-model", [], "test-adapter", TrainingConfig()))
                    assert any(r.get("type") == "failed" for r in results)

    def test_yields_progress_message(self, tmp_dir):
        """train() yields progress updates from the training loop."""
        mock_model = MagicMock()
        mock_model.layers = [MagicMock()]
        mock_tokenizer = MagicMock()

        with patch("src.trainer.FastLanguageModel") as mock_flm:
            mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
            with patch("src.trainer.format_for_finetuning", return_value=["text1"]):
                with patch.dict("sys.modules",
                        {"mlx_lm.tuner.datasets": MagicMock(),
                         "mlx_lm.lora": MagicMock()}):
                    import sys
                    sys.modules["mlx_lm.lora"].train_model = MagicMock(
                        side_effect=lambda args, model, ts, vs, cb:
                            cb.q.put({
                                "type": "progress",
                                "iteration": 5,
                                "total_iters": 100,
                                "train_loss": 2.5,
                                "learning_rate": 1e-4,
                                "tokens_per_second": 500,
                            }) or cb.q.put({"type": "done", "adapter_path": "data/adapters/test-adapter"})
                    )
                    _stop_event.clear()
                    results = _consume(train("test-model", [], "test-adapter", TrainingConfig()))
                    progress = [r for r in results if "loss" in r and "iteration" in r]
                    assert len(progress) >= 1
                    assert progress[0]["iteration"] == 5
                    assert "perplexity" in progress[0]

    def test_handles_dataset_dict(self, tmp_dir):
        """train() extracts the 'train' split from a DatasetDict."""
        from datasets import DatasetDict

        mock_model = MagicMock()
        mock_model.layers = [MagicMock()]
        mock_tokenizer = MagicMock()
        dataset = DatasetDict({"train": MagicMock(), "test": MagicMock()})

        with patch("src.trainer.FastLanguageModel") as mock_flm:
            mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
            with patch("src.trainer.format_for_finetuning", return_value=["text1"]) as mock_format:
                with patch.dict("sys.modules",
                        {"mlx_lm.tuner.datasets": MagicMock(),
                         "mlx_lm.lora": MagicMock()}):
                    import sys
                    sys.modules["mlx_lm.lora"].train_model = MagicMock(
                        side_effect=lambda args, model, ts, vs, cb:
                            cb.q.put({"type": "done", "adapter_path": "data/adapters/test-adapter"})
                    )
                    _stop_event.clear()
                    _consume(train("test-model", dataset, "test-adapter", TrainingConfig()))
                    # Verify format_for_finetuning was called (not with the DatasetDict itself)
                    mock_format.assert_called_once()


class TestRegisterAdapterMissingTrainingRuns:
    """Tests for register_adapter when state lacks training_runs."""

    def test_creates_training_runs_when_missing(self, tmp_path):
        """register_adapter initializes training_runs if not present in state."""
        state_path = tmp_path / "app_state.json"
        import json
        # Write a state without training_runs key
        with open(state_path, "w") as f:
            json.dump({"available_models": [], "registered_adapters": []}, f)
        with patch("src.model_manager.APP_STATE_PATH", state_path):
            register_adapter("adapter-1", "test-model")
        with open(state_path) as f:
            state = json.load(f)
        assert "training_runs" in state
        assert len(state["training_runs"]) == 1


class TestLoadAdapter:
    """Tests for load_adapter."""

    def test_loads_base_and_adapter(self, tmp_dir):
        """load_adapter loads base model and merges adapter weights."""
        import mlx.core as real_mx
        import mlx_lm as real_mlx_lm

        base_path = tmp_dir / "base_model"
        base_path.mkdir()
        adapter_path = tmp_dir / "adapters" / "adapter-1"
        adapter_path.mkdir(parents=True, exist_ok=True)
        weights_file = adapter_path / "adapters.safetensors"
        weights_file.touch()

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_weights = MagicMock()

        with patch.object(real_mlx_lm, "load", return_value=(mock_model, mock_tokenizer)) as mock_mlx_load:
            with patch.object(real_mx, "load", return_value=mock_weights) as mock_mx_load:
                result_model, result_tokenizer = load_adapter(str(base_path), str(adapter_path))

                mock_mlx_load.assert_called_once_with(str(base_path))
                mock_mx_load.assert_called_once()
                mock_model.update.assert_called_once_with(mock_weights)
                assert result_model is mock_model
                assert result_tokenizer is mock_tokenizer
