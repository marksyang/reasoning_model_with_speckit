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
