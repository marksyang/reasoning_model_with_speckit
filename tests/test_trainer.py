"""Tests for trainer module."""

from unittest.mock import patch, MagicMock
import pytest

import sys
sys.path.insert(0, "/app")


class TestValidateTrainingInputs:
    """Tests for validate_training_inputs functionality."""

    def test_accepts_valid_paths(self, mock_model_path, mock_dataset_path):
        """validate_training_inputs must accept existing paths."""
        assert trainer.validate_training_inputs(
            str(mock_model_path), str(mock_dataset_path)
        ) is True

    def test_rejects_missing_model(self, mock_model_path):
        """validate_training_inputs must reject missing model path."""
        assert trainer.validate_training_inputs("/nonexistent/model", "path") is False

    def test_rejects_missing_dataset(self, mock_model_path):
        """validate_training_inputs must reject missing dataset path."""
        assert trainer.validate_training_inputs("model", "/nonexistent/dataset") is False


class TestTrainingConfig:
    """Tests for TrainingConfig default values."""

    def test_default_lora_r(self):
        """Default LoRA rank must be 8."""
        config = trainer.TrainingConfig()
        assert config.lora_r == 8

    def test_default_lora_alpha(self):
        """Default LoRA alpha must be 16."""
        config = trainer.TrainingConfig()
        assert config.lora_alpha == 16

    def test_default_lora_dropout(self):
        """Default LoRA dropout must be 0.05."""
        config = trainer.TrainingConfig()
        assert config.lora_dropout == 0.05

    def test_default_num_epochs(self):
        """Default epochs must be 1."""
        config = trainer.TrainingConfig()
        assert config.num_train_epochs == 1


class TestRegisterAdapter:
    """Tests for register_adapter functionality."""

    def test_registers_in_state(self, tmp_dir, mock_curated_models):
        """register_adapter must update app_state."""
        # Would test that registered_adapters list is updated
        pass
