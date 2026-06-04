"""Tests for src/inference module."""

from unittest.mock import MagicMock, patch

import pytest

from src.inference import base_inference, finetuned_inference


class TestBaseInference:
    """Tests for base_inference."""

    def test_returns_generator(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "test input"

        with patch("src.inference.load", return_value=(mock_model, mock_tokenizer)):
            with patch("src.inference.generate", return_value="test output"):
                results = list(base_inference("test/model", "Hello", {}))
        assert len(results) >= 1
        assert results[0] == "test output"

    def test_uses_default_generation_kwargs(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "test input"

        with patch("src.inference.load", return_value=(mock_model, mock_tokenizer)):
            with patch("src.inference.generate") as mock_generate:
                with patch("src.inference.make_sampler") as mock_sampler:
                    list(base_inference("test/model", "Hello", {}))

        # Verify generate was called with correct max_tokens
        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs.get("max_tokens") == 512


class TestFinetunedInference:
    """Tests for finetuned_inference."""

    def test_returns_generator(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "test input"

        with patch("src.inference.load", return_value=(mock_model, mock_tokenizer)):
            with patch("src.inference.generate", return_value="test output"):
                results = list(finetuned_inference("base/model", "adapter/path", "Hello", {}))
        assert len(results) >= 1
        assert results[0] == "test output"

    def test_loads_adapter(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "test input"

        with patch("src.inference.load", return_value=(mock_model, mock_tokenizer)) as mock_load:
            with patch("src.inference.generate"):
                list(finetuned_inference("base/model", "adapter/path", "Hello", {}))

        # Verify load was called with adapter_path
        mock_load.assert_called_once_with("base/model", adapter_path="adapter/path")
