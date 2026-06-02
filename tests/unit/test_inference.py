"""Tests for src/inference module."""

from unittest.mock import MagicMock, patch

import pytest

from src.inference import base_inference, finetuned_inference


class _MockInputs:
    """Mock tokenizer output that supports .to() and dict-like access."""

    def __init__(self):
        self.data = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }

    def to(self, device):
        return self

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    def get(self, key, default=None):
        return self.data.get(key, default)

    def keys(self):
        return self.data.keys()


def _build_mock_model_and_tokenizer():
    """Build shared mocks for inference tests."""
    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_generated = MagicMock()
    mock_tokens = MagicMock()
    mock_tokens.shape = MagicMock()
    mock_tokens.__getitem__ = MagicMock(return_value=range(10))
    mock_generated.__getitem__ = MagicMock(return_value=mock_tokens)
    mock_model.generate.return_value = mock_generated

    mock_inputs = _MockInputs()

    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "test output"
    mock_tokenizer.apply_chat_template.return_value = "test input"
    mock_tokenizer.return_value = mock_inputs
    mock_tokenizer.eos_token_id = 2

    return mock_model, mock_tokenizer


class TestBaseInference:
    """Tests for base_inference."""

    def test_returns_generator(self):
        mock_model, mock_tokenizer = _build_mock_model_and_tokenizer()

        with patch("src.inference.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
            with patch("src.inference.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
                with patch("src.inference.PeftModel.from_pretrained", side_effect=FileNotFoundError):
                    results = list(base_inference("test/model", "Hello", {}))
        assert len(results) >= 1

    def test_uses_default_generation_kwargs(self):
        mock_model, mock_tokenizer = _build_mock_model_and_tokenizer()

        with patch("src.inference.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
            with patch("src.inference.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
                with patch("src.inference.PeftModel.from_pretrained", side_effect=FileNotFoundError):
                    list(base_inference("test/model", "Hello", {}))
        # Verify default kwargs were set
        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs.get("max_new_tokens") == 512
        assert call_kwargs.get("temperature") == 0.7
        assert call_kwargs.get("top_p") == 0.9
        assert call_kwargs.get("do_sample") is True


class TestFinetunedInference:
    """Tests for finetuned_inference."""

    def test_returns_generator(self):
        mock_model, mock_tokenizer = _build_mock_model_and_tokenizer()

        with patch("src.inference.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
            with patch("src.inference.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
                with patch("src.inference.PeftModel.from_pretrained", return_value=mock_model):
                    results = list(finetuned_inference("base/model", "adapter/path", "Hello", {}))
        assert len(results) >= 1

    def test_loads_adapter(self):
        mock_model, mock_tokenizer = _build_mock_model_and_tokenizer()

        with patch("src.inference.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
            with patch("src.inference.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
                with patch("src.inference.PeftModel.from_pretrained", return_value=mock_model) as mock_peft:
                    list(finetuned_inference("base/model", "adapter/path", "Hello", {}))
        mock_peft.assert_called_once()
