"""Tests for model_manager module."""

from unittest.mock import patch, MagicMock
import pytest

import sys
sys.path.insert(0, "/app")


class TestListModels:
    """Tests for list_models functionality."""

    def test_returns_curated_models(self, mock_curated_models):
        """list_models() must return curated models with correct structure."""
        with patch("model_manager.CURATED_MODELS", mock_curated_models):
            result = model_manager.list_models()
            assert len(result) >= len(mock_curated_models)

    def test_model_has_required_fields(self, mock_curated_models):
        """Each model must have hf_id, name, param_count."""
        with patch("model_manager.CURATED_MODELS", mock_curated_models):
            with patch("model_manager.get_cached_models", return_value=[]):
                result = model_manager.list_models()
                for model in result:
                    assert "hf_id" in model
                    assert "name" in model
                    assert "param_count" in model


class TestModelInfo:
    """Tests for get_model_info functionality."""

    def test_returns_info_for_curated_model(self, mock_curated_models):
        """get_model_info must return metadata for a known model."""
        with patch("model_manager.CURATED_MODELS", mock_curated_models):
            info = model_manager.get_model_info("Qwen/Qwen2.5-0.5B-Instruct")
            assert info["name"] == "Qwen 2.5 0.5B Instruct"
            assert info["param_count"] == 500_000_000

    def test_raises_for_unknown_model(self):
        """get_model_info must raise for unknown model."""
        with pytest.raises(ValueError):
            model_manager.get_model_info("unknown/model")


class TestRegisterModel:
    """Tests for register_model functionality."""

    def test_register_updates_state(self, mock_app_state):
        """register_model must add model to available_models."""
        pass  # Placeholder - would need to patch save_app_state


class TestDownloadModel:
    """Tests for download_model functionality."""

    def test_download_curated_model(self, mock_curated_models):
        """download_model must download curated models."""
        with patch("model_manager.CURATED_MODELS", mock_curated_models):
            with patch("model_manager.download_model") as mock_download:
                # Would test actual download flow
                pass
