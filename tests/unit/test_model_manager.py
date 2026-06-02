"""Tests for src/model_manager module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.model_manager import (
    CURATED_MODELS,
    ModelInfo,
    download_model,
    get_cached_models,
    get_model_cache_path,
    get_model_info,
    list_models,
    list_models_by_param,
    load_app_state,
    register_model,
    save_app_state,
)


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_model_has_required_fields(self):
        model = ModelInfo(
            hf_id="test/model",
            name="Test Model",
            param_count=1000000,
        )
        assert model.hf_id == "test/model"
        assert model.name == "Test Model"
        assert model.param_count == 1000000
        assert model.max_tokens == 512
        assert model.note == ""
        assert model.is_cached is False
        assert model.status == "pending"

    def test_model_info_custom_values(self):
        model = ModelInfo(
            hf_id="test/model",
            name="Test Model",
            param_count=2000000,
            max_tokens=1024,
            note="Custom note",
            is_cached=True,
            status="cached",
        )
        assert model.max_tokens == 1024
        assert model.note == "Custom note"
        assert model.is_cached is True
        assert model.status == "cached"


class TestAppState:
    """Tests for app state functions."""

    def test_load_app_state_empty(self, tmp_dir):
        with patch("src.model_manager.APP_STATE_PATH", tmp_dir / "app_state.json"):
            state = load_app_state()
        assert state == {
            "available_models": [],
            "available_datasets": [],
            "registered_adapters": [],
            "selected_model": None,
            "selected_dataset": None,
            "training_runs": [],
        }

    def test_load_app_state_existing(self, tmp_dir):
        state_path = tmp_dir / "app_state.json"
        state_path.write_text(json.dumps({"available_models": ["test/model"]}))
        with patch("src.model_manager.APP_STATE_PATH", state_path):
            state = load_app_state()
        assert state["available_models"] == ["test/model"]

    def test_save_and_load_app_state(self, tmp_dir):
        state_path = tmp_dir / "app_state.json"
        state = {"available_models": ["test/model"]}
        with patch("src.model_manager.APP_STATE_PATH", state_path):
            save_app_state(state)
            loaded = load_app_state()
        assert loaded["available_models"] == ["test/model"]


class TestModelCache:
    """Tests for model cache functions."""

    def test_get_model_cache_path(self):
        path = get_model_cache_path()
        assert path.exists()
        assert "huggingface" in str(path)
        assert "hub" in str(path)

    def test_get_cached_models_empty(self, tmp_dir):
        empty_cache = tmp_dir / "empty_cache"
        with patch("src.model_manager.get_model_cache_path", return_value=empty_cache):
            with patch("src.model_manager.APP_STATE_PATH", tmp_dir / "app_state.json"):
                cached = get_cached_models()
        assert cached == []

    def test_get_cached_models_with_dir(self, tmp_dir):
        cache_path = tmp_dir / "cache" / "huggingface" / "hub"
        cache_path.mkdir(parents=True)
        model_dir = cache_path / "models--test--model"
        model_dir.mkdir()

        with patch("src.model_manager.get_model_cache_path", return_value=cache_path):
            with patch("src.model_manager.APP_STATE_PATH", tmp_dir / "app_state.json"):
                cached = get_cached_models()
        assert len(cached) == 1
        assert cached[0]["hf_id"] == "test/model"
        assert cached[0]["is_cached"] is True


class TestListModels:
    """Tests for list_models functions."""

    def test_list_models_curated_only(self, tmp_dir):
        with patch("src.model_manager.APP_STATE_PATH", tmp_dir / "app_state.json"):
            with patch("src.model_manager.list_models_by_param", return_value=[]):
                models = list_models()
        hf_ids = {m["hf_id"] for m in models}
        for curated_id in CURATED_MODELS:
            assert curated_id in hf_ids

    def test_list_models_includes_local(self, tmp_dir):
        local_model = MagicMock()
        local_model.hf_id = "local/model"
        local_model.name = "Local Model"
        local_model.param_count = 1000000

        with patch("src.model_manager.APP_STATE_PATH", tmp_dir / "app_state.json"):
            with patch("src.model_manager.list_models_by_param", return_value=[local_model]):
                models = list_models()
        hf_ids = {m["hf_id"] for m in models}
        assert "local/model" in hf_ids

    def test_list_models_by_param_returns_models(self):
        mock_model = MagicMock()
        mock_model.id = "test/model"
        mock_model.card_data = {"model_index": {"Parameters": "1000000"}}

        mock_api = MagicMock()
        mock_api.list_models.return_value = [mock_model]

        with patch("src.model_manager.HfApi", return_value=mock_api):
            models = list_models_by_param(max_params=3_000_000_000)
        assert len(models) >= 1
        assert models[0].hf_id == "test/model"

    def test_list_models_by_param_empty_on_error(self):
        mock_api = MagicMock()
        mock_api.list_models.side_effect = Exception("API error")

        with patch("src.model_manager.HfApi", return_value=mock_api):
            models = list_models_by_param(max_params=3_000_000_000)
        assert models == []


class TestModelInfo:
    """Tests for get_model_info."""

    def test_returns_info_for_curated_model(self):
        info = get_model_info("meta-llama/Llama-3.2-1B-Instruct")
        assert info["hf_id"] == "meta-llama/Llama-3.2-1B-Instruct"
        assert info["name"] == "Meta LLaMA 3.2 1B Instruct"
        assert info["param_count"] == 1_000_000_000

    def test_raises_for_unknown_model(self):
        with pytest.raises(ValueError, match="not in curated list"):
            get_model_info("unknown/model")


class TestRegisterModel:
    """Tests for register_model."""

    def test_register_updates_state(self, tmp_dir):
        state_path = tmp_dir / "app_state.json"
        with patch("src.model_manager.APP_STATE_PATH", state_path):
            register_model("new/model")
            state = load_app_state()
        assert "new/model" in state["available_models"]

    def test_register_does_not_duplicate(self, tmp_dir):
        state_path = tmp_dir / "app_state.json"
        with patch("src.model_manager.APP_STATE_PATH", state_path):
            register_model("new/model")
            register_model("new/model")
            state = load_app_state()
        assert state["available_models"].count("new/model") == 1


class TestDownloadModel:
    """Tests for download_model."""

    def test_download_model_success(self, tmp_dir):
        with patch("src.model_manager.APP_STATE_PATH", tmp_dir / "app_state.json"):
            with patch("src.model_manager.AutoConfig.from_pretrained"):
                with patch("src.model_manager.AutoTokenizer.from_pretrained"):
                    with patch("src.model_manager.AutoModelForCausalLM.from_pretrained"):
                        path = download_model("meta-llama/Llama-3.2-1B-Instruct")
                        assert path is not None

    def test_download_model_not_curated(self, tmp_dir):
        with patch("src.model_manager.APP_STATE_PATH", tmp_dir / "app_state.json"):
            with pytest.raises(ValueError, match="not in curated list"):
                download_model("unknown/model")
