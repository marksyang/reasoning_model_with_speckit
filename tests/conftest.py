"""Test configuration and fixtures."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Add project root to path for imports
sys_path = str(Path(__file__).parent.parent)
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)


@pytest.fixture
def tmp_dir(tmp_path):
    """Create a temporary directory structure."""
    data_dir = tmp_path / "data"
    adapters_dir = data_dir / "adapters"
    logs_dir = data_dir / "logs"

    data_dir.mkdir()
    adapters_dir.mkdir()
    logs_dir.mkdir()

    app_state = {
        "available_models": [],
        "available_datasets": [],
        "registered_adapters": [],
        "selected_model": None,
        "selected_dataset": None,
        "training_runs": [],
    }
    with open(data_dir / "app_state.json", "w") as f:
        json.dump(app_state, f)

    return data_dir


@pytest.fixture
def mock_model_path(tmp_dir):
    """Create a mock model path."""
    model_path = tmp_dir / "models--test-model--blobs"
    model_path.mkdir(parents=True, exist_ok=True)
    return model_path


@pytest.fixture
def mock_dataset_path(tmp_dir):
    """Create a mock dataset path."""
    return tmp_dir / "datasets" / "mock_dataset"


@pytest.fixture
def mock_curated_models():
    """Mock curated models list."""
    return {
        "meta-llama/Llama-3.2-1B-Instruct": {
            "name": "Meta LLaMA 3.2 1B Instruct",
            "param_count": 1_000_000_000,
            "max_tokens": 512,
        },
        "Qwen/Qwen2.5-0.5B-Instruct": {
            "name": "Qwen 2.5 0.5B Instruct",
            "param_count": 500_000_000,
            "max_tokens": 512,
        },
    }


@pytest.fixture
def mock_curated_datasets():
    """Mock curated datasets list."""
    return {
        "openai/gsm8k": {
            "name": "GSM8K",
            "description": "Grade school math word problems",
            "column_names": ["question", "answer"],
            "config": "main",
        },
        "rajpurkar/squad": {
            "name": "SQuAD",
            "description": "Stanford Question Answering Dataset",
            "column_names": ["id", "title", "context", "question", "answers"],
        },
    }


@pytest.fixture
def mock_app_state(tmp_dir):
    """Mock app state."""
    state_path = tmp_dir / "app_state.json"

    class MockState:
        def load(self):
            if state_path.exists():
                with open(state_path) as f:
                    return json.load(f)
            return {}

        def save(self, data):
            with open(state_path, "w") as f:
                json.dump(data, f, indent=2)

    return MockState()


@pytest.fixture
def model_manager(tmp_dir, mock_curated_models):
    """Create a ModelManager instance with mocked state."""
    from src.model_manager import CURATED_MODELS, load_app_state, save_app_state

    # Patch CURATED_MODELS
    with patch("src.model_manager.CURATED_MODELS", mock_curated_models):
        # Patch state paths
        with patch("src.model_manager.APP_STATE_PATH", tmp_dir / "app_state.json"):
            yield MagicMock(
                curated_models=mock_curated_models,
                state_path=tmp_dir / "app_state.json",
            )


@pytest.fixture
def dataset_manager(tmp_dir, mock_curated_datasets):
    """Create a DatasetManager instance with mocked state."""
    from src.dataset_manager import CURATED_DATASETS

    with patch("src.dataset_manager.CURATED_DATASETS", mock_curated_datasets):
        yield MagicMock(
            curated_datasets=mock_curated_datasets,
        )
