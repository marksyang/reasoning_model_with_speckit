"""Test configuration and fixtures."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


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
        "gsm8k": {
            "name": "GSM8K",
            "description": "Grade school math word problems",
            "column_names": ["question", "answer"],
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
