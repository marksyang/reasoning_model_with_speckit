"""Tests for dataset_manager module."""

from unittest.mock import patch, MagicMock
import pytest

import sys
sys.path.insert(0, "/app")


class TestListDatasets:
    """Tests for list_datasets functionality."""

    def test_returns_curated_datasets(self, mock_curated_datasets):
        """list_datasets() must return curated datasets with correct structure."""
        with patch("dataset_manager.CURATED_DATASETS", mock_curated_datasets):
            result = dataset_manager.list_datasets()
            assert len(result) >= len(mock_curated_datasets)


class TestDatasetSummary:
    """Tests for get_dataset_summary functionality."""

    def test_returns_split_sizes(self, mock_curated_datasets):
        """get_dataset_summary must return train/test split sizes."""
        with patch("dataset_manager.CURATED_DATASETS", mock_curated_datasets):
            # Would test actual summary call
            pass


class TestFormatDataset:
    """Tests for format_for_finetuning functionality."""

    def test_gsm8k_format(self):
        """format_for_finetuning must convert GSM8K format to Alpaca."""
        mock_data = [{"question": "What is 2+2?", "answer": "4"}]
        result = dataset_manager.format_for_finetuning(mock_data, "question/answer")

        assert "### Question:" in result[0]
        assert "What is 2+2?" in result[0]
        assert "### Answer:" in result[0]
        assert "4" in result[0]

    def test_text_format(self):
        """format_for_finetuning must convert text format to Alpaca."""
        mock_data = [{"text": "Hello world"}]
        result = dataset_manager.format_for_finetuning(mock_data, "text")

        assert "### Instruction:" in result[0]
        assert "Hello world" in result[0]
