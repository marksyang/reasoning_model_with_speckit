"""Tests for src/dataset_manager module."""

from unittest.mock import MagicMock, patch

import pytest

from src.dataset_manager import (
    CURATED_DATASETS,
    DatasetInfo,
    download_dataset,
    format_for_finetuning,
    get_available_datasets,
    get_dataset_summary,
    list_datasets,
)


class TestDatasetInfo:
    """Tests for DatasetInfo dataclass."""

    def test_dataset_info_defaults(self):
        info = DatasetInfo(hf_id="test/dataset", name="Test")
        assert info.hf_id == "test/dataset"
        assert info.name == "Test"
        assert info.description == ""
        assert info.train_size == 0
        assert info.test_size == 0
        assert info.column_names == []
        assert info.is_cached is False
        assert info.status == "pending"

    def test_dataset_info_custom_values(self):
        info = DatasetInfo(
            hf_id="test/dataset",
            name="Test",
            description="A test dataset",
            train_size=1000,
            test_size=200,
            column_names=["col1", "col2"],
            is_cached=True,
            status="cached",
        )
        assert info.description == "A test dataset"
        assert info.train_size == 1000
        assert info.is_cached is True


class TestListDatasets:
    """Tests for list_datasets and get_available_datasets."""

    def test_list_datasets_returns_all(self):
        datasets = list_datasets()
        assert len(datasets) == len(CURATED_DATASETS)
        hf_ids = {d["hf_id"] for d in datasets}
        for curated_id in CURATED_DATASETS:
            assert curated_id in hf_ids

    def test_dataset_has_required_fields(self):
        datasets = list_datasets()
        for ds in datasets:
            assert "hf_id" in ds
            assert "name" in ds
            assert "description" in ds
            assert "column_names" in ds

    def test_get_available_datasets_same_as_list(self):
        available = get_available_datasets()
        listed = list_datasets()
        assert len(available) == len(listed)
        assert {d["hf_id"] for d in available} == {d["hf_id"] for d in listed}


class TestDownloadDataset:
    """Tests for download_dataset."""

    def test_download_dataset_success(self):
        mock_dataset = MagicMock()
        with patch("src.dataset_manager.load_dataset", return_value=mock_dataset):
            result = download_dataset("openai/gsm8k")
        assert result == "openai/gsm8k"

    def test_download_dataset_with_config(self):
        mock_dataset = MagicMock()
        with patch("src.dataset_manager.load_dataset", return_value=mock_dataset):
            result = download_dataset("openai/gsm8k", config="main")
        assert result == "openai/gsm8k"

    def test_download_dataset_raises_on_error(self):
        with patch("src.dataset_manager.load_dataset", side_effect=Exception("Network error")):
            with pytest.raises(Exception, match="Network error"):
                download_dataset("openai/gsm8k")


class TestDatasetSummary:
    """Tests for get_dataset_summary."""

    def test_get_dataset_summary(self):
        from datasets import Dataset

        mock_split = MagicMock(spec=Dataset)
        mock_split.column_names = ["question", "answer"]
        len_call = MagicMock(return_value=1000)
        mock_split.__len__ = len_call

        mock_dataset = MagicMock()
        mock_dataset.items.return_value = [("train", mock_split)]

        with patch("src.dataset_manager.load_dataset", return_value=mock_dataset):
            with patch("src.dataset_manager.Dataset", Dataset):
                summary = get_dataset_summary("openai/gsm8k")
        assert summary["hf_id"] == "openai/gsm8k"
        assert summary["name"] == "GSM8K"
        assert "train_size" in summary
        assert "train_columns" in summary


class TestFormatForFinetuning:
    """Tests for format_for_finetuning."""

    def test_format_question_answer(self):
        dataset = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is 3+3?", "answer": "6"},
        ]
        prompts = format_for_finetuning(dataset, source_format="question/answer")
        assert len(prompts) == 2
        assert "### Question:" in prompts[0]
        assert "What is 2+2?" in prompts[0]
        assert "### Answer:" in prompts[0]
        assert "4" in prompts[0]

    def test_format_text(self):
        dataset = [
            {"text": "Hello world"},
            {"text": "Goodbye world"},
        ]
        prompts = format_for_finetuning(dataset, source_format="text")
        assert len(prompts) == 2
        assert "### Instruction:" in prompts[0]
        assert "Hello world" in prompts[0]

    def test_format_empty_dataset(self):
        prompts = format_for_finetuning([], source_format="question/answer")
        assert prompts == []

    def test_format_missing_keys(self):
        dataset = [{"question": "test"}]
        prompts = format_for_finetuning(dataset, source_format="question/answer")
        assert len(prompts) == 1
        assert "### Answer:" in prompts[0]
        assert "" in prompts[0]
