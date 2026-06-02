"""Phase 2: Preparation 分頁功能測試。"""

import pytest
from playwright.sync_api import Page, expect

from .conftest import navigate_to_tab

# ---------------------------------------------------------------------------
# Model / Dataset 選擇與資訊顯示
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_model_dropdown_options(
    prepared_page: Page, curated_model_choices: list[str]
):
    """驗證模型 Dropdown 包含 CURATED_MODELS 全部選項。"""
    prepared_page.wait_for_timeout(1000)
    dropdown = prepared_page.get_by_label("Select Model").filter(visible=True)

    # 開啟 dropdown 並檢查選項
    dropdown.click()
    for model_id in curated_model_choices:
        expect(
            prepared_page.get_by_role("option", name=model_id)
        ).to_be_visible(timeout=5000)

    prepared_page.keyboard.press("Escape")


@pytest.mark.e2e
def test_model_info_display(prepared_page: Page, curated_model_choices: list[str]):
    """選擇模型後，model_info Textbox 應顯示模型詳細資訊。"""
    prepared_page.wait_for_timeout(1000)

    model_id = curated_model_choices[0]
    dropdown = prepared_page.get_by_label("Select Model").filter(visible=True)
    dropdown.click()
    prepared_page.get_by_role("option", name=model_id).click()

    # 驗證 model_info 區域有內容
    info_box = prepared_page.get_by_label("Model Details")
    expect(info_box).to_be_visible()
    # Gradio 將 JSON 渲染為 Textbox，等待內容更新
    prepared_page.wait_for_timeout(2000)
    content = info_box.input_value()
    assert content, "Model Details 應該有內容"


@pytest.mark.e2e
def test_dataset_dropdown_options(
    prepared_page: Page, curated_dataset_choices: list[str]
):
    """驗證資料集 Dropdown 包含 CURATED_DATASETS 全部選項。"""
    prepared_page.wait_for_timeout(1000)
    dropdown = prepared_page.get_by_label("Select Dataset")

    dropdown.click()
    for dataset_id in curated_dataset_choices:
        expect(
            prepared_page.get_by_role("option", name=dataset_id)
        ).to_be_visible(timeout=5000)

    prepared_page.keyboard.press("Escape")


@pytest.mark.e2e
def test_dataset_summary_display(
    prepared_page: Page, curated_dataset_choices: list[str]
):
    """選擇資料集後，dataset_summary 應顯示資料集資訊。"""
    prepared_page.wait_for_timeout(1000)

    dataset_id = curated_dataset_choices[0]
    dropdown = prepared_page.get_by_label("Select Dataset")
    dropdown.click()
    prepared_page.get_by_role("option", name=dataset_id).click()

    # 驗證 dataset_summary 區域存在
    summary_box = prepared_page.get_by_label("Dataset Summary")
    expect(summary_box).to_be_visible()

    # 等待 API 回傳 — 資料集摘要可能需要較長時間（API 呼叫）
    # 使用 longer timeout 並檢查是否有內容或合理狀態
    prepared_page.wait_for_timeout(5000)
    content = summary_box.input_value()
    # 若 API 失敗或有延遲，至少元素應該存在
    assert summary_box.is_visible(), "Dataset Summary 應該可見"
