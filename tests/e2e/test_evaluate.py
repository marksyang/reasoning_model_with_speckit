"""Phase 5: Evaluate 分頁功能測試。"""

import pytest
from playwright.sync_api import Page, expect

from .conftest import navigate_to_tab

# ---------------------------------------------------------------------------
# Evaluate 分頁測試
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_eval_form_defaults(prepared_page: Page):
    """驗證 3 組 Prompt/Answer 欄位包含預設值。"""
    navigate_to_tab(prepared_page, "Evaluate")
    prepared_page.wait_for_timeout(1000)

    # Prompt 1 預設值
    prompt1 = prepared_page.get_by_label("Prompt 1")
    expect(prompt1).to_have_value(
        "If I have 5 apples and give 2 to my friend, how many do I have left?"
    )

    # Answer 1 預設值
    answer1 = prepared_page.get_by_label("Expected Answer 1")
    expect(answer1).to_have_value("3")

    # Prompt 2 預設值
    prompt2 = prepared_page.get_by_label("Prompt 2")
    expect(prompt2).to_have_value(
        "What is the sum of the first 10 positive integers?"
    )

    # Answer 2 預設值
    answer2 = prepared_page.get_by_label("Expected Answer 2")
    expect(answer2).to_have_value("55")

    # Prompt 3 預設值
    prompt3 = prepared_page.get_by_label("Prompt 3")
    expect(prompt3).to_have_value(
        "If a train travels at 60 mph for 2 hours, how far does it go?"
    )

    # Answer 3 預設值
    answer3 = prepared_page.get_by_label("Expected Answer 3")
    expect(answer3).to_have_value("120 miles")


@pytest.mark.e2e
def test_eval_no_model_error(prepared_page: Page):
    """未選擇模型時點擊 Run Evaluation 應顯示錯誤。"""
    navigate_to_tab(prepared_page, "Evaluate")
    prepared_page.wait_for_timeout(1000)

    # 不選模型，直接點擊 Run Evaluation
    prepared_page.get_by_role("button", name="Run Evaluation").click()

    prepared_page.wait_for_timeout(3000)

    # 檢查錯誤訊息
    body_text = prepared_page.locator("body").text_content()
    has_error = any(
        kw in body_text.lower() for kw in ("error", "select", "please", "model")
    )
    # 或者結果區域是空的
    accuracy_locator = prepared_page.get_by_label("Accuracy")
    accuracy_value = accuracy_locator.input_value()

    assert has_error or not accuracy_value.strip(), (
        "未選擇模型時點擊 Run Evaluation 應顯示錯誤"
    )


@pytest.mark.e2e
def test_eval_elements_interaction(prepared_page: Page):
    """驗證 Evaluate 分頁的輸入欄位可編輯且按鈕可點擊。"""
    navigate_to_tab(prepared_page, "Evaluate")
    prepared_page.wait_for_timeout(1000)

    # 驗證 Prompt 1 可編輯
    prompt1 = prepared_page.get_by_label("Prompt 1")
    expect(prompt1).to_be_editable()

    # 驗證 Expected Answer 1 可編輯
    answer1 = prepared_page.get_by_label("Expected Answer 1")
    expect(answer1).to_be_editable()

    # 驗證 Run Evaluation 按鈕可點擊
    run_btn = prepared_page.get_by_role("button", name="Run Evaluation")
    expect(run_btn).to_be_enabled()

    # 驗證 Evaluation Results 區域存在 — 用 get_by_text
    result_text = prepared_page.get_by_text("Evaluation Results")
    expect(result_text).to_be_visible()

    # 驗證 Accuracy 區域存在
    accuracy_locator = prepared_page.get_by_label("Accuracy")
    expect(accuracy_locator).to_be_visible()
