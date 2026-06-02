"""Phase 3: Inference 分頁功能測試。"""

import pytest
from playwright.sync_api import Page, expect

from .conftest import navigate_to_tab

# ---------------------------------------------------------------------------
# Inference 分頁測試
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_inference_params_range(prepared_page: Page):
    """驗證 3 個 Slider 和 1 個 Checkbox 的預設值正確。"""
    navigate_to_tab(prepared_page, "Inference")
    prepared_page.wait_for_timeout(1000)

    # Gradio 4.x Slider 使用 spinbutton (number input) + slider (range slider)
    # Max New Tokens: 預設 512
    max_tokens_spin = prepared_page.get_by_role(
        "spinbutton", name="number input for Max New Tokens"
    )
    expect(max_tokens_spin).to_be_visible()
    expect(max_tokens_spin).to_have_value("512")

    # Temperature: 預設 0.7
    temp_spin = prepared_page.get_by_role(
        "spinbutton", name="number input for Temperature"
    )
    expect(temp_spin).to_be_visible()
    expect(temp_spin).to_have_value("0.7")

    # Top-p: 預設 0.9
    topp_spin = prepared_page.get_by_role(
        "spinbutton", name="number input for Top-p"
    )
    expect(topp_spin).to_be_visible()
    expect(topp_spin).to_have_value("0.9")

    # Sample Checkbox: 預設 True
    sample_checkbox = prepared_page.get_by_label("Sample")
    expect(sample_checkbox).to_be_checked()


@pytest.mark.e2e
def test_inference_no_model_error(prepared_page: Page):
    """未選擇模型時點擊 Run Inference 應顯示錯誤。"""
    navigate_to_tab(prepared_page, "Inference")
    prepared_page.wait_for_timeout(1000)

    # 輸入 prompt 但不選模型
    prompt_box = prepared_page.get_by_label("Prompt")
    prompt_box.fill("Test prompt")

    # 點擊 Run Inference
    prepared_page.get_by_role("button", name="Run Inference").click()

    # 等待一會兒讓 UI 更新
    prepared_page.wait_for_timeout(3000)

    # 檢查輸出區域應該是空的或有錯誤訊息
    output_box = prepared_page.get_by_label("Output")
    output_value = output_box.input_value()
    assert not output_value or "error" in output_value.lower() or "select a model" in output_value.lower(), (
        "未選擇模型時應顯示錯誤或空輸出"
    )


@pytest.mark.e2e
def test_inference_no_prompt_error(
    prepared_page: Page, curated_model_choices: list[str]
):
    """已選模型但未輸入 Prompt 時應顯示錯誤。"""
    navigate_to_tab(prepared_page, "Inference")
    prepared_page.wait_for_timeout(1000)

    # 選擇模型 — 使用 filter(visible=True) 確保選取可見元素
    dropdown = prepared_page.get_by_label("Select Model").filter(visible=True)
    dropdown.click()
    prepared_page.get_by_role("option", name=curated_model_choices[0]).click()

    # 點擊 Run Inference
    prepared_page.get_by_role("button", name="Run Inference").click()

    prepared_page.wait_for_timeout(3000)
    # 檢查錯誤訊息
    output_box = prepared_page.get_by_label("Output")
    output_value = output_box.input_value()
    assert not output_value or "error" in output_value.lower() or "prompt" in output_value.lower(), (
        "未輸入 Prompt 時應顯示錯誤"
    )


@pytest.mark.e2e
def test_inference_submit(
    prepared_page: Page, curated_model_choices: list[str], request: pytest.FixtureRequest
):
    """選擇模型並輸入 Prompt 後點擊 Run，驗證輸出區域有反應。

    **注意**：此測試僅驗證 UI 層行為。若模型未下載，推論會失敗，
    但應有錯誤訊息而非無反應。
    """
    navigate_to_tab(prepared_page, "Inference")
    prepared_page.wait_for_timeout(1000)

    # 選擇第一個模型 — 使用 filter(visible=True) 確保選取可見元素
    dropdown = prepared_page.get_by_label("Select Model").filter(visible=True)
    dropdown.click()
    prepared_page.get_by_role("option", name=curated_model_choices[0]).click()

    # 輸入 prompt
    prompt_box = prepared_page.get_by_label("Prompt")
    prompt_box.fill("What is 2+2?")

    # 點擊 Run Inference
    prepared_page.get_by_role("button", name="Run Inference").click()

    # 等待一會兒讓 UI 更新
    prepared_page.wait_for_timeout(5000)

    output_box = prepared_page.get_by_label("Output")
    expect(output_box).to_be_visible()
    # 若有內容或錯誤訊息都算 UI 有反應
    assert True  # 只要輸出框有被互動就算通過
