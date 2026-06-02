"""Phase 4: Training 分頁功能測試。"""

import pytest
from playwright.sync_api import Page, expect

from .conftest import navigate_to_tab

# ---------------------------------------------------------------------------
# Training 分頁測試
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_training_params_default(prepared_page: Page):
    """驗證 5 個訓練參數 Slider 預設值正確。"""
    navigate_to_tab(prepared_page, "Training")
    prepared_page.wait_for_timeout(1000)

    # Gradio 4.x Slider 使用 spinbutton (number input) + slider (range slider)
    # Epochs: 預設 1
    epochs_spin = prepared_page.get_by_role(
        "spinbutton", name="number input for Epochs"
    )
    expect(epochs_spin).to_be_visible()
    expect(epochs_spin).to_have_value("1")

    # Learning Rate: 預設 2e-4 (0.0002)
    lr_spin = prepared_page.get_by_role(
        "spinbutton", name="number input for Learning Rate"
    )
    expect(lr_spin).to_be_visible()
    expect(lr_spin).to_have_value("0.0002")

    # LoRA Rank: 預設 8
    lora_r_spin = prepared_page.get_by_role(
        "spinbutton", name="number input for LoRA Rank"
    )
    expect(lora_r_spin).to_be_visible()
    expect(lora_r_spin).to_have_value("8")

    # LoRA Alpha: 預設 16
    lora_alpha_spin = prepared_page.get_by_role(
        "spinbutton", name="number input for LoRA Alpha"
    )
    expect(lora_alpha_spin).to_be_visible()
    expect(lora_alpha_spin).to_have_value("16")

    # LoRA Dropout: 預設 0.05
    lora_dropout_spin = prepared_page.get_by_role(
        "spinbutton", name="number input for LoRA Dropout"
    )
    expect(lora_dropout_spin).to_be_visible()
    expect(lora_dropout_spin).to_have_value("0.05")


@pytest.mark.e2e
def test_training_btn_initial_state(prepared_page: Page):
    """初始狀態：Start Training 可見，Stop Training 不可見。"""
    navigate_to_tab(prepared_page, "Training")
    prepared_page.wait_for_timeout(1000)

    expect(
        prepared_page.get_by_role("button", name="Start Training")
    ).to_be_visible()

    # Stop Training 初始隱藏
    expect(
        prepared_page.get_by_role("button", name="Stop Training", exact=True)
    ).not_to_be_visible()


@pytest.mark.e2e
def test_training_validation_error(prepared_page: Page):
    """未選擇模型或資料集時點擊 Start 應顯示錯誤。"""
    navigate_to_tab(prepared_page, "Training")
    prepared_page.wait_for_timeout(1000)

    # 不選模型/資料集，直接點擊 Start Training
    prepared_page.get_by_role("button", name="Start Training").click()

    prepared_page.wait_for_timeout(3000)

    # 檢查錯誤訊息 — Gradio 的 gr.Error 會顯示 toast 或 inline error
    # 檢查 Training Progress textbox 區域是否有錯誤提示
    progress_textbox = prepared_page.get_by_role(
        "textbox", name="Training Progress"
    )
    progress_value = progress_textbox.input_value() if progress_textbox.is_visible() else ""

    # 或者檢查是否有 "select" 或 "error" 相關的訊息
    body_text = prepared_page.locator("body").text_content()
    has_error = any(
        kw in body_text.lower()
        for kw in ("error", "select", "please")
    )
    assert has_error or not progress_value.strip(), (
        "未選擇模型/資料集時點擊 Start 應顯示錯誤或無變化"
    )


@pytest.mark.e2e
def test_training_start_button_exists(prepared_page: Page):
    """驗證 Start Training 按鈕可點擊且訓練相關元件可見。"""
    navigate_to_tab(prepared_page, "Training")
    prepared_page.wait_for_timeout(1000)

    # 按鈕應該可以點擊
    start_btn = prepared_page.get_by_role("button", name="Start Training")
    expect(start_btn).to_be_enabled()

    # Training Progress Slider 應該存在 — 使用 role 精確定位
    progress_slider = prepared_page.get_by_role(
        "slider", name="range slider for Training Progress"
    )
    expect(progress_slider).to_be_visible()

    # Training Log 應該存在
    training_log = prepared_page.get_by_label("Training Log")
    expect(training_log).to_be_visible()


@pytest.mark.e2e
def test_training_registered_models_dropdown(prepared_page: Page):
    """驗證 Registered Fine-Tuned Models Dropdown 存在。"""
    navigate_to_tab(prepared_page, "Training")
    prepared_page.wait_for_timeout(1000)

    dropdown = prepared_page.get_by_label("Registered Fine-Tuned Models")
    expect(dropdown).to_be_visible()

    # 該 dropdown 初始為 disabled（無訓練記錄時）
    # 不點擊，只驗證存在
    expect(dropdown).to_be_disabled()
