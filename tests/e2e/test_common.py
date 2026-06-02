"""Phase 1 & 6: 基礎頁面結構與共用元件測試。"""

import pytest
from playwright.sync_api import Page, expect

from .conftest import navigate_to_tab

# ---------------------------------------------------------------------------
# Phase 1: 基礎頁面結構
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_app_loads(prepared_page: Page):
    """驗證應用程式成功載入且標題正確。"""
    expect(prepared_page).to_have_title("Gradio")


@pytest.mark.e2e
def test_tabs_exist(prepared_page: Page):
    """確認 4 個分頁標籤存在。"""
    for tab_name in ("Preparation", "Inference", "Training", "Evaluate"):
        expect(
            prepared_page.get_by_role("tab", name=tab_name, exact=True)
        ).to_be_visible()


@pytest.mark.e2e
def test_tab_switching(prepared_page: Page):
    """依序點擊每個分頁，驗證分頁內容正確切換。"""
    for tab_name in ("Preparation", "Inference", "Training", "Evaluate"):
        navigate_to_tab(prepared_page, tab_name)
        expect(
            prepared_page.get_by_role("tab", name=tab_name, exact=True)
        ).to_be_visible()


@pytest.mark.e2e
def test_preparation_elements(prepared_page: Page):
    """確認 Preparation 分頁所有關鍵元件存在。"""
    prepared_page.wait_for_timeout(1000)

    # Model section — 使用 filter(visible=True) 確保選取可見元素
    expect(
        prepared_page.get_by_label("Select Model").filter(visible=True)
    ).to_be_visible()
    expect(
        prepared_page.get_by_role("button", name="Download Model")
    ).to_be_visible()
    expect(prepared_page.get_by_label("Model Status")).to_be_visible()
    expect(prepared_page.get_by_label("Model Details")).to_be_visible()

    # Dataset section
    expect(prepared_page.get_by_label("Select Dataset")).to_be_visible()
    expect(
        prepared_page.get_by_role("button", name="Download Dataset")
    ).to_be_visible()
    expect(prepared_page.get_by_label("Dataset Status")).to_be_visible()
    expect(prepared_page.get_by_label("Dataset Summary")).to_be_visible()


@pytest.mark.e2e
def test_inference_elements(prepared_page: Page):
    """確認 Inference 分頁所有關鍵元件存在。"""
    navigate_to_tab(prepared_page, "Inference")
    prepared_page.wait_for_timeout(500)

    # Model dropdown — 使用 filter(visible=True) 確保選取可見元素
    expect(
        prepared_page.get_by_label("Select Model").filter(visible=True)
    ).to_be_visible()

    # Prompt textbox
    expect(prepared_page.get_by_label("Prompt")).to_be_visible()

    # Sliders — Gradio 4.x 使用 spinbutton/slider 而非 input
    expect(
        prepared_page.get_by_role("slider", name="range slider for Max New Tokens")
    ).to_be_visible()
    expect(
        prepared_page.get_by_role("slider", name="range slider for Temperature")
    ).to_be_visible()
    expect(
        prepared_page.get_by_role("slider", name="range slider for Top-p")
    ).to_be_visible()

    # Checkbox
    expect(prepared_page.get_by_label("Sample")).to_be_checked()

    # Run button
    expect(
        prepared_page.get_by_role("button", name="Run Inference")
    ).to_be_visible()

    # Output
    expect(prepared_page.get_by_label("Output")).to_be_visible()


@pytest.mark.e2e
def test_training_elements(prepared_page: Page):
    """確認 Training 分頁所有關鍵元件存在。"""
    navigate_to_tab(prepared_page, "Training")
    prepared_page.wait_for_timeout(500)

    # Dropdowns — 使用 filter(visible=True) 確保選取可見元素
    expect(
        prepared_page.get_by_label("Select Model").filter(visible=True)
    ).to_be_visible()
    expect(
        prepared_page.get_by_label("Select Dataset").filter(visible=True)
    ).to_be_visible()

    # Sliders — 使用 role=slider
    expect(
        prepared_page.get_by_role("slider", name="range slider for Epochs")
    ).to_be_visible()
    expect(
        prepared_page.get_by_role("slider", name="range slider for Learning Rate")
    ).to_be_visible()
    expect(
        prepared_page.get_by_role("slider", name="range slider for LoRA Rank")
    ).to_be_visible()
    expect(
        prepared_page.get_by_role("slider", name="range slider for LoRA Alpha")
    ).to_be_visible()
    expect(
        prepared_page.get_by_role("slider", name="range slider for LoRA Dropout")
    ).to_be_visible()

    # Buttons
    expect(
        prepared_page.get_by_role("button", name="Start Training")
    ).to_be_visible()

    # Training progress — 使用 role 精確定位
    expect(
        prepared_page.get_by_role("slider", name="range slider for Training Progress")
    ).to_be_visible()
    expect(prepared_page.get_by_label("Training Log")).to_be_visible()
    expect(
        prepared_page.get_by_label("Registered Fine-Tuned Models")
    ).to_be_visible()


@pytest.mark.e2e
def test_evaluate_elements(prepared_page: Page):
    """確認 Evaluate 分頁所有關鍵元件存在。"""
    navigate_to_tab(prepared_page, "Evaluate")
    prepared_page.wait_for_timeout(500)

    # Model dropdown
    expect(
        prepared_page.get_by_label("Select Fine-Tuned Model")
    ).to_be_visible()

    # Prompt/Answer textboxes
    for i in (1, 2, 3):
        expect(prepared_page.get_by_label(f"Prompt {i}")).to_be_visible()
        expect(prepared_page.get_by_label(f"Expected Answer {i}")).to_be_visible()

    # Run button
    expect(
        prepared_page.get_by_role("button", name="Run Evaluation")
    ).to_be_visible()

    # Results — Evaluation Results 是 JSON 元件，用 text 檢查
    expect(
        prepared_page.get_by_text("Evaluation Results")
    ).to_be_visible()
    expect(prepared_page.get_by_label("Accuracy")).to_be_visible()

# ---------------------------------------------------------------------------
# Phase 6: 共用元件測試
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_model_dropdown_consistency(
    prepared_page: Page, curated_model_choices: list[str]
):
    """驗證所有分頁中的模型 Dropdown 選項一致。"""
    expected_set = set(curated_model_choices)

    # Preparation tab (default)
    prepared_page.wait_for_timeout(500)
    prepared_page.get_by_label("Select Model").filter(visible=True).click()
    prep_options = prepared_page.locator("[role=option]").all_text_contents()
    prepared_page.keyboard.press("Escape")
    prepared_page.wait_for_timeout(300)

    # Inference tab
    navigate_to_tab(prepared_page, "Inference")
    prepared_page.get_by_label("Select Model").filter(visible=True).click()
    inf_options = prepared_page.locator("[role=option]").all_text_contents()
    prepared_page.keyboard.press("Escape")
    prepared_page.wait_for_timeout(300)

    # Training tab
    navigate_to_tab(prepared_page, "Training")
    prepared_page.get_by_label("Select Model").filter(visible=True).click()
    train_options = prepared_page.locator("[role=option]").all_text_contents()
    prepared_page.keyboard.press("Escape")
    prepared_page.wait_for_timeout(300)

    # Compare — Gradio 選項可能包含 ✓ 符號，需 stripping
    def strip_checkmark(items: list[str]) -> set[str]:
        return {item.strip().removeprefix("✓ ").strip() for item in items if item.strip()}

    prep_clean = strip_checkmark(prep_options)
    inf_clean = strip_checkmark(inf_options)
    train_clean = strip_checkmark(train_options)

    assert expected_set.issubset(prep_clean), (
        f"Preparation dropdown missing models: "
        f"{expected_set - prep_clean}"
    )
    assert expected_set.issubset(inf_clean), (
        f"Inference dropdown missing models: "
        f"{expected_set - inf_clean}"
    )
    assert expected_set.issubset(train_clean), (
        f"Training dropdown missing models: "
        f"{expected_set - train_clean}"
    )
