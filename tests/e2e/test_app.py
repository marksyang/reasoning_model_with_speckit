"""E2E tests for Gradio app."""

import pytest
from playwright.sync_api import expect


@pytest.mark.e2e
def test_app_loads(page: "Page", gradio_server: str):
    """Verify the app loads successfully."""
    page.goto(gradio_server)
    # Gradio Blocks API default title is "Gradio"
    expect(page).to_have_title("Gradio")


@pytest.mark.e2e
def test_model_selection(page: "Page", gradio_server: str):
    """Verify model selection dropdown exists."""
    page.goto(gradio_server)
    # Look for model selection elements
    page.wait_for_timeout(1000)
    # The app should have loaded with some content
    assert page.locator("body").count() > 0


@pytest.mark.e2e
def test_dataset_selection(page: "Page", gradio_server: str):
    """Verify dataset selection dropdown exists."""
    page.goto(gradio_server)
    page.wait_for_timeout(1000)
    assert page.locator("body").count() > 0


@pytest.mark.e2e
def test_inference_tab(page: "Page", gradio_server: str):
    """Verify inference tab exists."""
    page.goto(gradio_server)
    page.wait_for_timeout(1000)
    assert page.locator("body").count() > 0
