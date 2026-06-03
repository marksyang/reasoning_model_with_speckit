"""E2E test fixtures for Gradio app."""

import subprocess
import sys
import time
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def wait_for_gradio_ready(page: Page, timeout: int = 30_000) -> None:
    """等待 Gradio 完成初始渲染。

    Gradio Blocks 应用會先插入 ``<gradio-app>`` 佔位元素，
    當內部框架載入完畢後該元素才會出現。
    """
    page.wait_for_selector("gradio-app", state="attached", timeout=timeout)


def navigate_to_tab(page: Page, tab_name: str) -> None:
    """導航到指定分頁並等待渲染。"""
    page.get_by_role("tab", name=tab_name, exact=True).click()
    # Gradio 分頁切換有動畫，稍作等待確保 DOM 穩定
    page.wait_for_timeout(300)

# ---------------------------------------------------------------------------
# Server Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def gradio_server():
    """Start the Gradio app for E2E testing."""
    project_root = Path(__file__).parent.parent.parent
    port = 7860

    proc = subprocess.Popen(
        [sys.executable, "-m", "src.app", "--port", str(port)],
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    max_wait = 120
    start = time.time()
    while time.time() - start < max_wait:
        # 檢查子程序是否提前退出
        return_code = proc.poll()
        if return_code is not None:
            _, stderr = proc.communicate()
            pytest.fail(
                f"Gradio process exited prematurely with code {return_code}. "
                f"stderr: {stderr.decode('utf-8', errors='replace')}"
            )
        try:
            import urllib.request

            urllib.request.urlopen(f"http://localhost:{port}", timeout=1)
            break
        except Exception:
            time.sleep(1)
    else:
        proc.kill()
        _, stderr = proc.communicate()
        pytest.fail(
            f"Gradio server failed to start within {max_wait}s. "
            f"stderr: {stderr.decode('utf-8', errors='replace')}"
        )

    yield f"http://localhost:{port}"

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture
def browser_context_args(browser_context_args):
    return {
        **browser_context_args,
        "viewport": {"width": 1280, "height": 720},
    }

# ---------------------------------------------------------------------------
# Page Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prepared_page(page: Page, gradio_server: str) -> Page:
    """載入首頁並等待 Gradio 渲染完成。"""
    page.goto(gradio_server)
    wait_for_gradio_ready(page)
    return page

# ---------------------------------------------------------------------------
# Data Fixtures — 動態讀取 CURATED_* 清單
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def curated_model_choices():
    """從 model_manager 讀取 CURATED_MODELS 鍵清單。"""
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.model_manager import CURATED_MODELS

    return list(CURATED_MODELS.keys())


@pytest.fixture(scope="session")
def curated_dataset_choices():
    """從 dataset_manager 讀取 CURATED_DATASETS 鍵清單。"""
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.dataset_manager import CURATED_DATASETS

    return list(CURATED_DATASETS.keys())
