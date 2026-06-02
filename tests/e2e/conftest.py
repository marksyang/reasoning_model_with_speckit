"""E2E test fixtures for Gradio app."""

import subprocess
import time
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect


@pytest.fixture(scope="session")
def gradio_server():
    """Start the Gradio app for E2E testing."""
    project_root = Path(__file__).parent.parent.parent
    port = 7860

    proc = subprocess.Popen(
        ["python", "-m", "src.app", "--port", str(port)],
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    max_wait = 30
    start = time.time()
    while time.time() - start < max_wait:
        try:
            import urllib.request
            urllib.request.urlopen(f"http://localhost:{port}", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        proc.kill()
        pytest.fail("Gradio server failed to start")

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
