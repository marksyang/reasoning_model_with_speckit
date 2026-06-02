# 測試計畫說明文件

## 1. 概述

本專案建立完整的測試系統，涵蓋 unit tests、E2E tests、CI/CD 自動化的完整流程，確保程式碼品質與測試覆蓋率達 90% 以上。

## 2. 測試架構

```
tests/
├── conftest.py              # 全域 fixtures
├── unit/                    # Unit tests
│   ├── test_model_manager.py
│   ├── test_dataset_manager.py
│   ├── test_trainer.py
│   └── test_inference.py
└── e2e/                     # E2E tests
    ├── conftest.py          # Gradio server fixture
    └── test_app.py          # UI 流程測試
```

## 3. 測試工具

| 工具 | 用途 |
|------|------|
| pytest | 測試框架 |
| pytest-cov | 覆蓋率分析 |
| pytest-playwright | 瀏覽器 E2E 測試 |
| playwright | 瀏覽器自動化 |
| pytest-xdist | 平行測試 |
| pytest-timeout | 測試超时保護 |
| GitHub Actions | CI/CD |
| Codecov | 覆蓋率報告 |

## 4. 測試策略

### Unit Tests
- **model_manager**: ModelInfo, app state, cache, list/download/register
- **dataset_manager**: DatasetInfo, list, download, format_for_finetuning
- **trainer**: TrainingConfig, _ProgressCallback, validate, register, stop
- **inference**: base_inference, finetuned_inference

### E2E Tests
- **app_loads**: 驗證首頁載入
- **model_selection**: 測試模型下拉選單
- **dataset_selection**: 測試資料集下拉選單
- **inference_tab**: 測試推理頁面

### Coverage 目標
- **全部 src/ 程式碼**: 90%+
- **報告格式**: term-missing, HTML, XML

## 5. 執行方式

```bash
# 安裝依賴
uv pip install -e ".[dev]"
playwright install --with-deps chromium

# 執行測試
make test              # 所有測試
make test-unit         # Unit tests
make test-e2e          # E2E tests
make coverage          # 覆蓋率報告
```

## 6. CI/CD

| 階段 | 內容 |
|------|------|
| Unit Tests | pytest + coverage 90% |
| Coverage | 上傳 Codecov |
| E2E Tests | Playwright + Gradio UI |
| 觸發條件 | push/PR to main |

## 7. 測試報告

| 報告類型 | 路徑 |
|----------|------|
| 終端機 | `pytest` 輸出 |
| HTML | `htmlcov/` |
| XML | `coverage.xml` |
| Codecov | GitHub PR comment |

## 8. 注意事項

- Unit tests 使用 `unittest.mock` 模擬外部依賴（HuggingFace, transformers, unsloth）
- E2E tests 需要 Gradio server 啟動，使用 `--timeout=120` 防止測試卡住
- Coverage 門檻設為 90%，未達標會失敗
- CI 使用 macOS-latest runner 以支援 MLX 後端
