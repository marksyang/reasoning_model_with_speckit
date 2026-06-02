# E2E 測試增強計畫

> **版本**: 1.0
> **建立日期**: 2026-06-02
> **狀態**: 待審閱 (Draft)
> **作者**: Claude Code

---

## 一、現況分析

| 項目 | 現況 |
|------|------|
| **測試框架** | pytest + pytest-playwright + Playwright (Chromium) |
| **CI 流程** | GitHub Actions，unit 測試與 E2E 分為兩個 job，E2E 依賴 unit 通過後執行 |
| **現有測試數** | 4 筆（全部在 `tests/e2e/test_app.py`） |
| **Fixture** | session scope 的 `gradio_server` 自動啟動/關閉伺服器 |
| **覆蓋率** | 僅驗證頁面載入及 body 存在，**實質功能覆蓋率 < 10%** |

### 現有測試清單

| # | 測試函數 | 測試內容 | 問題 |
|---|---------|---------|------|
| 1 | `test_app_loads` | 驗證頁面標題為 "Gradio" | 尚可，但應擴展 |
| 2 | `test_model_selection` | `assert page.locator("body").count() > 0` | 僅檢查 body 存在，未驗證任何實際功能 |
| 3 | `test_dataset_selection` | `assert page.locator("body").count() > 0` | 僅檢查 body 存在，未驗證任何實際功能 |
| 4 | `test_inference_tab` | `assert page.locator("body").count() > 0` | 僅檢查 body 存在，未驗證任何實際功能 |

---

## 二、問題識別

1. **測試深度不足**：3 筆測試僅檢查 `body.count() > 0`，未驗證實際功能
2. **缺少分頁測試**：Training、Evaluate 分頁完全未測試
3. **缺少互動測試**：未測試按鈕點擊、輸入表單、參數調整等操作
4. **缺少錯誤處理測試**：未驗證輸入驗證和錯誤訊息顯示
5. **缺少狀態變更測試**：未測試訓練流程中的按鈕切換、進度條更新等
6. **缺少 Timeout 防護**：CI 設定 120s timeout，但本地測試缺少 page timeout 設定

---

## 三、增強目標

| 維度 | 目標值 |
|------|--------|
| **功能覆蓋率** | 涵蓋 4 個分頁所有核心功能 (> 90%) |
| **測試總數** | 從 4 筆擴展至 23 筆 |
| **分層覆蓋** | 頁面載入 → 元件存在 → 互動操作 → 狀態變更 → 錯誤處理 |
| **執行時間** | 整體 E2E 在 CI 中 < 120 秒（維持現有限制） |

---

## 四、應用程式功能映射

應用程式包含 **4 個主要分頁**，各分頁功能如下：

### Preparation 分頁

| 功能 | 關鍵 UI 元件 | 事件處理器 |
|------|-------------|-----------|
| 模型選擇 | `model_dropdown` (Dropdown) | `on_model_select` |
| 模型資訊顯示 | `model_info` (Textbox) | — |
| 模型下載 | `download_model_btn` (Button) | `on_model_download` |
| 模型下載狀態 | `download_model_result` (Textbox) | — |
| 資料集選擇 | `dataset_dropdown` (Dropdown) | `on_dataset_select` |
| 資料集摘要 | `dataset_summary` (Textbox) | — |
| 資料集下載 | `download_dataset_btn` (Button) | `on_dataset_download` |
| 資料集下載狀態 | `download_dataset_result` (Textbox) | — |

### Inference 分頁

| 功能 | 關鍵 UI 元件 | 事件處理器 |
|------|-------------|-----------|
| 模型選擇 | `model_dropdown2` (Dropdown) | — |
| 輸入 Prompt | `prompt_textbox` (Textbox) | — |
| 最大 token 數 | `max_tokens` (Slider, 64-1024, 預設 512) | — |
| 溫度參數 | `temperature` (Slider, 0.1-1.0, 預設 0.7) | — |
| Top-p 採樣 | `top_p` (Slider, 0.1-1.0, 預設 0.9) | — |
| 是否採樣 | `do_sample` (Checkbox, 預設 True) | — |
| 執行推論 | `run_btn` (Button) | `on_run_inference` |
| 輸出顯示 | `inference_output` (Textbox) | — |

### Training 分頁

| 功能 | 關鍵 UI 元件 | 事件處理器 |
|------|-------------|-----------|
| 模型選擇 | `model_dropdown3` (Dropdown) | — |
| 資料集選擇 | `dataset_dropdown3` (Dropdown) | — |
| 訓練週期數 | `epochs` (Slider, 1-10, 預設 1) | — |
| 學習率 | `learning_rate` (Slider, 1e-5-1e-3, 預設 2e-4) | — |
| LoRA 秩 | `lora_r` (Slider, 4-32, 預設 8) | — |
| LoRA Alpha | `lora_alpha` (Slider, 4-64, 預設 16) | — |
| LoRA Dropout | `lora_dropout` (Slider, 0.01-0.1, 預設 0.05) | — |
| 開始訓練 | `train_btn` (Button) | `on_start_training` |
| 停止訓練 | `stop_btn` (Button, 初始隱藏) | `stop_training` |
| 訓練進度條 | `training_progress_bar` (Slider, 0-100) | — |
| 訓練進度訊息 | `training_progress` (Textbox) | — |
| 訓練日誌 | `training_log` (Textbox) | — |
| 註冊模型顯示 | `registered_models_dropdown` (Dropdown) | — |

### Evaluate 分頁

| 功能 | 關鍵 UI 元件 | 事件處理器 |
|------|-------------|-----------|
| 模型選擇 | `evaluate_model_dropdown` (Dropdown) | — |
| 問題 1 | `eval_prompt1` (Textbox, 含預設值) | — |
| 預期答案 1 | `eval_answer1` (Textbox, 含預設值) | — |
| 問題 2 | `eval_prompt2` (Textbox, 含預設值) | — |
| 預期答案 2 | `eval_answer2` (Textbox, 含預設值) | — |
| 問題 3 | `eval_prompt3` (Textbox, 含預設值) | — |
| 預期答案 3 | `eval_answer3` (Textbox, 含預設值) | — |
| 執行評估 | `eval_run_btn` (Button) | `on_run_evaluation` |
| 評估結果 (JSON) | `eval_result` (JSON) | — |
| 準確率顯示 | `eval_accuracy` (Textbox) | — |

---

## 五、新增測試明細

### Phase 1：基礎頁面結構測試（5 筆）

| # | 測試名稱 | 驗證內容 | 優先級 |
|---|---------|---------|-------|
| 1 | `test_tabs_exist` | 確認 4 個分頁標籤（Preparation / Inference / Training / Evaluate）存在 | P0 |
| 2 | `test_tab_switching` | 依序點擊每個分頁，確認分頁內容正確切換 | P0 |
| 3 | `test_preparation_elements` | 確認 Preparation 分頁所有元件存在（2 Dropdown, 2 按鈕, 3 個 Textbox） | P0 |
| 4 | `test_inference_elements` | 確認 Inference 分頁所有元件存在（Dropdown, 3 Slider, 1 Checkbox, 1 按鈕, 1 Textbox） | P0 |
| 5 | `test_training_elements` | 確認 Training 分頁所有元件存在（2 Dropdown, 5 Slider, 2 按鈕, 2 Textbox, 1 Slider 進度條） | P0 |

### Phase 2：Preparation 分頁功能測試（4 筆）

| # | 測試名稱 | 驗證內容 | 優先級 |
|---|---------|---------|-------|
| 6 | `test_model_dropdown_options` | 驗證 Dropdown 包含 CURATED_MODELS 全部選項（Llama-3.2-1B, Qwen2.5-0.5B 等） | P0 |
| 7 | `test_model_info_display` | 選擇模型後，model_info Textbox 應顯示 JSON 格式的模型詳情 | P0 |
| 8 | `test_dataset_dropdown_options` | 驗證 Dropdown 包含 CURATED_DATASETS 全部選項（gsm8k, squad 等） | P0 |
| 9 | `test_dataset_summary_display` | 選擇資料集後，dataset_summary 應顯示資料集資訊 | P1 |

### Phase 3：Inference 分頁功能測試（4 筆）

| # | 測試名稱 | 驗證內容 | 優先級 |
|---|---------|---------|-------|
| 10 | `test_inference_params_range` | 驗證 3 個 Slider 和 1 個 Checkbox 的預設值與範圍正確 | P0 |
| 11 | `test_inference_no_model_error` | 未選擇模型時點擊 Run Inference 應顯示錯誤 | P0 |
| 12 | `test_inference_no_prompt_error` | 已選模型但未輸入 Prompt 時應顯示錯誤 | P0 |
| 13 | `test_inference_submit` | 選擇模型並輸入 Prompt 後點擊 Run，驗證輸出區域有內容或合理狀態 | P1 |

### Phase 4：Training 分頁功能測試（5 筆）

| # | 測試名稱 | 驗證內容 | 優先級 |
|---|---------|---------|-------|
| 14 | `test_training_params_default` | 驗證 5 個訓練參數 Slider 預設值正確（epochs=1, lr=2e-4, lora_r=8, lora_alpha=16, lora_dropout=0.05） | P0 |
| 15 | `test_training_btn_initial_state` | 初始狀態：Start Training 可見，Stop Training 不可見 | P0 |
| 16 | `test_training_validation_error` | 未選擇模型或資料集時點擊 Start 應顯示錯誤 | P0 |
| 17 | `test_training_start_state_change` | 點擊 Start 後驗證按鈕狀態切換（Start 隱藏 → Stop 顯示） | P1 |
| 18 | `test_training_progress_display` | 訓練啟動後驗證進度條和日誌區域有內容更新 | P1 |

### Phase 5：Evaluate 分頁功能測試（3 筆）

| # | 測試名稱 | 驗證內容 | 優先級 |
|---|---------|---------|-------|
| 19 | `test_eval_form_defaults` | 驗證 3 組 Prompt/Answer 欄位包含預設值 | P0 |
| 20 | `test_eval_no_model_error` | 未選擇模型時點擊 Run Evaluation 應顯示錯誤 | P0 |
| 21 | `test_eval_submit` | 選擇模型後點擊 Run Evaluation，驗證結果區域有 JSON 輸出 | P1 |

### Phase 6：共用元件與整合測試（2 筆）

| # | 測試名稱 | 驗證內容 | 優先級 |
|---|---------|---------|-------|
| 22 | `test_model_dropdown_consistency` | 驗證 4 個分頁中的模型 Dropdown 選項一致 | P1 |
| 23 | `test_app_theme` | 驗證頁面標題為 "Gradio" 且套用 Soft 主題 | P2 |

---

## 六、技術實作策略

### 6.1 測試檔案結構重組

```
tests/e2e/
├── conftest.py                 # 共用 fixtures
├── test_common.py              # Phase 1 & 6: 基礎結構與共用測試
├── test_preparation.py         # Phase 2: Preparation 分頁
├── test_inference.py           # Phase 3: Inference 分頁
├── test_training.py            # Phase 4: Training 分頁
└── test_evaluate.py            # Phase 5: Evaluate 分頁
```

### 6.2 新增 Fixtures（在 `conftest.py`）

```python
# 輔助函數：導航到指定分頁
def navigate_to_tab(page, tab_name: str):
    """點擊指定分頁並等待渲染完成"""

# Fixture：載入首頁並等待 Gradio 渲染完成
@pytest.fixture
def prepared_page(page, gradio_server):
    """載入首頁並等待 Gradio 渲染完成"""

# Fixture：E2E 操作逾時時間
@pytest.fixture
def timeout_ms():
    """E2E 操作逾時時間 (30000ms)"""
```

### 6.3 元素定位策略

| 元件類型 | 定位方式 | 範例 |
|---------|---------|------|
| Tabs | `page.get_by_role("tab", name="...")` | `get_by_role("tab", name="Training")` |
| Dropdown | `page.get_by_label("Select Model")` | 依 label 文字定位 |
| Buttons | `page.get_by_role("button", name="...")` | `get_by_role("button", name="Start Training")` |
| Sliders | `page.get_by_label("Epochs")` | 依 label 文字定位 |
| Textbox | `page.get_by_label("Prompt")` | 依 label 文字定位 |
| Error | `page.locator(".error")` 或 getText 比對 | Gradio 錯誤彈窗 |

### 6.4 已知限制與應對

| 限制 | 應對方式 |
|------|---------|
| 實際訓練需要下載模型（數 GB） | 僅測試 UI 驗證層（未選模型/資料集時的錯誤），不執行真實訓練 |
| 推論需要模型已快取 | 僅測試輸入驗證，不執行真實推論 |
| Gradio 渲染有延遲 | 使用 `page.wait_for_selector()` 和 `expect().to_be_visible()` 替代固定等待 |
| CI 120s timeout | 拆分測試檔案，使用 xdist 平行執行；長時操作加 skip 標記 |

---

## 七、CI/CD 調整

### 現況

```yaml
# .github/workflows/test.yml
- name: Run E2E tests
  run: |
    pytest tests/e2e/ -v --timeout=120
```

### 建議調整

```yaml
- name: Run E2E tests
  run: |
    pytest tests/e2e/ -v --timeout=120 -n auto  # 使用 xdist 平行執行
```

> **注意**：`pytest-xdist` 已在 `pyproject.toml` 的 dev 依賴中，無需額外安裝。

---

## 八、執行時程

| 階段 | 內容 | 預估測試數 | 依賴 |
|------|------|-----------|------|
| **Phase 1** | 基礎結構 + 共用 fixtures | 5 + helpers | 無 |
| **Phase 2** | Preparation 功能 | 4 | Phase 1 |
| **Phase 3** | Inference 功能 | 4 | Phase 1 |
| **Phase 4** | Training 功能 | 5 | Phase 1 |
| **Phase 5** | Evaluate 功能 | 3 | Phase 1 |
| **Phase 6** | 整合測試 + CI 調整 | 2 | Phase 1-5 |
| **合計** | | **23 筆新測試**（取代舊 4 筆） | |

---

## 九、驗收標準

- [ ] 所有 P0 測試在 CI 中穩定通過
- [ ] 總執行時間 < 120 秒
- [ ] 無 flaky test（連續 3 次 CI 運行全數通過）
- [ ] 舊有 4 筆測試移除或重構整合
- [ ] 測試檔案按功能分頁拆分為 6 個檔案
- [ ] 共用 fixtures 收錄於 `conftest.py`

---

## 十、風險與因應

| 風險 | 影響 | 因應措施 |
|------|------|---------|
| Gradio 版本更新導致 DOM 結構變化 | 元素定位失效 | 使用 `get_by_role` 和 `get_by_label` 等語意化定位器 |
| CI 執行環境渲染差異 | 部分元件未載入 | 新增 `wait_for_selector` 等待機制，設定合理 timeout |
| 測試間相互影響 | Flaky test | 每筆測試獨立啟動頁面，不共用 session state |
| 模型/資料集清單變更 | Dropdown 選項比對失敗 | 動態讀取 CURATED_MODELS/DATASETS 鍵值進行比對，不寫死 |
