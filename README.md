# 思考模型微調示範應用程式

基於 Unsloth 的思考模型微調本地 Gradio 網頁應用程式，支援 Apple Silicon（MLX）與 NVIDIA GPU（PyTorch）。可從 Hugging Face Hub 選擇推理模型與資料集，執行微調前後推論，並觀察即時訓練進度。

## 功能特色

- **模型下載**：精選的推理模型清單，支援自動下載與快取
- **資料集下載**：預先選取公開推理資料集，支援串流下載與快取檢測
- **基礎模型推論**：以可調整參數（temperature、top-p、max tokens）執行模型推論
- **LoRA 微調**：Unsloth 驱动的 4-bit 微調，具備即時進度列、串流日誌與停止訓練功能
- **模型評估**：在驗證範例上評估微調後模型的準確率

## 系統需求

- **作業系統**：macOS（Apple Silicon，MLX 後端）或 Ubuntu 22.04+ 搭配 NVIDIA GPU（PyTorch 後端）
- **GPU**：NVIDIA GPU ≥12GB VRAM 或 Apple Silicon ≥16GB 共用記憶體
- **Python**：3.10+（Unsloth 不支援 3.14+）
- **Git**
- **uv**（建議安裝，加速套件安裝）

## 快速開始

```bash
# 1. 克隆專案
git clone https://github.com/marksyang/reasoning_model_with_speckit.git
cd reasoning_model_with_speckit

# 2. 建立虛擬環境
uv venv .venv --python 3.10
source .venv/bin/activate

# 3. 安裝相依套件
uv pip install -r requirements.txt

# 4. 啟動應用程式
python -m src.app
# -> 開啟 http://127.0.0.1:7860
```

## 專案結構

```
.
|-- src/
|   |-- app.py             # Gradio Blocks UI 與事件處理器
|   |-- model_manager.py   # 模型下載、快取、推論
|   |-- dataset_manager.py # 資料集下載與微調格式轉換
|   |-- trainer.py         # Unsloth QLoRA 訓練（MLX/PyTorch 後端）
|   `-- inference.py       # 基礎模型與微調模型推論
`-- data/
    |-- app_state.json     # 應用程式狀態（模型、資料集、訓練記錄）
    |-- adapters/          # 微調 LoRA 適配器（產生）
    `-- logs/              # 訓練日誌（產生）
```

## 使用指南

### 1. 準備分頁

選擇模型與資料集後下載：

**模型**（適合 12GB+ VRAM）：

| 模型 | 參數 | 4-bit VRAM | 備註 |
|-------|--------|------------|-------|
| Qwen 2.5 0.5B | 500M | ~1 GB | 速度最快，適合示範 |
| Qwen 2.5 1.5B | 1.5B | ~1.5 GB | 效能平衡 |
| LLaMA 3.2 1B | 1B | ~2 GB | 推理能力強 |
| LLaMA 3.1 8B | 8B | ~6 GB | 品質最佳（需梯度檢查點） |

**資料集**：

| 資料集 | 說明 | 用途 |
|---------|-------------|--------|
| GSM8K | 小學數學應用題 | 數學推理（建議） |
| SQuAD | 史丹佛問答資料集 | 問答任務 |
| CoT GSM8K | 附思考鏈的 GSM8K | 思考鏈訓練 |
| No Robots | 人手寫指令資料 | 指令微調 |
| OpenAssistant | 多輪對話資料集 | 對話 |
| UltraChat 200K | 高品質多輪對話 | 大規模對話 |

### 2. 推論分頁

選擇已下載的模型，輸入推理提示（如數學應用題），調整生成參數後執行推論。輸出會即時逐 Token 串流顯示。

### 3. 訓練分頁

1. 選擇模型與資料集
2. 設定 LoRA 參數（r、alpha、dropout）
3. 設定訓練超參數（epochs、學習率）
4. 點擊「開始訓練」
5. 觀看進度列、串流日誌與迭代次數
6. 點擊「停止訓練」可中斷訓練
7. 完成後，微調模型會自動註冊至下拉選單

**建議參數**：
- `LoRA Rank`：8
- `LoRA Alpha`：16
- `LoRA Dropout`：0.05
- `Epochs`：1（快速示範）
- `學習率`：2e-4

### 4. 評估分頁

1. 選擇模型（基礎或微調）
2. 輸入 3 組提示與答案對（已預填數學範例）
3. 點擊「執行評估」
4. 查看各題準確率與整體準確率

## 架構設計

### 後端選擇

應用程式使用 Unsloth 進行最佳化的模型載入與微調。Unsloth 會自動選擇適當的後端：

| 平台 | 後端 | 框架 | 訓練引擎 |
|----------|---------|-----------|------------------|
| macOS (Apple Silicon) | MLX | `mlx_lm` | `mlx_lm.lora.train_model` |
| Linux (NVIDIA GPU) | PyTorch | `transformers` | `trl.SFTTrainer` / `Trainer` |

### 進度串流

訓練進度透過基於佇列的回呼模式從 MLX/PyTorch 訓練迴路串流至 Gradio UI：

```
訓練迴路 -> Callback.on_train_loss_report()
  -> queue.Queue.put(progress_dict)
  -> train() generator yields from queue
  -> Gradio handler yields tuple to (progress_bar, progress_textbox, log_textbox, dropdown, buttons)
```

### 資料集格式

所有資料集使用完整的 HuggingFace 命名空間（如 `openai/gsm8k` 而非 `gsm8k`），並透過 `config` 參數處理多配置資料集。資料格式化為 Alpaca 樣式提示：

```
### Question:
{question}

### Answer:
{answer}
```

### 適配器儲存

微調的 LoRA 適配器存為 `data/adapters/{adapter_id}/adapters.safetensors`。適配器中繼資料會註冊至 `data/app_state.json`，確保應用程式重新啟動後仍可存取。

## 疑難排解

| 問題 | 解決方案 |
|---------|----------|
| `cuda:0 out of memory` | 選擇較小模型（0.5B 而非 8B），或降低 LoRA rank |
| `HF Hub download fails` | 確認網路連線；中國大陸可設定 `HF_ENDPOINT=https://hf-mirror.com` |
| `Unsloth import error` | 確認 Python 版本為 3.10-3.13（非 3.14+） |
| `ImportError: cannot import name 'FastLanguageModel'` | 執行 `pip install --upgrade unsloth` 升級 |
| `ModuleNotFoundError: No module named 'mlx'` | 安裝 MLX：`pip install mlx mlx-lm`（僅 macOS） |
| Gradio 顯示 "Connection refused" | 確認通訊埠 7860 未被佔用；嘗試 `python -m src.app --port 7861` |
| 訓練顯示 `local variable referenced before assignment` | 近幾版已修復，請拉取最新程式碼 |

## 授權

MIT
