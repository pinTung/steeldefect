# 維護筆記

給自己看的維護紀錄。專案介紹請見 [README](README.md)。

## 模型與資料存放位置

| 檔案 | 位置 |
|---|---|
| `unet_ag_shared_resnet34.keras` | Google Drive（ID 在 `app.py` 的 `MODEL_OPTIONS`） |
| `unet_ag_shared_efficientnetb4.keras` | 同上 |
| `unet_ag_shared_inceptionv3.keras` | 同上 |
| `train.csv` | Google Drive（ID 在 `app.py` 的 `load_csv()`） |

- 擁有者帳號：（請填入）
- 分享權限必須保持「知道連結的任何人都能檢視」
- **不可刪除**，否則線上 demo 會壞掉

### 更換模型連結

1. 新檔案上傳到 Google Drive，分享設為「知道連結的任何人」
2. 從 `https://drive.google.com/file/d/【檔案ID】/view` 複製 ID
3. 替換 `app.py` 中對應的 ID，commit
4. 到 share.streamlit.io →「⋮」→ **Reboot app**，讓雲端重新下載
5. 三個模型都測試過、確認正常後，才刪除舊檔案

## 雲端更新流程

1. GitHub repo → 點檔案 → 鉛筆 ✏️ → 修改 → **Commit changes**
2. Streamlit Cloud 會在一兩分鐘內自動重新部署
3. 打開 demo，上傳圖片實際測試
4. 出錯時：demo 右下角 **Manage app** 查看 log
5. 改壞了：repo → **Commits** 找到上一個正常版本，把內容改回去再 commit

⚠️ 每次 commit 都會直接上線，口試或展示前不要改。

## 版本

- 訓練環境（Colab）：TensorFlow 2.21.0 / Keras 3.13.2
- 雲端 Python 版本：share.streamlit.io → app → Settings → Advanced settings
- 套件版本：`requirements.txt`

## 本地執行常見問題

| 錯誤訊息 | 原因與解法 |
|---|---|
| `File does not exist: app.py` | 不在專案資料夾內。用 `pwd` 確認，再 `cd` 進去。 |
| `No such command 'app.py'` | 少了 `run`。正確：`python -m streamlit run app.py` |
| `No module named 'tensorflow'` | 沒有 `conda activate steel`，或環境裡沒裝套件。 |
| `Could not deserialize class 'Functional'` | Keras 版本不對。`which streamlit` 可能指向 Mac 內建的 Python 3.9，一律用 `python -m streamlit` 啟動，並安裝 `keras==3.13.2`。 |

其他注意事項：

- 專案不要放在有 iCloud 同步的「桌面」或「文件」資料夾（檔案可能只存在雲端，也會弄壞 `.git`）。建議放在 `~/Projects/`。
- GitHub token、密碼不要放在專案資料夾裡。
