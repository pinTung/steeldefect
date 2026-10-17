# 鋼板瑕疵分割

[English](README.md) | 中文

以 CSAD U-Net（具共享編碼器與注意力閘門的 Attention U-Net 變體）對鋼板表面瑕疵進行像素級分割。

**線上展示：** https://steeldefect-m5jqwkguuhhhjg9gkxkvyw.streamlit.app/

![展示畫面](docs/demo.png)

## 簡介

模型輸入灰階鋼板影像，輸出 4 類瑕疵的分割遮罩。實驗比較了三種骨幹網路：ResNet34、EfficientNet-B4、InceptionV3。

## 實驗結果

泛化實驗結果，與 Aboulhosn et al.（2024）的基準方法比較。

| 骨幹網路 | 方法 | Macro DSC | Macro IoU |
|---|---|---|---|
| EfficientNet-B4 | Aboulhosn et al. (2024) | 0.7159 | 0.6977 |
| EfficientNet-B4 | **CSAD U-Net（本研究）** | **0.8347** | **0.8163** |
| ResNet34 | Aboulhosn et al. (2024) | 0.5467 | 0.5275 |
| ResNet34 | **CSAD U-Net（本研究）** | **0.7270** | **0.7083** |
| InceptionV3 | Aboulhosn et al. (2024) | 0.7468 | 0.7278 |
| InceptionV3 | **CSAD U-Net（本研究）** | **0.8458** | **0.8270** |

<details>
<summary>逐類別 DSC</summary>

| 骨幹網路 | 方法 | Defect 1 | Defect 2 | Defect 3 | Defect 4 |
|---|---|---|---|---|---|
| EfficientNet-B4 | Aboulhosn et al. (2024) | 0.6736 | 0.7735 | 0.6230 | 0.7934 |
| EfficientNet-B4 | CSAD U-Net（本研究） | 0.8077 | 0.9172 | 0.6925 | 0.9215 |
| ResNet34 | Aboulhosn et al. (2024) | 0.3436 | 0.6103 | 0.4664 | 0.7662 |
| ResNet34 | CSAD U-Net（本研究） | 0.6508 | 0.8512 | 0.5427 | 0.8634 |
| InceptionV3 | Aboulhosn et al. (2024) | 0.6520 | 0.7736 | 0.6751 | 0.8865 |
| InceptionV3 | CSAD U-Net（本研究） | 0.8464 | 0.9328 | 0.6685 | 0.9358 |

</details>

<details>
<summary>逐類別 IoU</summary>

| 骨幹網路 | 方法 | Defect 1 | Defect 2 | Defect 3 | Defect 4 |
|---|---|---|---|---|---|
| EfficientNet-B4 | Aboulhosn et al. (2024) | 0.6630 | 0.7706 | 0.5717 | 0.7856 |
| EfficientNet-B4 | CSAD U-Net（本研究） | 0.7970 | 0.9144 | 0.6406 | 0.9133 |
| ResNet34 | Aboulhosn et al. (2024) | 0.3326 | 0.6075 | 0.4121 | 0.7576 |
| ResNet34 | CSAD U-Net（本研究） | 0.6405 | 0.8484 | 0.4894 | 0.8550 |
| InceptionV3 | Aboulhosn et al. (2024) | 0.6410 | 0.7708 | 0.6214 | 0.8780 |
| InceptionV3 | CSAD U-Net（本研究） | 0.8361 | 0.9300 | 0.6148 | 0.9271 |

</details>

DSC 與 IoU 皆只計算有 Ground Truth 標註的類別，並取平均（macro average）。

## 資料集

[Severstal: Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection)（Kaggle）：4 類瑕疵，影像 1600 × 256（訓練時縮放為 800 × 128），像素級 RLE 標註。本 repository 不包含資料集。

## 本地執行

```bash
conda create -n steel python=3.11 -y
conda activate steel
pip install -r requirements.txt
python -m streamlit run app.py
```

模型以 **TensorFlow 2.21.0 / Keras 3.13.2** 訓練，其他版本的 Keras 可能無法載入。模型權重放在 Google Drive，第一次使用時會自動下載。
