import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import keras
import os
import pandas as pd
import gdown

IMG_H      = 128
IMG_W      = 800
THRESHOLD  = 0.5
CSV_PATH   = "train.csv"

MODEL_OPTIONS = {
    "Attention UNet + ResNet34":       ("unet_ag_shared_resnet34.keras",       "1p-Ycq000psGXTwm2OyjvrrxRbi3ofU2W"),
    "Attention UNet + EfficientNetB4": ("unet_ag_shared_efficientnetb4.keras", "1eeODbUSENKwnWwUZm5PHNGfb6FJ6YvWP"),
    "Attention UNet + InceptionV3":    ("unet_ag_shared_inceptionv3.keras",    "1sCuswfRLKVvsrI9AzvJuxWiQ4o2ZRNxq"),
}

CLASS_NAMES  = ["Class 1", "Class 2", "Class 3", "Class 4"]
CLASS_COLORS = [
    (255,  80,  80),
    ( 80, 200,  80),
    ( 80, 120, 255),
    (255, 200,  50),
]

def rle_decode(mask_rle, shape=(256, 1600)):
    if not isinstance(mask_rle, str) or not mask_rle.strip():
        return np.zeros(shape, dtype=np.uint8)
    s = mask_rle.strip().split()
    starts  = np.asarray(s[0::2], dtype=int) - 1
    lengths = np.asarray(s[1::2], dtype=int)
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, starts + lengths):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")

@st.cache_data
def load_csv():
    if not os.path.exists(CSV_PATH):
        return None
    df = pd.read_csv(CSV_PATH)
    if "ImageId_ClassId" in df.columns:
        df["ImageId"] = df["ImageId_ClassId"].str.rsplit("_", n=1).str[0]
        df["ClassId"] = df["ImageId_ClassId"].str.rsplit("_", n=1).str[1].astype(int)
    return df

@st.cache_resource
def load_model(model_name):
    filename, file_id = MODEL_OPTIONS[model_name]
    if not os.path.exists(filename):
        with st.spinner(f"從 Google Drive 下載 {filename}（約 700MB，請稍候）..."):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, filename, quiet=False)
    return keras.models.load_model(filename, compile=False)

def preprocess(img_u8):
    if img_u8.ndim == 3 and img_u8.shape[-1] == 4:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGBA2GRAY)
    elif img_u8.ndim == 3 and img_u8.shape[-1] == 3:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img_u8, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    img = np.stack([img, img, img], axis=-1).astype("float32")
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img[np.newaxis, ...]

def get_gt_masks(img_id, df, orig_h, orig_w):
    masks = []
    for cls in range(1, 5):
        row = df[(df["ImageId"] == img_id) & (df["ClassId"] == cls)]
        if len(row) > 0:
            rle = row.iloc[0]["EncodedPixels"]
            m = rle_decode(rle, shape=(256, 1600))
            m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        else:
            m = np.zeros((orig_h, orig_w), dtype=np.uint8)
        masks.append(m)
    return masks

def make_overlay(orig_rgb, masks):
    overlay = orig_rgb.copy().astype(np.float32)
    for i, m in enumerate(masks):
        color = np.array(CLASS_COLORS[i], dtype=np.float32)
        overlay[m == 1] = overlay[m == 1] * 0.4 + color * 0.6
    return overlay.clip(0, 255).astype(np.uint8)

def dice_iou(pred, gt):
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    inter = (pred & gt).sum()
    dice  = (2 * inter) / (pred.sum() + gt.sum() + 1e-6)
    iou   = inter / ((pred | gt).sum() + 1e-6)
    return float(dice), float(iou)

def macro_dice_iou(pred_masks, gt_masks):
    dices, ious = [], []
    for p, g in zip(pred_masks, gt_masks):
        if g.sum() > 0:
            d, u = dice_iou(p, g)
            dices.append(d)
            ious.append(u)
    if not dices:
        return 0.0, 0.0
    return float(np.mean(dices)), float(np.mean(ious))

# ── UI ───────────────────────────────────────────────────────────
st.set_page_config(page_title="鋼板瑕疵分割 Demo", layout="wide")

st.markdown("""
<div style='text-align:center;padding:14px 0 6px;margin-bottom:16px'>
<span style='font-size:1.4rem;font-weight:bold;letter-spacing:3px'>
應用展示 WEB DEMO（STREAMLIT）</span>
</div>
""", unsafe_allow_html=True)

st.divider()

left, right = st.columns([1, 1.6])

with left:
    selected_model_name = st.selectbox("選擇模型", list(MODEL_OPTIONS.keys()))
    uploaded  = st.file_uploader("上傳鋼板圖片（JPG / PNG）", type=["jpg","jpeg","png"])
    threshold = st.slider("Mask 閾值", 0.1, 0.9, THRESHOLD, 0.05)

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        orig_bgr   = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        orig_rgb   = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = orig_rgb.shape[:2]

        img_id = uploaded.name
        df = load_csv()
        has_gt = df is not None and img_id in df["ImageId"].values

        st.markdown(f"<small><b>圖片 ID：</b> {img_id}</small>", unsafe_allow_html=True)
        if has_gt:
            st.success("✅ 找到 GT 標籤")
        else:
            st.info("ℹ️ 此圖無 GT（測試集）")

        st.markdown("**原圖**")
        st.image(orig_rgb, use_container_width=True)

        if has_gt:
            gt_masks   = get_gt_masks(img_id, df, orig_h, orig_w)
            gt_overlay = make_overlay(orig_rgb, gt_masks)
            st.markdown("**GT 標籤**")
            st.image(gt_overlay, use_container_width=True)

with right:
    if uploaded:
        with st.spinner(f"推論中（{selected_model_name}）..."):
            model = load_model(selected_model_name)
            inp   = preprocess(orig_rgb)
            preds = model.predict(inp, verbose=0)

        pred_masks = []
        for p in preds:
            m = (p[0, :, :, 0] > threshold).astype(np.uint8)
            m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            pred_masks.append(m)

        pred_overlay = make_overlay(orig_rgb, pred_masks)
        st.markdown("**模型預測結果**")
        st.image(pred_overlay, use_container_width=True)
        st.caption(f"Predicted by {selected_model_name}")

        mask_cols = st.columns(4)
        for i, m in enumerate(pred_masks):
            vis = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            vis[m == 1] = CLASS_COLORS[i]
            hex_c = "#{:02x}{:02x}{:02x}".format(*CLASS_COLORS[i])
            mask_cols[i].image(vis, use_container_width=True)
            mask_cols[i].markdown(
                f'<small><span style="background:{hex_c};color:white;padding:1px 6px;'
                f'border-radius:3px">{CLASS_NAMES[i]}</span></small>',
                unsafe_allow_html=True)

        if has_gt:
            st.divider()
            macro_d, macro_u = macro_dice_iou(pred_masks, gt_masks)
            mc1, mc2 = st.columns(2)
            mc1.metric("Dice Score", f"{macro_d:.4f}")
            mc2.metric("IoU",        f"{macro_u:.4f}")
            st.caption("※ 僅計算有 GT 標籤的類別平均")
        else:
            found = [CLASS_NAMES[i] for i, m in enumerate(pred_masks) if m.sum() > 0]
            if found:
                st.error(f"⚠️ 偵測到瑕疵：{', '.join(found)}")
            else:
                st.success("✅ 未偵測到瑕疵")