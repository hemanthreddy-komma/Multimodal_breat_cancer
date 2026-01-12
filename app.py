import streamlit as st
import numpy as np
import cv2
cv2.setNumThreads(0)
import tensorflow as tf
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries

st.set_page_config(layout="wide")

model = tf.keras.models.load_model("multimodal_model.h5")
IMG_SIZE = 224

# ---------- IMAGE PREPROCESS ----------
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img.astype(np.float32)

# ---------- GRADCAM ----------
def gradcam(img):
    last_conv = model.get_layer("block14_sepconv2_act")

    grad_model = tf.keras.models.Model(
        model.inputs,
        [last_conv.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model([img, np.zeros((1,3))])
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_out = conv_out[0]
    heatmap = np.zeros(conv_out.shape[:2], dtype=np.float32)

    for i in range(len(pooled_grads)):
        heatmap += pooled_grads[i] * conv_out[:,:,i]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-8
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    return heatmap

# ---------- LIME ----------
def lime_explain(img):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img,
        lambda x: model.predict([x, np.zeros((len(x),3))]),
        top_labels=1,
        num_samples=800
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True)
    return mark_boundaries(temp, mask)

# ---------- SHAP ----------
def shap_explain(clinical):
    explainer = shap.KernelExplainer(
        lambda x: model.predict([np.zeros((len(x),224,224,3)), x]),
        np.zeros((1,3))
    )
    return explainer.shap_values(clinical)

# ---------- UI ----------
st.title("🧬 Multimodal Breast Cancer Diagnosis")

uploaded = st.file_uploader("Upload Mammogram", type=["jpg","png"])
age = st.slider("Age", 30, 90, 45)
density = st.slider("Density", 1, 4, 2)
shape = st.slider("Mass Shape", 0, 5, 2)

if uploaded:
    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)
    proc = preprocess(img)
    clinical = np.array([[age, density, shape]])

    pred = model.predict([proc[None], clinical])[0][0]
    label = "Malignant" if pred > 0.5 else "Benign"

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Mammogram")

    with col2:
        st.metric("Diagnosis", label)
        st.metric("Cancer Probability", f"{pred:.2f}")

        if pred > 0.75:
            st.error("🚨 High malignancy — tumor likely spreading.")
        elif pred > 0.5:
            st.warning("⚠ Moderate malignancy — clinical review required.")
        else:
            st.success("🟢 Benign tissue pattern detected.")

    # ---------- GRADCAM ----------
    heat = gradcam(proc[None])
    overlay = cv2.applyColorMap(np.uint8(255*heat), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.resize(img,(224,224)),0.6,overlay,0.4,0)

    st.subheader("🔥 Tumor Localization Heatmap")
    st.image(overlay)

    # ---------- LIME ----------
    st.subheader("🧩 LIME Local Explanation")
    st.image(lime_explain(proc))

    # ---------- SHAP ----------
    st.subheader("📊 Clinical Feature Impact")
    shap_vals = shap_explain(clinical)
    st.write(dict(zip(["Age","Density","Mass Shape"], shap_vals[0][0])))
