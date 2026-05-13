import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)
MODEL_PATH = "ai_image_resnet50_model.keras"

st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #111827 50%, #020617 100%);
        color: #f8fafc;
    }

    .main-container {
        padding: 2rem 3rem;
    }

    .hero {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.8rem;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #cbd5e1;
        max-width: 800px;
        margin: auto;
        line-height: 1.7;
    }

    .card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.35);
        margin-top: 1rem;
    }

    .result-real {
        background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(22,163,74,0.08));
        border: 1px solid rgba(34,197,94,0.4);
        padding: 1.4rem;
        border-radius: 18px;
        text-align: center;
    }

    .result-ai {
        background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(220,38,38,0.08));
        border: 1px solid rgba(239,68,68,0.4);
        padding: 1.4rem;
        border-radius: 18px;
        text-align: center;
    }

    .result-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .confidence {
        font-size: 2.5rem;
        font-weight: 900;
        margin-top: 0.5rem;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .feature-box {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 16px;
        padding: 1rem;
        height: 100%;
    }

    .feature-title {
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0.3rem;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 2rem 0 1rem 0;
        font-size: 0.85rem;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.7);
        border: 1px dashed rgba(148, 163, 184, 0.4);
        border-radius: 18px;
        padding: 1rem;
    }

    .stButton > button {
        width: 100%;
        border-radius: 14px;
        border: none;
        padding: 0.8rem 1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #6d28d9);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Model Loading
# -----------------------------
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_trained_model()
except Exception as e:
    st.error("Model file not found or could not be loaded.")
    st.code(str(e))
    st.stop()


# -----------------------------
# Header / Hero Section
# -----------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">AI-Generated Image Detection</div>
    <div class="hero-subtitle">
        A ResNet50 transfer learning system that analyzes an uploaded image and predicts whether it is 
        <b>Real</b> or <b>AI-Generated</b>. Built with TensorFlow, Keras, and Streamlit.
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Feature Cards
# -----------------------------
f1, f2, f3 = st.columns(3)

with f1:
    st.markdown("""
    <div class="feature-box">
        <div class="feature-title">🧠 ResNet50 CNN</div>
        <div class="small-muted">Uses transfer learning with a pretrained ResNet50 model for image feature extraction.</div>
    </div>
    """, unsafe_allow_html=True)

with f2:
    st.markdown("""
    <div class="feature-box">
        <div class="feature-title">⚡ Fast Prediction</div>
        <div class="small-muted">Upload a JPG or PNG image and get an instant prediction with confidence score.</div>
    </div>
    """, unsafe_allow_html=True)

with f3:
    st.markdown("""
    <div class="feature-box">
        <div class="feature-title">📊 Demo Accuracy</div>
        <div class="small-muted">Trained on real and AI-generated image samples with around 89% validation accuracy.</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------
# Main App Layout
# -----------------------------
left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📤 Upload Image")
    st.write("Choose an image file to analyze.")

    uploaded_file = st.file_uploader(
        "Upload JPG, JPEG, or PNG",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div class="small-muted">
        Recommended: use clear images for better predictions. This demo model may be less reliable on edited,
        compressed, screenshot, or out-of-distribution images.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 Prediction Panel")

    if uploaded_file is None:
        st.info("Upload an image to start detection.")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img = image.resize(IMG_SIZE)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]

        # Class order:
        # ai_generated = 0
        # real = 1
        if prediction >= 0.5:
            label = "Real Image"
            confidence = prediction * 100
            result_class = "result-real"
            emoji = "✅"
        else:
            label = "AI-Generated Image"
            confidence = (1 - prediction) * 100
            result_class = "result-ai"
            emoji = "⚠️"

        st.markdown(f"""
        <div class="{result_class}">
            <div class="result-title">{emoji} {label}</div>
            <div class="small-muted">Model confidence</div>
            <div class="confidence">{confidence:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        
        st.progress(int(confidence))

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# About Section
# -----------------------------
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="card">
    <h3>📌 About This Project</h3>
    <p class="small-muted">
        This project uses a ResNet50-based convolutional neural network with transfer learning.
        The base ResNet50 layers are frozen, and a custom classifier head is trained for binary classification:
        <b>Real Image</b> vs <b>AI-Generated Image</b>.
    </p>
    <p class="small-muted">
        Disclaimer: This is a demo-level academic project. Predictions may be inaccurate for images outside the
        training distribution, such as high-resolution phone images, screenshots, heavily edited images, or images
        from sources not represented in the training dataset.
    </p>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="footer">
    Built by Mihir Shah | TensorFlow • ResNet50 • Streamlit
</div>
""", unsafe_allow_html=True)