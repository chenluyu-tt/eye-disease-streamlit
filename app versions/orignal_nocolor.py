#This one is the original one without color app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Eye Disease Classifier",
    page_icon="🧿",
    layout="centered",
    initial_sidebar_state="expanded"
)
# ----- Main Header -----
st.markdown("<h1 style='text-align: center;'>🧿 Eye Disease Classifier</h1>", unsafe_allow_html=True)
st.write("Upload an eye image (JPEG or PNG). The model will analyze it and predict the condition.")

# ----- Load Model with Error Handling -----
@st.cache_resource
def load_model():
    try:
        st.write("📦 Loading model...")
        model = tf.keras.models.load_model("Functional_DenseNet201.keras")
        st.success(f"✅ Model loaded!This is the model: {model}")
        return model
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# ----- Sidebar -----
with st.sidebar:
    st.title(" About")
    st.markdown(
        """
        This tool uses a deep learning model to classify eye images into:
        -  Cataract  
        -  Diabetic Retinopathy  
        -  Glaucoma  
        -  Normal

        **Upload a retinal scan** to get started.
        """
    )



# ----- Category Mapping -----
categories = {
    0: "Cataract",
    1: "Diabetic Retinopathy",
    2: "Glaucoma",
    3: "Normal",
}

# ----- Upload and Display -----
uploaded_file = st.file_uploader("Choose an eye image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((456, 456))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    try:
        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction))
        confidence = prediction[0][predicted_class]

        # ----- Display Result -----
        st.markdown("---")
        st.subheader("🧠 Model Prediction")
        st.success(f"**{categories[predicted_class]}** ({confidence * 100:.2f}% confidence)")

        st.markdown("### 📊 Class Probabilities")
        for i, prob in enumerate(prediction[0]):
            st.write(f"{categories[i]} — {prob*100:.2f}%")
            st.progress(float(prob))
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
else:
    st.info("👆 Please upload an image to start diagnosis.")