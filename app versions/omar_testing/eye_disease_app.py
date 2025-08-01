# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import os
from datetime import datetime

# -------------------------------
# SETUP
# -------------------------------
st.set_page_config(
    page_title="Eye Disease Classifier",
    page_icon="🧿",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------------------
# MODEL LOADING
# -------------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("EfficientNet B5.keras",compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# -------------------------------
# CATEGORY LABELS
# -------------------------------
categories = {
    0: "Cataract",
    1: "Diabetic Retinopathy",
    2: "Glaucoma",
    3: "Normal",
}

# -------------------------------
# UI HEADER & SIDEBAR
# -------------------------------
st.markdown("<h1 style='text-align: center;'>🧿 Eye Disease Classifier</h1>", unsafe_allow_html=True)
st.write("Upload an eye image (JPEG or PNG). The model will analyze it and predict the condition.")

with st.sidebar:
    st.title("About")
    st.markdown(
        """
        This tool uses a deep learning model to classify eye images into:
        - Cataract  
        - Diabetic Retinopathy  
        - Glaucoma  
        - Normal  

        **Upload a retinal scan** to get started.
        """
    )

# -------------------------------
# IMAGE UPLOAD & PREDICTION
# -------------------------------
uploaded_file = st.file_uploader("Choose an eye image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Save image with timestamped filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_name = f"{timestamp}_{uploaded_file.name}"
    os.makedirs("feedback_images", exist_ok=True)
    image_path = os.path.join("feedback_images", image_name)
    image.save(image_path)

    # Preprocess for prediction
    img = image.resize((456, 456))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    try:
        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction))
        confidence = float(prediction[0][predicted_class])

        st.markdown("---")
        st.subheader("🧠 Model Prediction")
        st.success(f"**{categories[predicted_class]}** ({confidence * 100:.2f}% confidence)")

        st.markdown("### 📊 Class Probabilities")
        for i, prob in enumerate(prediction[0]):
            st.write(f"{categories[i]} — {prob * 100:.2f}%")
            st.progress(float(prob))

        # -------------------------------
        # FEEDBACK SECTION
        # -------------------------------
        st.markdown("## 📝 Was this prediction helpful?")
        feedback = st.radio("How accurate was the prediction?", ["Correct", "Incorrect", "Not sure"])
        comment = st.text_area("Optional comment or suggestion:")

        if st.button("Submit Feedback"):
            entry = {
                "timestamp": datetime.now().isoformat(),
                "image_name": image_name,
                "prediction": categories[predicted_class],
                "confidence": confidence,
                "feedback": feedback,
                "comment": comment
            }

            file_path = "user_feedback.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
            else:
                df = pd.DataFrame(columns=entry.keys())

            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
            df.to_csv(file_path, index=False)

            st.success("✅ Thank you for your feedback! Image and data saved.")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# -------------------------------
# VIEW FEEDBACK AND IMAGES (OWNER ONLY)
# -------------------------------
st.markdown("---")
st.markdown("## 🗂️ Collected Feedback")

if os.path.exists("user_feedback.csv"):
    # Force image_name column to string to avoid float/NaN issues
    df = pd.read_csv("user_feedback.csv", dtype={"image_name": str})

    with st.expander("📄 View Feedback Table", expanded=False):
        st.dataframe(df)

    # Owner-only image access
    with st.expander("📷 Owner-only: View submitted images"):
        password = st.text_input("Enter owner password to view images:", type="password")

        if password == "mysecret123":  # 🔒 Replace with your actual password
            for i, row in df.iterrows():
                image_name = str(row["image_name"]) if not pd.isna(row["image_name"]) else None

                if image_name:
                    image_path = os.path.join("feedback_images", image_name)
                    if os.path.exists(image_path):
                        st.image(
                            image_path,
                            caption=f"{image_name} | Prediction: {row['prediction']} | Feedback: {row['feedback']}",
                            use_column_width=False,
                            width=300
                        )
                    else:
                        st.warning(f"⚠️ Image not found: {image_name}")
        elif password:
            st.error("❌ Incorrect password.")
else:
    st.info("No feedback submitted yet.")






