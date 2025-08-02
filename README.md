Hi!Welcome！
email: chenluyu06@gmail.com

# 👁️ Eye Disease Classifier Web App

A CNN-powered tool that classifies retinal images into four categories—**Cataract, Diabetic Retinopathy, Glaucoma, and Normal**—and presents the results through an interactive **Streamlit** web interface.

> 🚧 This project is actively being developed. I'm looking for collaborators with experience in CNNs or web app development—see below!


# Why I am stuck, go to the article: Why My Eye Disease Classifier Didn’t Work—and What It Taught Me About AI

---

## 🔍 What This Project Does

- Uses pretrained **EfficientNetB0 / B5/ InceptionV3 / DenseNet201** models for image classification
- Handles preprocessed retina images with high accuracy (up to 94%)
- Deploys via **Streamlit**, with a simple UI for uploading and classifying images
- Automatically outputs predictions and optionally saves logs

---

## 🧠 Technologies Used

| Category        | Stack                          |
|----------------|---------------------------------|
| Model           | TensorFlow, Keras              |
| Frontend / App  | Streamlit                      |
| Backend logic   | Python 3.10                    |
| Image processing| Pillow, NumPy, pandas          |

---

## ✅ Current Status

- [ ] Some model training completed (EfficientNetB0 & B5)
- [x] Streamlit UI for image upload and prediction
- [x] Integrated model selection and image resizing
- [x] GitHub repo + virtual environment setup
- [ ] UI/UX improvements (WIP)
- [ ] Add feedback or logging functionality
- [ ] External dataset testing / model evaluation

---

## 🤝 Looking For Collaborators!

I'm currently seeking teammates who can help with:
- 📈 **Model performance tuning** or adding model visualization (e.g., Grad-CAM)
- 🖥 **Streamlit frontend refinement**: better UI, animations, or layout polish
- 🔧 **Testing and debugging** across OS/browser environments
- 📊 **New feature**: user feedback loop or result export

If you're excited about applying machine learning to healthcare or want a real project for your portfolio—**join me!**

---

## 🚀 Get Started

```bash
# Step 1: Clone the repo
git clone https://github.com/your-username/eye-disease-streamlit.git
cd eye-disease-streamlit

# Step 2: Create and activate environment (conda or venv)
conda create -n eye python=3.10
conda activate eye

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the app
streamlit run app.py


email: chenluyu06@gmail.com
