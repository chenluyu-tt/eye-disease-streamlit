import streamlit as st

st.markdown("""
### 👁️ About the Project

This tool uses deep learning to classify eye images into four categories:  
**Cataract**, **Diabetic Retinopathy**, **Glaucoma**, and **Normal** — all from a single fundus photo.  
It's like a mini ophthalmologist in your browser (minus the white coat and laser pointer).

But we’re not stopping there.  
We want this website to grow into a space where students, researchers, and curious humans can explore medical AI hands-on.  
We're inviting volunteers to upload more eye images so we can diversify our dataset and improve model performance for all kinds of eyes — not just the ones that show up in textbook examples.

The more diverse the data, the smarter the model!

In the long term, we hope our work supports real clinicians with early detection tools —  
because good vision shouldn’t be a luxury, and AI shouldn't be a black box.
""")

st.markdown("""
### 📌 What We Did
- Built and trained a CNN using EfficientNetB5, ResNet50, DenseNet201
- Achieved ~91% validation accuracy
- Deployed a real-time web app for public use
""")