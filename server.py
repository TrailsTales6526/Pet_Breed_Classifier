import streamlit as st
from fastai.vision.all import *
import urllib.request

st.title("Pet Classifier")
st.text("Built by David Prasetyo (Github TrailsTales6526)")

model = load_learner("pet_classification.pkl")

uploaded_file= st.file_uploader("Choose an image...", type=["png", "jpeg", "jpg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded File", use_column_width=True)
    
    if st.button("Predict") == True:
        image = PILImage.create(uploaded_file)
        guess, _, _ = model.predict(image)
        st.write(guess)