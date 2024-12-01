import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

def load_model():
    model = tf.keras.models.load_model('my_tumor_model.keras')
    return model

model = load_model()

st.write("""
         # Brain Tumor Classification
         """)

file = st.file_uploader('Please upload the tumor image', type=['jpg', 'png'])

def image_and_prediction(image_data, model):
    size = (256, 256)
    processed_image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.array(processed_image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)

    return prediction

if file is None:
    st.text("Please Upload an Image File")
else:
    uploaded_image = Image.open(file)
    st.image(uploaded_image, use_column_width=True)
    prediction = image_and_prediction(uploaded_image, model)
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    result_string = f"This Image Most Likely Is: {class_names[np.argmax(prediction)]}"
    st.success(result_string)
