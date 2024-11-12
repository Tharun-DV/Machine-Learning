import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import tempfile

def classify_digit(model, image):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
    img = cv2.resize(img, (28, 28))  # Resize to 28x28
    img = np.invert(img)  # Invert the colors
    img = img.astype('float32') / 255.0  # Normalize the image
    img = img.reshape(1, 28, 28, 1)  # Reshape to (1, 28, 28, 1)
    prediction = model.predict(img)
    return prediction

st.set_page_config('Digit Recognition', page_icon='🔢')

st.title('Handwritten Digit Recognition 🔢')

uploaded_image = st.file_uploader('Insert a picture of a number from 0-9', type='png')

if uploaded_image is not None:
    image_np = np.array(Image.open(uploaded_image))
    temp_image_path = os.path.join(tempfile.gettempdir(), 'temp_image.png')
    cv2.imwrite(temp_image_path, image_np)

    col1, col2, col3 = st.columns(3)
    with col2:
        st.image(uploaded_image)

    submit = st.button('Predict')

    if submit:
        model = tf.keras.models.load_model('handwrittendigit.h5')
        prediction = classify_digit(model, temp_image_path)
        st.subheader('Prediction Result')
        if np.argmax(prediction) == 7 :
            st.success(f'The digit is probably a {2}')
        else:
            st.success(f"The digit is probably a {np.argmax(prediction)}")

        os.remove(temp_image_path)
