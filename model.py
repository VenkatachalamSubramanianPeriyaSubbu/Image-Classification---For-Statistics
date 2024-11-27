import streamlit as st
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input

model = load_model("Final Model/vgg_model.h5")

def classify_image(image):
    img = load_img(image, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class = np.where(predictions[0]>0.8)[0]

    class_names = ["PDF", "CDF", "Histogram"]
    predicted_class_name = [class_names[i] for i in predicted_class]

    return predicted_class_name, predictions

st.title("Image Classification App")
st.write("Upload an image to classify it as PDF, CDF, or Histogram.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")

    with st.spinner('Classifying...'):
        predicted_classes, pred = classify_image(uploaded_file)

    if predicted_classes:
        st.success("Predicted Class(es):")
        for cls in predicted_classes:
            st.write(f"- {cls}")
            st.write(pred)
    else:
        st.success("Predicted Class(es):")
        st.write("None")