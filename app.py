import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Define your class labels (same as used during training)
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Load the model
model = tf.keras.models.load_model("model_resnet.h5")

# Preprocess function
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# Streamlit UI
st.title("🌼 Flower Classifier")
st.write("Upload a flower image and see the prediction.")

file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_data = preprocess_image(image)
    predictions = model.predict(input_data)
    
    confidence = np.max(predictions)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]

    st.markdown(f"### 🌸 Prediction: **{predicted_class}**")
    st.markdown(f"### 🔍 Confidence: **{confidence:.2f}**")

    # Show top 3 probabilities (optional)
    st.markdown("#### Top 3 Predictions:")
    top_indices = predictions[0].argsort()[-3:][::-1]
    for i in top_indices:
        st.write(f"{CLASS_NAMES[i]}: {predictions[0][i]:.2f}")
