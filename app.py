import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ----------------------------
# Sidebar Information
# ----------------------------
st.sidebar.title("About the App")
st.sidebar.markdown("""
This is a **Solar Panel Condition Classifier** powered by a deep learning model.

Upload an image of a solar panel, and the model will classify its condition as one of the following:

- ✅ Clean  
- 🕊️ Bird-drop  
- 🌫️ Dusty  
- ⚡ Electrical-damage  
- 🔨 Physical-Damage  
- ❄️ Snow-Covered

This can help identify potential issues and improve solar panel maintenance and efficiency.
""")

# ----------------------------
# Load the Model
# ----------------------------
model = tf.keras.models.load_model("solar_panel_classifier.keras")

# Class names (must match training)
class_names = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

# ----------------------------
# App Main Area
# ----------------------------
st.title("🔍 Solar Panel Condition Classifier")
st.write("Upload an image of a solar panel to detect its condition using a trained deep learning model.")

# Upload image
uploaded_file = st.file_uploader("Choose a solar panel image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict
    try:
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = class_names[class_index]
        confidence = float(np.max(prediction)) * 100

        st.success(f"**Prediction:** {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
else:
    st.info("Please upload an image file to begin.")
