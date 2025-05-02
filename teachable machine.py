import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load the model
model = tf.keras.models.load_model('path_to_your_model_directory')

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to the input size of the model
    image = np.array(image) / 255.0  # Normalize the image
    return np.expand_dims(image, axis=0)

# Streamlit app layout
st.title("Anomaly Detection in Manufactured Products")

# Option to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess and predict
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    class_names = ['Normal', 'Anomaly']
    predicted_class = class_names[np.argmax(predictions)]

    st.write(f"Prediction: {predicted_class}")

# Bonus: Real-time camera feed
if st.button("Start Camera"):
    st.write("Starting camera...")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        frame_resized = cv2.resize(frame, (224, 224))
        frame_normalized = frame_resized / 255.0
        frame_input = np.expand_dims(frame_normalized, axis=0)

        # Make prediction
        predictions = model.predict(frame_input)
        predicted_class = class_names[np.argmax(predictions)]

        # Display the result
        cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        st.image(frame, channels="BGR")

    cap.release()