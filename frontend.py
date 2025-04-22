import streamlit as st
import numpy as np
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession("student_model.onnx")

st.title("🌾 Irrigation Scheduling Predictor")
st.write("Enter sensor readings below to predict irrigation class.")

# Define inputs
feature_names = ["temperature", "humidity", "soil_moisture", "altitude", "rainfall", "wind_speed"]
class_labels = ["very dry", "dry", "wet", "very wet"]  # Class labels

inputs = []
for name in feature_names:
    value = st.number_input(f"{name}", value=0.0)
    inputs.append(value)

if st.button("Predict"):
    features = np.array([inputs], dtype=np.float32)
    output = session.run(["output"], {"input": features})
    predicted_class = int(np.argmax(output[0]))
    class_label = class_labels[predicted_class]
    st.success(f"Predicted Irrigation Class: {class_label}")
