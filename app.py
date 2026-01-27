import keras.saving
import streamlit as st
import os
import random
import glob
import numpy as np
from pathlib import Path
from recogniser import Recogniser

loaded_model = keras.saving.load_model("model/audio_model.keras")

# Configure page
st.set_page_config(page_title="Speaker Recognition Demo", layout="wide")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This application demonstrates a speaker-dependent name recognition system using deep learning.
    
    **Dataset:**
    - 400 audio samples (20 names × 20 samples each)
    - 40 MFCC features extracted per sample
    
    **How to use:**
    1. Listen to the sample audio files
    2. Click "Predict" to identify the spoken name
    3. View the prediction with confidence score
    
    **Technology Stack:**
    - **Librosa** - Audio feature extraction (MFCCs)
    - **Keras/TensorFlow** - CNN model training & inference
    - **Streamlit** - Web interface
    """)

    st.markdown("---")
    st.caption("Speaker Recognition Demo v1.0")

# Title and description
st.title("🎙️ Speaker-Dependent Name Recognition")
st.markdown("Listen to audio samples and test the recognition model")

# Audio files directory
AUDIO_DIR = Path("test_audio_files")


def get_audio_data():
    AUDIO_FILES = {}
    list_of_files = sorted(glob.glob(('test_audio_files/*.wav')))
    files_length = len(list_of_files)
    start = 0
    while start < files_length:
        file = list_of_files[start]
        split = file.split('/')
        file_name = split[len(split)-1]
        AUDIO_FILES[file_name] = file
        start = start + 1
    return AUDIO_FILES
AUDIO_FILES = get_audio_data()

# Create columns for layout
st.markdown("---")
st.subheader("Audio Samples")


def predict_speaker(audio_file, name):
    """
    Triggers the speaker recognition model & makes a prediction.

    Args:
        audio_file: Uploaded audio file

    Returns:
        str: Predicted speaker name
    """
    classes = Recogniser.get_classes()

    features = Recogniser.extract_features(audio_file, 100)
    features = np.array(features, dtype="float32")

    max = np.max(features)
    if max != 0:
        features = features / max

    features = np.expand_dims(features, axis=-1)
    features = np.expand_dims(features, axis=0)

    prediction = loaded_model.predict(features)
    predicted_id = np.argmax(prediction, axis=1)
    predicted_value = classes[predicted_id[0]]
    confidence = np.max(prediction)

    st.session_state['predicted_value'] = predicted_value
    st.session_state['actual_value'] = name
    st.session_state['confidence'] = confidence


if "predicted_value" not in st.session_state:
    st.session_state.predicted_value = None
    st.session_state.actual_value = None
    st.session_state.confidence = None

# Display Result
# confusion_matrix = {
#     "Predicted Value": [st.session_state.predicted_value],
#     "Actual Value": [st.session_state.actual_value],
# }
# container = st.container(width='content', height='content', horizontal_alignment="center")
# container.table(confusion_matrix, )

# Display audio files in a grid
cols = st.columns(2)
for idx, (name, audio_file) in enumerate(AUDIO_FILES.items()):
    col = cols[idx % 2]
    with col:
        with st.container(border=True):
            with st.container(horizontal=True):
                st.markdown(f"**{name}**")
                st.space("stretch")
                st.button("Predict", icon="🚀", key=f"{name}_right", on_click=predict_speaker, args=(audio_file,name))
            prediction = st.session_state.predicted_value
            actual = st.session_state.actual_value
            confidence = st.session_state.confidence
            if st.session_state.actual_value == name:
                trim = name.split('.')[0]
                if prediction == trim.lower():
                    st.success(f"Predicted: {st.session_state.predicted_value} ({st.session_state.confidence:.2%})")
                else:
                    st.error(f"Predicted: {st.session_state.predicted_value} ({st.session_state.confidence:.2%})")
                st.caption(f'Actual: {st.session_state.actual_value}')
            audio_path = Path(audio_file)
            if audio_path.exists():
                # Display audio player
                audio_bytes = open(audio_file, 'rb').read()
                st.audio(audio_bytes, format='audio/wav')
            else:
                st.warning(f"Audio file not found: {name}")

# Separator
# st.markdown("---")

# # Model prediction section
# st.subheader("Model Prediction")
#
# # File uploader for testing
# uploaded_file = st.file_uploader("Upload an audio file to test the model", type=['wav', 'mp3', 'ogg'])
#
# if uploaded_file is not None:
#     # Display uploaded audio
#     st.audio(uploaded_file, format='audio/wav')
#
#     # Predict button
#     if st.button("🎯 Run Prediction", type="primary", use_container_width=True):
#         with st.spinner("Running model prediction..."):
#             # Placeholder for your model prediction logic
#             # Replace this with your actual model code
#             predicted_name = predict_speaker(uploaded_file)
#
#             # Display results
#             st.success(f"**Predicted Name:** {predicted_name}")
#
#             # You can add confidence scores or additional metrics here
#             st.info("Replace the `predict_speaker()` function with your actual model implementation")




