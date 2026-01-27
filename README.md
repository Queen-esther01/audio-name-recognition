# Voice Name Recognition

A deep learning-based speaker-dependent name recognition system that identifies spoken names from audio using MFCC features and a Convolutional Neural Network (CNN).

## Overview

This project trains a CNN model to recognize 20 different spoken names from audio recordings. It extracts Mel-frequency cepstral coefficients (MFCCs) as features and uses a sequential neural network for classification. The system includes a Streamlit web interface for interactive testing.

## Features

- **Audio Feature Extraction**: Extracts 40 MFCC features from audio files using Librosa
- **CNN Classification**: Uses a Convolutional Neural Network with Keras/TensorFlow
- **Video to Audio Conversion**: Supports extracting audio from video files using MoviePy
- **Web Interface**: Interactive Streamlit demo for testing predictions
- **Training Visualization**: Generates accuracy, loss, and confusion matrix plots

## Supported Names

The model recognizes the following 20 names:
Ahmed, Amber, Charlie, Christopher, Dominic, Emad, Emma, Hannah, Imogen, Jess, Josh, Joshua, Kailong, Kira, Manwel, Mateusz, Ngozi, Riley, Sivaprasath, Zack

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/voice-name-recognition.git
cd voice-name-recognition

# Install dependencies with uv
uv sync

# Or with pip
pip install -r requirements.txt
```

## Usage

### Running the Web Demo

```bash
streamlit run app.py
```

This launches an interactive web interface where you can:
1. Listen to sample audio files
2. Click "Predict" to identify the spoken name
3. See the predicted name with confidence score

### Training the Model

To train a new model on your own audio dataset:

```python
from recogniser import Recogniser

videos_path = "path/to/videos/*.mov"
audios_path = "path/to/audio_files/*.wav"

recogniser = Recogniser(videos_path, audios_path)
recogniser.run()
```

### Making Predictions

```python
import keras.saving
import numpy as np
from recogniser import Recogniser

# Load the trained model
model = keras.saving.load_model("model/audio_model.keras")

# Extract features from audio
features = Recogniser.extract_features("audio.wav", max_frames=100)
features = np.array(features, dtype="float32")
features = features / np.max(features)
features = np.expand_dims(features, axis=(0, -1))

# Predict
prediction = model.predict(features)
predicted_id = np.argmax(prediction, axis=1)
classes = Recogniser.get_classes()
predicted_name = classes[predicted_id[0]]
```

## Model Architecture

```
Layer (type)                Output Shape              Param #
================================================================
InputLayer                  (None, 40, 100, 1)        0
Conv2D (64 filters, 3x3)    (None, 38, 98, 64)        640
MaxPooling2D (3x3)          (None, 12, 32, 64)        0
Flatten                     (None, 24576)             0
Dense (256 units, ReLU)     (None, 256)               6,291,712
Dense (20 units, Softmax)   (None, 20)                5,140
================================================================
```

## Project Structure

```
voice-name-recognition/
├── app.py                 # Streamlit web interface
├── recogniser.py          # Core recognition module
├── main.py                # Entry point
├── model/
│   └── audio_model.keras  # Trained model weights
├── test_audio_files/      # Sample audio files for testing
├── report/                # Training visualizations
│   ├── accuracy.png
│   ├── confusion_matrix.png
│   └── loss.png
├── pyproject.toml         # Project dependencies
└── README.md
```

## Dependencies

- **TensorFlow/Keras** - Deep learning framework
- **Librosa** - Audio feature extraction
- **Streamlit** - Web interface
- **MoviePy** - Video to audio conversion
- **scikit-learn** - Data preprocessing and metrics
- **NumPy/Pandas** - Data manipulation
- **Matplotlib/Plotly** - Visualization

## Training Details

- **Dataset**: 400 audio samples (20 samples per name × 20 names)
- **Features**: 40 MFCCs padded to 100 frames
- **Epochs**: 30
- **Batch Size**: 32
- **Optimizer**: Adam (learning rate: 0.01)
- **Loss Function**: Categorical Cross-Entropy

## License

MIT License
