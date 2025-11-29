---
license: mit
tags:
  - image-classification
  - tensorflow
  - keras
  - emotion-recognition
  - facial-expression
  - cnn
---

# Human Emotion Recognition

A deep learning model for classifying human facial emotions into 5 categories.

## Model Description

This project contains CNN-based models trained to recognize human emotions from facial images.

### Emotion Classes
- 😠 **Angry**
- 😨 **Fear**
- 😊 **Happy**
- 😢 **Sad**
- 😲 **Surprise**

## Models

Models are hosted on Hugging Face: [🤗 dafisnadhif/human-emotion-recognition](https://huggingface.co/Dafisns/human-emotion-recognition)

| Model | Format | Input Size | Description |
|-------|--------|------------|-------------|
| `model_base.h5` | Keras H5 | 128x128x1 (Grayscale) | Custom CNN model |
| `model_transfer_learning.keras` | Keras | 224x224x3 (RGB) | MobileNetV3Small |
| `tflite/best_model.tflite` | TFLite | 128x128x1 | Mobile/Edge deployment |
| `tfjs_model/` | TF.js | 128x128x1 | Web deployment |

## Quick Start

```python
# Download model from Hugging Face
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="Dafisns/human-emotion-recognition",
    filename="model_base.h5"
)

# Load and use
import tensorflow as tf
model = tf.keras.models.load_model(model_path)
```

## Training

See `notebook.ipynb` for full training code.

- **Dataset**: [Human Face Emotions](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions)
- **Framework**: TensorFlow/Keras
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)

## Requirements

```bash
pip install -r requirements.txt
```

## License

MIT License

## Author

Dafis Nadhif Saputra
