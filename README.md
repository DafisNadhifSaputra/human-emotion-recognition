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

## Model Performance

| Model | Test Accuracy | Test Loss | Epochs | Input Size |
|-------|---------------|-----------|--------|------------|
| **Base CNN** | **92.41%** | 0.268 | 33 | 128x128x1 (Grayscale) |
| MobileNetV3Small (Transfer Learning) | 81.56% | 0.551 | 50 | 224x224x3 (RGB) |

> 📌 Best model: **Base CNN** with 92.41% accuracy on test set

## Models

Models are hosted on Hugging Face: [🤗 DafisNadhifSaputra/human-emotion-recognition](https://huggingface.co/DafisNadhifSaputra/human-emotion-recognition)

| Model | Format | Input Size | Description |
|-------|--------|------------|-------------|
| `model_base.h5` | Keras H5 | 128x128x1 (Grayscale) | Custom CNN model ✅ Best |
| `model_transfer_learning.keras` | Keras | 224x224x3 (RGB) | MobileNetV3Small |
| `tflite/best_model.tflite` | TFLite | 128x128x1 | Mobile/Edge deployment |
| `tfjs_model/` | TF.js | 128x128x1 | Web deployment |

## Quick Start

```python
# Download model from Hugging Face
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="dafisnadhif/human-emotion-recognition",
    filename="model_base.h5"
)

# Load and use
import tensorflow as tf
model = tf.keras.models.load_model(model_path)
```

## Training Details

See `notebook.ipynb` for full training code.

| Parameter | Value |
|-----------|-------|
| **Dataset** | [Human Face Emotions](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions) |
| **Framework** | TensorFlow 2.x / Keras |
| **Optimizer** | AdamW (lr=1e-3, weight_decay=1e-4) |
| **Loss Function** | Sparse Categorical Crossentropy |
| **Batch Size** | 256 |
| **Callbacks** | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |
| **Train/Val/Test Split** | 80% / 10% / 10% |

## Requirements

```bash
pip install -r requirements.txt
```

## License

MIT License

## Author

Dafis Nadhif Saputra
