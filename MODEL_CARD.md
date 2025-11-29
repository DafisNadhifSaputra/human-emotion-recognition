---
license: mit
tags:
  - image-classification
  - tensorflow
  - keras
  - emotion-recognition
  - facial-expression
  - cnn
  - tflite
  - tfjs
datasets:
  - samithsachidanandan/human-face-emotions
metrics:
  - accuracy
pipeline_tag: image-classification
library_name: keras
---

# Human Emotion Recognition

Deep learning models for classifying human facial emotions.

## Emotion Classes
- 😠 Angry
- 😨 Fear  
- 😊 Happy
- 😢 Sad
- 😲 Surprise

## Model Performance

| Model | Test Accuracy | Test Loss | Epochs |
|-------|---------------|-----------|--------|
| **Base CNN** | **92.41%** | 0.268 | 33 |
| MobileNetV3Small | 81.56% | 0.551 | 50 |

> 🏆 Best model: **Base CNN** with 92.41% test accuracy

## Models

| File | Format | Input Size | Description |
|------|--------|------------|-------------|
| `model_base.h5` | Keras H5 | 128x128x1 | Custom CNN (Grayscale) |
| `model_transfer_learning.keras` | Keras | 224x224x3 | MobileNetV3Small (RGB) |
| `tflite/best_model.tflite` | TFLite | 128x128x1 | Mobile/Edge |
| `tfjs_model/` | TF.js | 128x128x1 | Web deployment |

## Usage

### Python

```python
from huggingface_hub import hf_hub_download
import tensorflow as tf
import numpy as np

# Download model
model_path = hf_hub_download(
    repo_id="dafisnadhif/human-emotion-recognition",
    filename="model_base.h5"
)

# Load model
model = tf.keras.models.load_model(model_path)

# Predict
CLASS_NAMES = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']
predictions = model.predict(img_batch)
print(CLASS_NAMES[np.argmax(predictions[0])])
```

### TensorFlow Lite

```python
from huggingface_hub import hf_hub_download
import tensorflow as tf

tflite_path = hf_hub_download(
    repo_id="dafisnadhif/human-emotion-recognition",
    filename="tflite/best_model.tflite"
)

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
```

### TensorFlow.js

```javascript
const model = await tf.loadLayersModel(
  'https://huggingface.co/dafisnadhif/human-emotion-recognition/resolve/main/tfjs_model/model.json'
);
```

## Training Details

| Parameter | Value |
|-----------|-------|
| **Dataset** | [Human Face Emotions](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions) |
| **Images** | ~47,000 facial images |
| **Source Code** | [GitHub](https://github.com/DafisNadhifSaputra/human-emotion-recognition) |
| **Framework** | TensorFlow 2.x / Keras |
| **Optimizer** | AdamW (lr=1e-3, weight_decay=1e-4) |
| **Loss** | Sparse Categorical Crossentropy |
| **Batch Size** | 256 |
| **Callbacks** | EarlyStopping (patience=8), ReduceLROnPlateau |

## License

MIT License

## Author

Dafis Nadhif Saputra