# 🤖 Machine Learning Pose Classification System

Complete ML pipeline untuk klasifikasi pose manusia menggunakan computer vision dan machine learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-red.svg)

## 📋 Daftar Isi

- [Fitur](#-fitur)
- [Struktur Project](#-struktur-project)
- [Instalasi](#-instalasi)
- [Quick Start](#-quick-start)
- [Pipeline ML](#-pipeline-ml)
- [VS Code Setup](#-vs-code-setup)
- [Advanced Usage](#-advanced-usage)

## 🎯 Fitur

### Machine Learning
- ✅ **Feature Extraction**: Ekstraksi 33 landmark points + angles + distances
- ✅ **Multiple Models**: Random Forest, SVM, KNN
- ✅ **Model Comparison**: Automatic evaluation & visualization
- ✅ **Real-time Prediction**: Klasifikasi pose dari webcam

### Data Processing
- ✅ **Data Collection**: Kumpulkan data training dari webcam
- ✅ **Data Augmentation**: Generate lebih banyak training samples
- ✅ **Auto Pipeline**: Automated end-to-end workflow

### Visualization
- ✅ **Confusion Matrix**: Visual performance metrics
- ✅ **Model Comparison Charts**: Bar charts untuk accuracy
- ✅ **Live Pose Detection**: Real-time visualization

## 📁 Struktur Project

```
pose-ml-project/
├── ml_pose_classifier.py       # Main ML training script
├── data_augmentation.py         # Data collection & augmentation
├── pose_detection_system.py     # Real-time detection (original)
├── app_streamlit.py            # Web interface
├── utils.py                    # Utility functions
├── config.ini                  # Configuration file
│
├── requirements.txt            # Basic dependencies
├── requirements_ml.txt         # ML dependencies
│
├── dataset/                    # Training data
│   ├── thinking/
│   ├── pointing_up/
│   ├── relaxed/
│   └── other/
│
├── dataset_augmented/          # Augmented data (auto-generated)
│
├── models/                     # Trained models
│   ├── pose_classifier.pkl
│   └── pose_dataset.pkl
│
├── results/                    # Training results
│   ├── confusion_matrix.png
│   └── model_comparison.png
│
└── pose-ml-project.code-workspace  # VS Code workspace
```

## 🚀 Instalasi

### 1. Clone/Download Project

```bash
cd pose-ml-project
```

### 2. Buat Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

**Basic Installation:**
```bash
pip install -r requirements_ml.txt
```

**Dengan Albumentations (untuk augmentation):**
```bash
pip install albumentations
```

**Full Installation (termasuk deep learning):**
```bash
pip install -r requirements_ml.txt
# Uncomment TensorFlow/PyTorch di requirements_ml.txt jika perlu
```

## ⚡ Quick Start

### Method 1: VS Code Tasks

1. Open project di VS Code
2. Tekan `Ctrl+Shift+P` (atau `Cmd+Shift+P` di Mac)
3. Pilih "Tasks: Run Task"
4. Pilih task yang diinginkan:
   - `Setup Virtual Environment`
   - `Install Dependencies`
   - `Collect Training Data`
   - `Train ML Model`
   - `Test Webcam Detection`

### Method 2: Command Line

**Step 1: Kumpulkan Data Training**

```bash
python data_augmentation.py
# Pilih: 1 (Collect data from webcam)
# Ikuti instruksi untuk setiap pose class
```

**Step 2: Augmentasi Data (Optional)**

```bash
python data_augmentation.py
# Pilih: 2 (Augment existing dataset)
```

**Step 3: Train Model**

```bash
python ml_pose_classifier.py
# Pilih: 3 (Train models)
```

**Step 4: Test Model**

```bash
python ml_pose_classifier.py
# Pilih: 4 (Test on webcam)
```

### Method 3: All-in-One Pipeline

```bash
python data_augmentation.py
# Pilih: 3 (Collect + Augment)
```

Kemudian:

```bash
python ml_pose_classifier.py
# Pilih: 3 (Train models)
```

## 🔄 Pipeline ML

### 1️⃣ Data Collection

```python
from data_augmentation import WebcamDataCollector

collector = WebcamDataCollector()
classes = ['thinking', 'pointing_up', 'relaxed', 'other']
collector.collect_all_classes(classes, images_per_class=50)
```

**Output:**
- `dataset/thinking/` - 50 images
- `dataset/pointing_up/` - 50 images
- `dataset/relaxed/` - 50 images
- `dataset/other/` - 50 images

### 2️⃣ Data Augmentation

```python
from data_augmentation import PoseDataAugmentor

augmentor = PoseDataAugmentor()
augmentor.augment_dataset('dataset', 'dataset_augmented', num_augments=5)
```

**Transformations Applied:**
- Horizontal flip
- Random rotation (±15°)
- Brightness/contrast adjustment
- Gaussian noise
- Motion blur
- HSV color shift

**Output:**
- Original 200 images → 1200+ augmented images

### 3️⃣ Feature Extraction

```python
from ml_pose_classifier import PoseFeatureExtractor

extractor = PoseFeatureExtractor()
features = extractor.extract_features('image.jpg')
```

**Features Extracted:**
- 33 landmarks × 4 values (x, y, z, visibility) = 132 features
- 8 key angles (elbow, shoulder, knee angles)
- 8 key distances (hand-to-face, hand-to-shoulder, etc.)
- **Total: 148 features per image**

### 4️⃣ Model Training

```python
from ml_pose_classifier import PoseClassifierTrainer

trainer = PoseClassifierTrainer()
X_train, X_test, y_train, y_test = trainer.prepare_data(features, labels)
results = trainer.train_all_models(X_train, X_test, y_train, y_test)
```

**Models Trained:**
1. **Random Forest** - Ensemble of decision trees
2. **SVM** - Support Vector Machine with RBF kernel
3. **KNN** - K-Nearest Neighbors (k=5)

**Output:**
- `pose_classifier.pkl` - Best model saved
- `confusion_matrix.png` - Performance visualization
- `model_comparison.png` - Accuracy comparison

### 5️⃣ Real-time Prediction

```python
from ml_pose_classifier import PosePredictor

predictor = PosePredictor('pose_classifier.pkl')
predictor.predict_webcam()
```

## 💻 VS Code Setup

### Recommended Extensions

1. **Python** (ms-python.python)
2. **Pylance** (ms-python.vscode-pylance)
3. **Jupyter** (ms-toolsai.jupyter)
4. **GitLens** (eamodio.gitlens)

### Debug Configurations

Project sudah include launch configurations:

- `F5` - Run current file
- `Ctrl+F5` - Run without debugging

**Available Configurations:**
- Python: ML Pose Classifier
- Python: Pose Detection System
- Python: Data Augmentation
- Python: Streamlit App
- Python: Train Model (Quick)

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Shift+P` | Command Palette |
| `Ctrl+Shift+B` | Run Build Task |
| `F5` | Start Debugging |
| `Ctrl+F5` | Run Without Debugging |
| `Ctrl+`` | Toggle Terminal |

## 🎓 Advanced Usage

### Custom Pose Classes

```python
# 1. Create new directory
os.makedirs('dataset/custom_pose', exist_ok=True)

# 2. Collect data
collector = WebcamDataCollector()
collector.collect_data('custom_pose', target_count=50)

# 3. Train with new class
# Model will automatically detect all classes in dataset/
```

### Hyperparameter Tuning

Edit `ml_pose_classifier.py`:

```python
self.models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200,      # Increase trees
        max_depth=20,          # Increase depth
        min_samples_split=5,   # Adjust splitting
        random_state=42
    ),
    'SVM': SVC(
        kernel='rbf',
        C=10,                  # Regularization
        gamma='scale',
        random_state=42
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=7,         # Increase neighbors
        weights='distance'     # Use distance weighting
    )
}
```

### Custom Feature Engineering

Add new features in `PoseFeatureExtractor`:

```python
def calculate_custom_features(self, landmarks):
    features = []
    lm = landmarks.reshape(-1, 4)
    
    # Example: Head tilt
    left_ear = lm[7][:2]
    right_ear = lm[8][:2]
    head_tilt = np.arctan2(right_ear[1] - left_ear[1], 
                           right_ear[0] - left_ear[0])
    features.append(head_tilt)
    
    # Add more custom features...
    
    return np.array(features)
```

### Model Ensemble

```python
from sklearn.ensemble import VotingClassifier

# Create ensemble
ensemble = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()),
        ('svm', SVC(probability=True)),
        ('knn', KNeighborsClassifier())
    ],
    voting='soft'
)

# Train
ensemble.fit(X_train, y_train)
```

### Deep Learning Approach (Optional)

```python
import tensorflow as tf
from tensorflow import keras

# Build neural network
model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(148,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, epochs=50, validation_split=0.2)
```

## 📊 Performance Metrics

### Expected Results

| Model | Accuracy | Training Time |
|-------|----------|---------------|
| Random Forest | 85-95% | ~5 seconds |
| SVM | 80-90% | ~10 seconds |
| KNN | 75-85% | ~2 seconds |

*Results may vary based on dataset quality and size*

### Improving Accuracy

1. **Collect More Data**: Aim for 100+ images per class
2. **Better Quality**: Good lighting, clear poses
3. **Data Augmentation**: Increase variety
4. **Feature Engineering**: Add domain-specific features
5. **Hyperparameter Tuning**: Grid search optimal parameters

## 🐛 Troubleshooting

### Import Error: albumentations

```bash
# If you get albumentations import error:
# The data_augmentation.py will automatically fallback to simple transforms
# Or install it:
pip install albumentations
```

### Low Accuracy

**Solutions:**
- Collect more training data
- Ensure poses are clearly different
- Check data quality (lighting, resolution)
- Try different train/test split ratios
- Add more augmentation

### Webcam Not Working

**Check:**
```python
# Test camera access
import cv2
cap = cv2.VideoCapture(0)  # Try 0, 1, 2
print(cap.isOpened())
```

### Memory Issues

**For large datasets:**
```python
# Process in batches
batch_size = 100
for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    # Process batch
```

## 📚 Resources

- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose)
- [scikit-learn](https://scikit-learn.org/stable/)
- [OpenCV Python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Albumentations](https://albumentations.ai/)

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Add more pose classes
- [ ] Implement LSTM for pose sequences
- [ ] Mobile deployment (TFLite)
- [ ] REST API for predictions
- [ ] Docker containerization

## 📄 License

Free to use for educational and personal projects.

---

**Happy Machine Learning! 🚀**

Made with ❤️ using MediaPipe, scikit-learn, and OpenCV
