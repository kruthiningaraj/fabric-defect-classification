# Fabric Defect Classification 🧵

Deep learning project for classifying knit fabric textures and dyeing defects using **CNNs** and **transfer learning (VGG16, ResNet50)**.

---

## 🚀 Features
- Image preprocessing: normalization, augmentation, standardization
- CNN model from scratch for texture classification
- Transfer learning using VGG16 and ResNet50
- Evaluation: confusion matrix, classification report

---

## 📂 Dataset
We use the [Fabric Defect Dataset](https://www.kaggle.com/datasets/rmshashi/fabric-defect-dataset).

### REQUIRED:
You must manually download this dataset from Kaggle and place it in the `data/` folder:
```
fabric-defect-classification/
└── data/
    └── Fabric Defects Dataset/
        ├── train/
        ├── test/
```

---

## 🛠️ Setup

1️⃣ Clone repository:
```bash
git clone https://github.com/your-username/fabric-defect-classification.git
cd fabric-defect-classification
```

2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🏋️ Training

- Train CNN from scratch:
```bash
python src/train_cnn.py
```

- Train Transfer Learning (VGG16 or ResNet50):
```bash
python src/train_transfer.py
```

---

## 📊 Evaluation
Evaluate the trained model:
```bash
python src/evaluate.py
```

---

## 📦 Tech Stack
- Python 3.x
- TensorFlow/Keras
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

---

## 📜 License
MIT License © 2025
