import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import get_data_generators

def evaluate_model(model_path, test_dir, img_size=(224,224), batch_size=32):
    model = tf.keras.models.load_model(model_path)
    _, test_gen = get_data_generators(test_dir, test_dir, img_size, batch_size)

    y_true = test_gen.classes
    y_pred = np.argmax(model.predict(test_gen), axis=-1)

    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=test_gen.class_indices.keys()))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=test_gen.class_indices.keys(), yticklabels=test_gen.class_indices.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    evaluate_model("models/vgg_fabric_defect.h5", "data/Fabric Defects Dataset/test")
