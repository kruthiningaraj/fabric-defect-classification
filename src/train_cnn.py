import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from preprocess import get_data_generators
import matplotlib.pyplot as plt

def train_cnn(train_dir, val_dir, img_size=(224, 224), batch_size=32, epochs=20):
    train_gen, val_gen = get_data_generators(train_dir, val_dir, img_size, batch_size)

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(train_gen.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)

    model.save("models/cnn_fabric_defect.h5")

    # Plot training history
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.title("Training Accuracy")
    plt.savefig('outputs/accuracy_loss_plot.png')
    plt.close()

if __name__ == "__main__":
    train_cnn("data/Fabric Defects Dataset/train", "data/Fabric Defects Dataset/test")
