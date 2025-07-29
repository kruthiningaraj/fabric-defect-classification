import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from preprocess import get_data_generators

def build_transfer_model(base_model, num_classes):
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=predictions)

def train_transfer(train_dir, val_dir, model_type="vgg", img_size=(224, 224), batch_size=32, epochs=10):
    train_gen, val_gen = get_data_generators(train_dir, val_dir, img_size, batch_size)
    num_classes = train_gen.num_classes

    if model_type == "vgg":
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
    elif model_type == "resnet":
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
    else:
        raise ValueError("Invalid model_type. Use 'vgg' or 'resnet'.")

    model = build_transfer_model(base_model, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)

    model.save(f"models/{model_type}_fabric_defect.h5")

if __name__ == "__main__":
    train_transfer("data/Fabric Defects Dataset/train", "data/Fabric Defects Dataset/test", model_type="vgg")
