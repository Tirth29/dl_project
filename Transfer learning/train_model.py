import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionResNetV2

# Paths
TRAIN_DIR = 'dataset2/train'
VAL_DIR = 'dataset2/val'
MODEL_SAVE_PATH = 'face_recognition_model.h5'

# Parameters
IMG_SIZE = 160
BATCH_SIZE = 8
NUM_CLASSES = 18

# Preprocess Data
train_datagen = ImageDataGenerator(rescale=1.0/255.0, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical'
)

# Load Pretrained Model
base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Freeze base model layers

# Add Custom Layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=val_generator.samples // BATCH_SIZE
)

class CustomScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale_factor, **kwargs):
        super(CustomScaleLayer, self).__init__(1.1,**kwargs)
        self.scale_factor = scale_factor

    def call(self, inputs):
        return inputs * self.scale_factor

    def get_config(self):
        config = super(CustomScaleLayer, self).get_config()
        config.update({"scale_factor": self.scale_factor})
        return config
from tensorflow.keras.models import load_model

custom_objects = {
    "CustomScaleLayer": CustomScaleLayer(1)
}

MODEL_PATH = 'face_recognition_model.h5'
model = load_model(MODEL_PATH, custom_objects=custom_objects)

import h5py

MODEL_PATH = 'face_recognition_model.h5'
with h5py.File(MODEL_PATH, 'r') as f:
    print(list(f.keys()))


# Save Model
model.save('face_recognition_model.h5', save_format='h5')
print(f"Model saved to {MODEL_SAVE_PATH}")
