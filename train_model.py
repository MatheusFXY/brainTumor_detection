import math
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight

if not os.path.exists("models"):
    os.makedirs("models")

train_dir = 'split_dataset/train'
val_dir = 'split_dataset/val'

img_width, img_height = 224, 224
batch_size = 32
initial_epochs = 30
fine_tune_epochs = 20
total_epochs = initial_epochs + fine_tune_epochs

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

steps_per_epoch = math.ceil(train_generator.samples / train_generator.batch_size)
validation_steps = math.ceil(validation_generator.samples / validation_generator.batch_size)
print("Steps per epoch:", steps_per_epoch)
print("Validation steps:", validation_steps)

classes = np.unique(train_generator.classes)
class_weights = class_weight.compute_class_weight('balanced', classes=classes, y=train_generator.classes)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('models/modelo_brain_tumor_best_phase1.keras', monitor='val_loss', save_best_only=True)

history_phase1 = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=initial_epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stop, checkpoint],
    class_weight=class_weights
)

with open('models/history_phase1.pkl', 'wb') as f:
    pickle.dump(history_phase1.history, f)

base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
checkpoint_fine = ModelCheckpoint('models/modelo_brain_tumor_best_finetune.keras', monitor='val_loss', save_best_only=True)

history_phase2 = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=fine_tune_epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[reduce_lr, checkpoint_fine],
    class_weight=class_weights
)

with open('models/history_phase2.pkl', 'wb') as f:
    pickle.dump(history_phase2.history, f)

model.save('models/modelo_brain_tumor_improved.keras')
