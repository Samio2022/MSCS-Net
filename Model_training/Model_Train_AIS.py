

import os
import numpy as np
import tensorflow as tf
import segmentation_models_3D as sm
from matplotlib import pyplot as plt
from ADV_AUG_CUSTOM_DATAGEN import imageLoader
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from datetime import datetime



# # Set up TensorFlow logging level
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# Data directories and lists


# Data directories and lists
train_img_dir = ".AIS/train/images/"
train_mask_dir = ".AIS/train/masks/"



val_img_dir =   ".AIS/val/images/"
val_mask_dir = ".AIS/val/masks/"

train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)
val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

# Set batch size and create custom generators
batch_size = 8

# For training, with augmentation
train_generator = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size, augment=True)

# For validation, without augmentation
val_generator = imageLoader(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size, augment=False)


# Define loss, metrics, and optimizer to be used for training
wt0, wt1 = 0.5, 0.5
dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = ['accuracy',
            sm.metrics.IOUScore(threshold=0.5, name='iou_score'),
            sm.metrics.Precision(threshold=0.5),
            sm.metrics.Recall(threshold=0.5),
            sm.metrics.FScore(threshold=0.5, name='Dice-score')
            ]

LR = 0.0001
optim = tf.keras.optimizers.Adam(LR)

# Define the 3D model
from MSCS_NET import  MSCS_Net 
model =  MSCS_Net  (IMG_HEIGHT=192,
                          IMG_WIDTH=192,
                          IMG_DEPTH=16,
                          IMG_CHANNELS=2,
                          num_classes=2)


model.compile(optimizer=optim, loss=total_loss, metrics=metrics)
print(model.summary())


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

class SaveModelEveryN(Callback):
    def __init__(self, save_path, save_every_n_epochs):
        super().__init__()
        self.save_path = save_path
        self.save_every_n_epochs = save_every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_every_n_epochs == 0:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            model_path = f"{self.save_path}/EFESeg_Yolo_V2_mod_epoch_{epoch + 1}_{timestamp}"
            self.model.save(model_path, save_format='tf')
            print(f"\nModel saved in TensorFlow SavedModel format at {model_path}")




# Define the early stopping callback
earlystop_callback = EarlyStopping(
    monitor='val_loss',
    patience=70,
    verbose=1,
    mode='auto',
    restore_best_weights=True
)


timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
checkpoint_path = f"D:/Research/Brain_Bleeding_Project/Models/MSCS/results/MCSeg_best_{timestamp}"
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_freq='epoch',
    save_format='tf'
)


# Create the custom save model every 100 epochs callback
save_every_100_epochs_path = "D:/Research/Brain_Bleeding_Project/Models/MSCS/results"
save_every_100_epochs_callback = SaveModelEveryN(save_every_100_epochs_path, 100)


# Fit the model
steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=100,
    verbose=1,
    validation_data=val_generator,
    validation_steps=val_steps_per_epoch,
    callbacks=[earlystop_callback, checkpoint_callback, save_every_100_epochs_callback]
)



# Plot the training and validation metrics at each epoch
plt.figure(figsize=(20, 5))

# Accuracy
plt.subplot(1, 5, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# IOU Score
plt.subplot(1, 5, 2)
plt.plot(history.history['iou_score'], label='Training')
plt.plot(history.history['val_iou_score'], label='Validation')
plt.title('IOU Score')
plt.xlabel('Epochs')
plt.ylabel('IOU Score')
plt.legend()

# Precision
plt.subplot(1, 5, 3)
plt.plot(history.history['precision'], label='Training')
plt.plot(history.history['val_precision'], label='Validation')
plt.title('Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()

# Recall
plt.subplot(1, 5, 4)
plt.plot(history.history['recall'], label='Training')
plt.plot(history.history['val_recall'], label='Validation')
plt.title('Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()

# F1-Score
plt.subplot(1, 5, 5)
plt.plot(history.history['Dice-score'], label='Training')
plt.plot(history.history['val_Dice-score'], label='Validation')
plt.title('Dice-Score')
plt.xlabel('Epochs')
plt.ylabel('Dice-Score')
plt.legend()

plt.tight_layout()
plt.show()

