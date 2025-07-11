
import os
import numpy as np
import tensorflow as tf
from ADV_AUG_CUSTOM_DATAGEN import imageLoader
import os

import tensorflow as tf
import keras
import numpy as np
import segmentation_models_3D as sm
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.metrics import MeanIoU






# Set test data directories and lists
# Set test data directory
test_img_dir =   ".AIS/test/images/"

test_mask_dir = ".AIS/test/masks/"


test_img_list = os.listdir(test_img_dir)
test_mask_list = os.listdir(test_mask_dir)

# Set batch size and create custom generator for test data
batch_size = 16
test_generator = imageLoader(test_img_dir, test_img_list, test_mask_dir, test_mask_list, batch_size)

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

# Load the model
model_path =  "D:/Research/Brain_Bleeding_Project/Models/MSCS/results/MCSeg_best_20241202135851/"
model = tf.keras.models.load_model(model_path, compile=False)

# Compile the model
model.compile(optimizer=optim, loss=total_loss, metrics=metrics)


# Evaluate the model on the test data
test_steps = len(test_img_list) // batch_size
test_loss, test_accuracy, test_iou_score, test_precision, test_recall, test_dice_score = model.evaluate(test_generator, steps=test_steps)

#print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test IOU Score: {test_iou_score}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")
print(f"Test Dice Score: {test_dice_score}")
