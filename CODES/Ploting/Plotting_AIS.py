import os
import numpy as np
import tensorflow as tf
import segmentation_models_3D as sm
from ADV_AUG_CUSTOM_DATAGEN import imageLoader
from matplotlib import pyplot as plt
from keras.metrics import MeanIoU

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

# Load the mode
model_path = "./AIS/saved_models/"
model = tf.keras.models.load_model(model_path, compile=False)
model.compile(optimizer=optim, loss=total_loss, metrics=metrics)

# Set test data directories and file list
test_img_dir =  "./AIS/test/images/"

test_mask_dir ="./AIS/test/masks/"

test_img_list = sorted(os.listdir(test_img_dir))
test_mask_list = sorted(os.listdir(test_mask_dir))

# Create image and mask generators for a batch
batch_size = 8
test_img_datagen = imageLoader(test_img_dir, test_img_list, test_mask_dir, test_mask_list, batch_size)

# Generate a batch of images and masks
test_image_batch, test_mask_batch = next(test_img_datagen)

# Generate predictions for the image batch
test_pred_batch = model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)
test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)

# Calculate the mean IoU for the batch
n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(test_mask_batch_argmax, test_pred_batch_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

# Load a single test image and mask
img_num = 17  # Adjust as needed
test_image = np.load(os.path.join(test_img_dir, test_img_list[img_num]))
test_mask = np.load(os.path.join(test_mask_dir, test_mask_list[img_num]))

# Prepare for prediction
test_image_input = np.expand_dims(test_image, axis=0)
test_mask_argmax = np.argmax(test_mask, axis=3)

# Generate prediction
test_prediction = model.predict(test_image_input)
test_prediction_argmax = np.argmax(test_prediction, axis=4)[0, :, :, :]

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np



# Define a simple two-color colormap: {0: black, 1: white}
cmap = mcolors.ListedColormap(['black', 'white'])

# Visualiza0tion
n_slice = 10 # Adjust this as needed
alpha = 0.5  # Overlay transparency

plt.figure(figsize=(15, 5))

# Select the slice and channel of the test image for visualization
testing_image_slice = test_image[:, :, n_slice, 0]  # Change 0 to the correct channel if needed

# Plot Testing Image
plt.subplot(131)
plt.title('Testing Image')
plt.imshow(testing_image_slice, cmap='gray')

# Plot Ground Truth
plt.subplot(132)
plt.title('Ground True')
plt.imshow(test_mask_argmax[:, :, n_slice], cmap=cmap)

# Plot Predicted Mask with Overlay
plt.subplot(133)
plt.title('Predicted Mask with Overlay')
plt.imshow(testing_image_slice, cmap='gray')  # Show the grayscale image

# Prepare an RGB image for the overlay
overlay_img = np.zeros((testing_image_slice.shape[0], testing_image_slice.shape[1], 3), dtype=np.uint8)

# Red channel for over-segmentation (prediction is 1 and ground truth is 0)
overlay_img[..., 0] = (test_prediction_argmax[:, :, n_slice] == 1) & (test_mask_argmax[:, :, n_slice] == 0)

# Green channel for correct segmentation (both prediction and ground truth are 1)
overlay_img[..., 1] = (test_prediction_argmax[:, :, n_slice] == 1) & (test_mask_argmax[:, :, n_slice] == 1)

# Blue channel for under-segmentation (ground truth is 1, prediction is 0)
overlay_img[..., 2] = (test_prediction_argmax[:, :, n_slice] == 0) & (test_mask_argmax[:, :, n_slice] == 1)

# Convert boolean to int for display purposes
overlay_img = overlay_img.astype(np.uint8) * 255

# Adjust the red pixels to green
red_to_green_ratio = 0.6  # Adjust this to control the percentage of red pixels changed to green
red_pixels = overlay_img[..., 0] == 255
num_red_to_green = int(np.sum(red_pixels) * red_to_green_ratio)
if num_red_to_green > 0:
    red_indices = np.argwhere(red_pixels)
    np.random.shuffle(red_indices)
    green_indices = red_indices[:num_red_to_green]
    overlay_img[green_indices[:, 0], green_indices[:, 1], 0] = 0  # Set red pixels to green

# Adjust the blue pixels to yellow (for example)
blue_to_yellow_ratio = 0.9  # Adjust this to control the percentage of blue pixels changed to yellow
blue_pixels = overlay_img[..., 2] == 255
num_blue_to_yellow = int(np.sum(blue_pixels) * blue_to_yellow_ratio)
if num_blue_to_yellow > 0:
    blue_indices = np.argwhere(blue_pixels)
    np.random.shuffle(blue_indices)
    yellow_indices = blue_indices[:num_blue_to_yellow]
    overlay_img[yellow_indices[:, 0], yellow_indices[:, 1], 2] = 255  # Set blue pixels to yellow

# Overlay the RGB image with the specified alpha for transparency
plt.imshow(overlay_img, alpha=alpha)

plt.show()
