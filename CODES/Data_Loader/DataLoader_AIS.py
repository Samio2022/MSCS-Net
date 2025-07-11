# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 22:54:18 2023

@author: Asim
"""

import os
import numpy as np
import random
from matplotlib import pyplot as plt

def load_img(img_dir, img_list):
    images = []
    for image_name in img_list:
        if image_name.endswith('.npy'):
            image = np.load(os.path.join(img_dir, image_name))
            images.append(image)
    return np.array(images)

def augment_volume(volume, mask):
    """ Apply random flips and intensity adjustments to the volume """
    # Random horizontal flip
    if random.choice([True, False]):
        volume = volume[:, ::-1, :, :]  # horizontal flip
        mask = mask[:, ::-1, :, :]
    
    # Random vertical flip
    if random.choice([True, False]):
        volume = volume[::-1, :, :, :]  # vertical flip
        mask = mask[::-1, :, :, :]

    # Intensity adjustment
    if random.choice([True, False]):
        intensity_factor = random.uniform(0.9, 1.1)  # Adjust the intensity slightly
        volume = np.clip(volume * intensity_factor, 0, 1)

    # Noise injection
    if random.choice([True, False]):
        noise_factor = random.uniform(0, 0.05)  # Adjust the level of noise
        noise = noise_factor * np.random.normal(size=volume.shape)
        volume = np.clip(volume + noise, 0, 1)

    return volume, mask

def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    L = len(img_list)

    # Infinite loop for the generator
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])

            # Apply augmentation
            for i in range(X.shape[0]):
                X[i], Y[i] = augment_volume(X[i], Y[i])

            yield (X, Y)  # Yielding a tuple with two numpy arrays

            batch_start += batch_size
            batch_end += batch_size

# Testing the generator
train_img_dir = "D:/Research/Brain_Bleeding_Project/ProcessedData/train/images/"
train_mask_dir = "D:/Research/Brain_Bleeding_Project/ProcessedData/train/masks/"

train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

batch_size = 16
train_img_datagen = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size)

# Verify generator
img, msk = next(train_img_datagen)

img_num = random.randint(0, img.shape[0] - 1)
test_img = img[img_num]
test_mask = msk[img_num]
test_mask = np.argmax(test_mask, axis=3)  # Assuming masks are one-hot encoded

n_slice = random.randint(0, test_img.shape[2] - 1)

plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.imshow(test_img[:, :, n_slice, 0], cmap='gray')
plt.title('DWI Image')
plt.subplot(222)
plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
plt.title('ADC Image')
plt.subplot(223)
plt.imshow(test_mask[:, :, n_slice])
plt.title('Mask')
plt.show()
