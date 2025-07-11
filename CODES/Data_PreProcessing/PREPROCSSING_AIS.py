

import os
import numpy as np
import nibabel as nib
import glob
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



# Set your dataset path
DATA_PATH = './AIS_DATASET/'

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Function to load and combine images with normalization
def load_and_combine_images(adc_path, dwi2_path):
    adc_data = nib.load(adc_path).get_fdata()
    dwi2_data = nib.load(dwi2_path).get_fdata()

    # Normalize each modality
    adc_data = scaler.fit_transform(adc_data.reshape(-1, adc_data.shape[-1])).reshape(adc_data.shape)
    dwi2_data = scaler.fit_transform(dwi2_data.reshape(-1, dwi2_data.shape[-1])).reshape(dwi2_data.shape)

    combined = np.stack([adc_data, dwi2_data], axis=-1)
    return combined



def central_crop(image, crop_dimensions):
    x_start, x_end = crop_dimensions['x']
    y_start, y_end = crop_dimensions['y']
    z_start, z_end = crop_dimensions['z']
    if image.ndim == 4:  # For 4D images (with channel dimension)
        return image[x_start:x_end, y_start:y_end, z_start:z_end, :]
    elif image.ndim == 3:  # For 3D images (without channel dimension)
        return image[x_start:x_end, y_start:y_end, z_start:z_end]
    else:
        raise ValueError("Unsupported image dimensionality")

# Load and process each sample
sample_dirs = glob.glob(os.path.join(DATA_PATH, 'sample_*'))
all_images = []
all_masks = []

for sample_dir in sample_dirs:
    adc_files = glob.glob(os.path.join(sample_dir, '*adc*.nii.gz'))
    dwi2_files = glob.glob(os.path.join(sample_dir, '*DWI2*.nii.gz'))
    mask_files = glob.glob(os.path.join(sample_dir, '*mask*.nii.gz'))

    if adc_files and dwi2_files and mask_files:
        combined_image = load_and_combine_images(adc_files[0], dwi2_files[0])
        mask = nib.load(mask_files[0]).get_fdata()

        # Adjust the crop dimensions as needed
        crop_dimensions = {'x': (29, 221), 'y': (29, 221), 'z': (2, 18)}  # Adjust z-range as needed
        cropped_image = central_crop(combined_image, crop_dimensions)
        cropped_mask = central_crop(mask, crop_dimensions)

        all_images.append(cropped_image)
        all_masks.append(cropped_mask)

# Convert lists to numpy arrays
all_images = np.array(all_images)
all_masks = np.array(all_masks)

# One-hot encode masks
all_masks_encoded = to_categorical(all_masks, num_classes=2)

def visualize_slice(image, mask, slice_num, title=""):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    if image.ndim == 4:
        plt.imshow(image[:, :, slice_num, 0], cmap='gray')  # For 4D images (with channel dimension)
    elif image.ndim == 3:
        plt.imshow(image[:, :, slice_num], cmap='gray')  # For 3D images (without channel dimension)
    plt.title(f'{title} - Image Slice {slice_num}')
    
    plt.subplot(1, 2, 2)
    if mask.ndim == 4:
        plt.imshow(mask[:, :, slice_num, 0], cmap='gray')  # For 4D masks (with channel dimension)
    elif mask.ndim == 3:
        plt.imshow(mask[:, :, slice_num], cmap='gray')  # For 3D masks (without channel dimension)
    plt.title(f'{title} - Mask Slice {slice_num}')
    plt.show()



# Visualize specific slices before preprocessing
for sample_dir in sample_dirs[:3]:  # Visualizing first 3 samples
    adc_files = glob.glob(os.path.join(sample_dir, '*adc*.nii.gz'))
    dwi2_files = glob.glob(os.path.join(sample_dir, '*DWI2*.nii.gz'))
    mask_files = glob.glob(os.path.join(sample_dir, '*mask*.nii.gz'))

    if adc_files and dwi2_files and mask_files:
        adc_data = nib.load(adc_files[0]).get_fdata()
        mask_data = nib.load(mask_files[0]).get_fdata()

        slice_num = 10  # Specify the slice number you want to visualize
        visualize_slice(adc_data, mask_data, slice_num, title="Before Preprocessing")

# Split data into train, validation, and test sets
train_images, test_images, train_masks, test_masks = train_test_split(all_images, all_masks_encoded, test_size=0.3, random_state=42)
val_images, test_images, val_masks, test_masks = train_test_split(test_images, test_masks, test_size=0.5, random_state=42)

# Create directories for saving data
base_save_path = 'D:/Research/Brain_Bleeding_Project/ProcessedData4/'
os.makedirs(os.path.join(base_save_path, 'train/images'), exist_ok=True)
os.makedirs(os.path.join(base_save_path, 'train/masks'), exist_ok=True)
os.makedirs(os.path.join(base_save_path, 'val/images'), exist_ok=True)
os.makedirs(os.path.join(base_save_path, 'val/masks'), exist_ok=True)
os.makedirs(os.path.join(base_save_path, 'test/images'), exist_ok=True)
os.makedirs(os.path.join(base_save_path, 'test/masks'), exist_ok=True)


# Save function
def save_data(images, masks, folder_path, set_name):
    image_path = os.path.join(folder_path, set_name, 'images')
    mask_path = os.path.join(folder_path, set_name, 'masks')
    for i in range(images.shape[0]):
        np.save(os.path.join(image_path, f'sample_{i}_image.npy'), images[i])
        np.save(os.path.join(mask_path, f'sample_{i}_mask.npy'), masks[i])

# Saving the split data
save_data(train_images, train_masks, base_save_path, 'train')
save_data(val_images, val_masks, base_save_path, 'val')
save_data(test_images, test_masks, base_save_path, 'test')

# Print the shape of final data
print("Shape of final data:")
print("Train Images Shape:", train_images.shape)
print("Train Masks Shape:", train_masks.shape)
print("Validation Images Shape:", val_images.shape)
print("Validation Masks Shape:", val_masks.shape)
print("Test Images Shape:", test_images.shape)
print("Test Masks Shape:", test_masks.shape)


# Randomly pick a sample and visualize
random_index = random.randint(0, train_images.shape[0] - 1)
random_slice_num = random.randint(0, 15)
visualize_slice(train_images[random_index], train_masks[random_index], random_slice_num)

# Visualize the same specific slices after preprocessing
sample_index = 0  # Specify the sample index you want to visualize
slice_num = 12  # Specify the slice number you want to visualize
visualize_slice(all_images[sample_index], all_masks[sample_index], slice_num, title="After Preprocessing")

