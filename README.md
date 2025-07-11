## MSCS-Net: A multi-scale contextual segmentation network for low-resolution acute stroke lesions


## Overview

**MSCS-Net** is a deep learning model for 3D medical image segmentation, specifically designed for low-resolution MRI sequences (e.g., ADC/DWI) commonly used in acute stroke triage. The network incorporates advanced multi-scale attention modules and is optimized for clinical deployment, offering both high segmentation accuracy and computational efficiency.

## Requirements
- `python==3.9`
- `tensorflow==2.10.0`
- `keras==2.10.0`
- `numpy==1.26.1`
- `pandas==2.1.3`
- `scikit-learn==1.3.2`
- `scikit-image==0.24.0`
- `matplotlib==3.8.1`
- `seaborn==0.13.2`
- `opencv-python==4.9.0.80`
- `nibabel==5.3.2`
- `Pillow==11.0.0`
- `segmentation-models-3D==1.0.4`
- `GPUtil==1.4.0`
- `SimpleITK==2.4.1`
- `statsmodels==0.14.4`
- `tqdm==4.67.1`

## Dataset (The dataset can be downloaded from the below link)
https://drive.google.com/file/d/17BHI51eEiYr-gxHTG49gBhRCAAbclrV-/view?usp=sharing


## To apply the model on a custom dataset, the data tree should be constructed as:
``` 
    ├── data
          ├── images
                ├── image_1.png
                ├── image_2.png
                ├── image_n.png
          ├── masks
                ├── image_1.png
                ├── image_2.png
                ├── image_n.png
```
##  HOW TO USE THIS MODEL?
1. Preprocessing Data: 
Data Preprocessing Steps:
Data Loading:
Set the dataset path and initialize MinMaxScaler.
Combine and Normalize Images:
Crop Image: Perform a central crop on the combined images and masks.
Process Each Sample: Load, combine, normalize, and crop each sample's images and masks.
Data Splitting: Split the data into training, validation, and test sets.
Save Data: Create directories and save the split data into .npy files.
Data Verification: Print the shapes of the final data arrays.
Visualization: Visualize a random slice from a randomly selected sample.


2. Custom Data Generator:
Utilize the custom data generator code provided, suitable for any dataset.
Includes data augmentation techniques to mitigate overfitting with small datasets.

HOW TO USE OR MODIFY?

The file "ADV_AUG_CUSTOM_DATAGEN.py" serves as the Custom Data Generator. If you prefer not to apply Data Augmentation, simply remove the relevant sections from this code. Otherwise, no changes are necessary. This code is designed to work seamlessly with any 3D dataset.


3. Model Configuration:

NOTE： no changes are necessary. This code is designed to work seamlessly with any 3D dataset.

the file "DMSA_Seg.py" is the model code. 
Modify the input shape of the model according to your dataset size.
Adjust the number of channels and classes as needed.
The model is a self-optimizer, automatically adjusting itself.



4. Model Training:

NOTE: You only need to update the directories for data loading.

Model Training Steps:
Data Setup: Define directories for training and validation data.
Data Generators: Create generators for training and validation data.
Model Definition: Define the 3D model architecture.
Compile Model: Compile the model with optimizer, loss, and metrics.
Callbacks: Implement early stopping and model checkpointing callbacks.
Training: Train the model with fit method, specifying generators and callbacks.
Metrics Visualization: Visualize training and validation metrics.


5. Evaluation

NOTE: You only need to update the directories for data loading and the pre-trained model.

Model Evaluation Steps:
Load Model: Load the pre-trained model.
Metrics and Loss: Utilize metrics like accuracy, IoU score, and loss functions such as Dice Loss.
Prepare Data: Set up directories for test images and masks.
Generate Predictions: Predict on test images using the model.
Evaluate Mean IoU: Compute Mean IoU score for predictions against ground truth masks.
Visualize Results: Display a sample visualization of test image, ground truth mask, and predicted mask.
