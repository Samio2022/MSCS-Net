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


## To apply the model on a custom dataset, the data should be converted into the following structure:
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
## Preprocessing
- `Data Loading: Set the dataset path and load images/masks`
- `Normalization: Apply MinMax scaling to images`
- `Cropping: Central crop to desired input shape (e.g., 192×192×16)`
- `Saving: Save preprocessed images/masks as .npy files`
- `Visualization: Use provided scripts to view random image-mask pairs`


## Custom Data Generator:
Utilize the custom data generator code provided, suitable for any dataset.
Includes data augmentation techniques to mitigate overfitting with small datasets.

## Model Configuration:

- Model architecture: DMSA_Seg.py
- Adjust input shape and number of channels/classes as needed.
- No further changes if your data matches the provided structure.



## Training:

- Set directory paths for train/validation data.
- Create data generators for training and validation.
- Compile the model with optimizer, loss, and metrics.
- Use callbacks: Early stopping and model checkpointing.
- Train the model via fit().
- Visualize metrics (loss/accuracy plots)

## Evaluation
- Load trained or pre-trained model.
- Prepare test images/masks as per training format.
- Generate predictions.
- Evaluate metrics: Accuracy, Dice, IoU, AHD.
- Visualize results for random test cases.

  ## Example Usage
- from ADV_AUG_CUSTOM_DATAGEN import imageLoader
- from DMSA_Seg import DMSA_Seg

