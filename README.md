## MSCS-Net: A deep learning framework for accurate segmentation of low-resolution stroke lesions
This repository contains the original implementation of "MSCS-Net : A deep learning framework for accurate segmentation of low-resolution stroke lesions" in Keras (Tensorflow as backend).

## Paper
MSCS-Net has been published in Measurement

Zaman, Asim, Rashid Khan, Mazen M. Yassin, Faizan Ahmad, Jiaxi Lu, Irfan Mehmud, Yu Luo, and Yan Kang. "MSCS-Net: A deep learning framework for accurate segmentation of low-resolution stroke lesions." Measurement (2025): 118925.

## Overview

**MSCS-Net** is a deep learning model for 3D medical image segmentation, specifically designed for low-resolution MRI sequences (e.g., ADC/DWI) commonly used in acute stroke triage. The network incorporates advanced multi-scale attention modules and is optimized for clinical deployment, offering both high segmentation accuracy and computational efficiency.

![Fig  1](https://github.com/user-attachments/assets/fdc59217-7739-4c57-a54a-0fde12738bc6)
<img width="6013" height="2776" alt="Fig  2" src="https://github.com/user-attachments/assets/7bfba16c-c901-4da6-9e74-17c8d0e7febd" />
![Fig  3](https://github.com/user-attachments/assets/44e791d7-d743-42ba-b69e-8a6906a60af3)


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

## Dataset (The AIS dataset can be downloaded from the below link)
https://drive.google.com/file/d/17BHI51eEiYr-gxHTG49gBhRCAAbclrV-/view?usp=sharing

## Citation Request
If you use Our AIS dataset in your project, please cite the following paper

@article{zaman2025mscs,
  title={MSCS-Net: A deep learning framework for accurate segmentation of low-resolution stroke lesions},
  author={Zaman, Asim and Khan, Rashid and Yassin, Mazen M and Ahmad, Faizan and Lu, Jiaxi and Mehmud, Irfan and Luo, Yu and Kang, Yan},
  journal={Measurement},
  pages={118925},
  year={2025},
  publisher={Elsevier}
}



## To apply the model on a custom dataset, the data should be converted into the following structure:
``` 
    ├── data
          ├── images
                ├── image_1.nii/npy
                ├── image_2.nii/npy
                ├── image_n.nii/npy
          ├── masks
                ├── image_1.nii/npy
                ├── image_2.nii/npy
                ├── image_n.nii/npy
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

## Citation Request
If you use MSCS-Net in your project, please cite the following paper

@article{zaman2025mscs,
  title={MSCS-Net: A deep learning framework for accurate segmentation of low-resolution stroke lesions},
  author={Zaman, Asim and Khan, Rashid and Yassin, Mazen M and Ahmad, Faizan and Lu, Jiaxi and Mehmud, Irfan and Luo, Yu and Kang, Yan},
  journal={Measurement},
  pages={118925},
  year={2025},
  publisher={Elsevier}
}




