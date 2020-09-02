# End-to-End Models for Covid-19 Classification
The repository contains two classification models (Resnet18 & MLP) and two Segmentation models (MaskRCNN, UNet) for an end-to-end classification of Covid-19
vs. Pneumonia (viral or bacterial) vs. Normal cases from chest X-ray images.

## Directory Structure
```
model_weights\ -> Saved weights for the models (used for Inference)
unet_pipeline\ -> Contains a UNet implementation for Lung segmentation
mlp_pipeline\ -> Contains a MLP implementation 3 class classification
utility\ -> Contains helper function for the overall pipeline
```
### Prerequisites
Running **train/predict** requires correct path to the input data and the following **packages** for ```python-3.x```
```
matplotlib==3.1.1
opencv-python==4.2.0
scikit-image==0.15.0
sklearn==0.23.2
numpy==1.17.4
torch==1.4.0+cu92
torchvision==0.5.0+cu92
```
### Data Statistics

* Number of Images used for training = 1680
* Number of Images used for validation = 420
* Number of Images used for testing = 150

### Data Processing
* The chest X-ray images are normalized to a mean of and a standard deviation of 1. The pixel values are scaled between 0 and 1.
* For the purpose of using CNNs with CUDA, the data was resized to a tensor size of [1, 256, 256].

### Model Parameter

* **Loss function For MaskRCNN**: Pixel-wise Binary Cross Entropy
* **Loss function For UNet**: Binary Cross Entropy + Soft Dice + Inverted Soft Dice
* **Optimizers**: Adam (UNet, Resnet18), SGD (MaskRCNN, MLP)
* **Epochs**: 100 (MaskRCNN, UNet), 200 (Resnet18), 500 (MLP)
* **Learning Rate**: 0.001 (reduces 1/10 if validation loss does not increase for 5 epochs) 

## Loss Graph
#### UNet | MaskRCNN
![Alt text](logs/train_losses_unet.png?raw=true "Title")
![Alt text](logs/mask_rcnn_train_loss.png?raw=true "Title")

#### Resnet18 | MLP (Radiomics only) | MLP (Radiomis + Metadata)
![Alt text](logs/resnet18_losses.png?raw=true "Title")
![Alt text](logs/mlp_radiomics_only.png?raw=true "Title")
![Alt text](logs/mlp_radiomics_with_metadata.png?raw=true "Title")

### Results

#### Segmentation Models 
Model Name | Validation Loss 
--- | --- |
UNet | 0.94 |
MaskRCNN | 0.72 |

##### Higest Iou (0.92) was acheived by UNet on an held-out external dataset.

#### Predictions

##### MaskRCNN (Image with GT Bbox, GT mask, Image with Predicted Bbox, Predicted Mask)
![Alt text](logs/sample_xray.png?raw=true "Title")
![Alt text](logs/gt_mask_1.png?raw=true "Title")
![Alt text](logs/sample_xray_predicted_bbox.png?raw=true "Title")
![Alt text](logs/predicted_mask_maskrcnn.png?raw=true "Title")

##### UNet (Image , GT mask, Predicted Mask)
![Alt text](logs/sample_xray_2.png?raw=true "Title")
![Alt text](logs/gt_mask_2.png?raw=true "Title")
![Alt text](logs/predicted_mask_unet.png?raw=true "Title")


#### Classifiction Models 

##### Resnet18
Metrics | Normal | Pneumonia | Covid  
--- | --- | --- | ---|
Accuracy | 0.85 | 0.87 | 0.95 |
Sensitivity | 0.87 | 0.64 | 1.0 |
Specificity | 0.84 | 0.98 | 0.92 |
Precision | 0.75 | 0.96 | 0.86 |

##### MLP (Radiomics only)
Metrics | Nomral | Pneumonia | Covid  
--- | --- | --- | ---|
Accuracy | 0.87 | 0.74 | 0.77 |
Sensitivity | 0.85 | 0.74 | 0.45 |
Specificity | 0.87 | 0.73 | 0.91 |
Precision | 0.78 | 0.60 | 0.70 |

##### MLP (Radiomics with Metadata)
Metrics | Nomral | Pneumonia | Covid  
--- | --- | --- | ---|
Accuracy | 0.86 | 0.84 | 0.96 |
Sensitivity | 0.81 | 0.80 | 0.87 |
Specificity | 0.88 | 0.86 | 1.0 |
Precision | 0.76 | 0.75 | 1.0 |