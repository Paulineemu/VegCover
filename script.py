import ultralytics
from ultralytics import YOLO

import cv2
import numpy as np
import os
import pandas as pd
import shutil
import random

# import function to predict on raw images and estimate percentage cover per species 
from functions import predict 

# ******************************************************************************
### Train frame detegtion model ### 
# path to training data for frame detection model 
path_to_data_frame="path_to/project_location" # path to the root where you have the yaml and the images and labels subfolder

# Structure of the folder for training the model

-root
      -images
              -img_1.jpeg
              -img_2.jpeg
              -...
      -labels
              -img_1.txt
              -img_2.txt
              -...
      -dataset.yaml
              -names:
                0: class1
                1: class2
                ...
              -path: /content/data/project_title
              -train: train.txt
              -val: val.txt
      -train.txt
              -./images/img_1.jpg
      -val.txt
              -./images/img_2.jpg

# Set YOLO parameters
IM_SIZE = 640 # Image size for training
N_EPOCHS = 50 # Number of epochs for training
BATCH_SIZE = 4 # Batch size for training
MODEL = 'yolov8n.pt' # Choose pretrained YOLO model version and size

# Train frame detection model
!yolo task=detect mode=train model=$MODEL data="{path_to_data_frame}/dataset.yaml", epochs=$N_EPOCHS, imgsz=$IM_SIZE, batch=$BATCH_SIZE, project=path_to_data_frame, name="species_detector_imgsz"+str(IM_SIZE)+"_modelSizeN"

# Construct the path to the trained model's best weights
trained_model_path_frame = f'{path_to_data_frame}/species_detector_imgsz{IM_SIZE}_modelSizeN/weights/best.pt'

# ******************************************************************************
### Train species segmentation model ###
# path to training data for species segmentation model 
path_to_data_species="path_to/project_location" # path to the root where you have the yaml and the images and labels subfolder

# Train species segmentation model
!yolo task=segment mode=train model=$MODEL data="{path_to_data_species}/dataset.yaml", epochs=$N_EPOCHS, imgsz=$IM_SIZE, batch=$BATCH_SIZE, project=path_to_data_species, name="species_detector_imgsz"+str(IM_SIZE)+"_modelSizeN"

# Construct the path to the trained model's best weights
trained_model_path_species = f'{path_to_data_species}/species_detector_imgsz{IM_SIZE}_modelSizeN/weights/best.pt'

# ******************************************************************************
### Predict percentage coverage per species ###
# path to raw images 
path_to_data_images = "path_to/images" # path to the folder containing the images taking in the field

# predict on raw images 
predict(path_to_data_images, path_to_model_frame=trained_model_path_frame, name_frame_class="frame", # parameter for cropping to frame
            path_to_model_species=trained_model_path_species, conf_treshold=0.1,                     # parameter for species segmenation
            number_of_classes=2):                                                                    # parameter for calculating percentage cover
