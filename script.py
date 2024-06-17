import ultralytics
from ultralytics import YOLO

import cv2
import numpy as np
import os
import pandas as pd
import shutil
import random

# ******************************************************************************
### Train frame detegtion model ### 
# path to training data for frame detection model 
path_to_data_frame="path_to/project_location" # path to the root where you have the yaml and the images and labels subfolder

# train frame detegtion model 
trained_model_path_frame = train_yolo(path_to_data_frame, task=detect, mode=train, im_size=640, n_epochs=50, batch_size=4, model='yolov8n.pt')

# import function to train yolo model
from functions import train_yolo

# call function 
trained_model_path_frame = train_yolo(path_to_data_frame) 

# ******************************************************************************
### Train species segmentation model ###
# path to training data for species segmentation model 
path_to_data_species="path_to/project_location" # path to the root where you have the yaml and the images and labels subfolder

# train species segmentation model 
trained_model_path_species = train_yolo(path_to_data_species, mode=segment, task=detect, mode=train, im_size=640, n_epochs=50, batch_size=4, model='yolov8n.pt'))

# ******************************************************************************
### Predict percentage coverage per species ###
# path to raw images 
path_to_data_images = "path_to/images" # path to the folder containing the images taking in the field

# predict on raw images 
predict(path_to_data_images, path_to_model_frame=trained_model_path_frame, name_frame_class="frame", # parameter for cropping to frame
            path_to_model_species=trained_model_path_species, conf_treshold=0.1,                     # parameter for species segmenation
            number_of_classes=2):                                                                    # parameter for calculating percentage cover
