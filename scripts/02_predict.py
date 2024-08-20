import ultralytics
from ultralytics import YOLO

import cv2
import numpy as np
import os
import pandas as pd
import shutil
import random

# import function to predict on raw images and estimate percentage cover per species 
from functions import predict_cover

###############################################
### Predict percentage coverage per species ###
###############################################

# path to raw images 
path_to_data_images = "path_to/images" # path to the folder containing the photos taken in the field

# predict on raw images 
predict_cover(path_to_data_images, path_to_model_frame=trained_model_path_frame, name_frame_class="frame", # parameter for cropping to frame
            path_to_model_species=trained_model_path_species, conf_treshold=0.10,                     # parameter for species segmentation
            number_of_classes=2)                                                                    # parameter for calculating percentage cover

 """
    Predict with YOLO models with the specified parameters:

    Parameters for predicting with frame detection model:
    - path_to_data_images (str): Path to the folder with orignal images taken in the field.
    - path to model  frame (str): Path to the YOLO model which crops orignal images to the inner frame.
    - name_frame_class (str): Name of the class representing the frame in the frame detection model. Default is "frame".

    Parameters for predicting with species segmentation model:
    - path_to_model_species (str):  Path to the YOLO model which segments species.
    - conf_thresold (int): Confidence threshold for the prediction. Default is 0.0.08.

    Parameters for calculating percentage cover of the species:
    - number_of_classes (int): Total number of classes in species segmentation model. Default is 2.

    """
