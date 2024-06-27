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

###############################################
### Predict percentage coverage per species ###
###############################################

# path to raw images 
path_to_data_images = "path_to/images" # path to the folder containing the photos taken in the field

# predict on raw images 
predict(path_to_data_images, path_to_model_frame=trained_model_path_frame, name_frame_class="frame", # parameter for cropping to frame
            path_to_model_species=trained_model_path_species, conf_treshold=0.1,                     # parameter for species segmentation
            number_of_classes=2):                                                                    # parameter for calculating percentage cover
