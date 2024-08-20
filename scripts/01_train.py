import ultralytics
from ultralytics import YOLO

###################################
### Train frame detection model ###
###################################

# path to training data for frame detection model 
path_to_data_frame="path_to/project_location" # path to the root where you have the yaml and the images and labels subfolder

# Set YOLO parameters
IM_SIZE = 640 # Image size for training
N_EPOCHS = 50 # Number of epochs for training
BATCH_SIZE = 4 # Batch size for training
MODEL = 'yolov8n.pt' # Choose pretrained YOLO model version and size

# Train frame detection model
!yolo task=detect mode=train model=$MODEL data="{path_to_data_frame}/dataset.yaml", epochs=$N_EPOCHS, imgsz=$IM_SIZE, batch=$BATCH_SIZE, project="Models" name="Frame_detection_model"

# Construct the path to the trained model's best weights
trained_model_path_frame = f'{path_to_data_frame}/Frame_detection_model/weights/best.pt'

########################################
### Train species segmentation model ###
########################################

# path to training data for species segmentation model 
path_to_data_species="path_to/project_location" # path to the root where you have the yaml and the images and labels subfolder

# Train species segmentation model
!yolo task=segment mode=train model=$MODEL data="{path_to_data_species}/dataset.yaml", epochs=$N_EPOCHS, imgsz=$IM_SIZE, batch=$BATCH_SIZE, project="Models", name="Species_segmentation_model 

# Construct the path to the trained model's best weights
trained_model_path_species = "Models/Frame_detection_model/weights/best.pt"
