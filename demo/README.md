# Demo instructions 

This demo allows you to perform the prediction and validation code on given data using given YOLO models. 

**1. Clone repo & create environment***
```
# clone repo
git clone https://github.com/Paulineemu/DeepField-Deep-Learning-for-Ground-Level-Vegetation-Monitoring.git
cd DeepField-Deep-Learning-for-Ground-Level-Vegetation-Monitoring

# create new environment
conda env create -f DeepField.yaml

# activate the created environment
conda activate DeepField

# install requirements
pip install -r requirements.txt
```
**2. Download images & models**
  - The Google Drive folder reached by the link in data&model_download.txt contains the images you can predict on and the two models:
    - frame detection model
    - species segmentation model

**3. Run predict.py**
  - insert paths to ...
    - your_path/Images_field
    - your_path/Frame_detection_model.pt
    - your_path/Species_segmentation_model.pt
  - ... in the function

**4. Run eval.py**
  - insert paths to ...
    - field data - data frame with coverage estimations from fieldworkers (serves as ground truth)
    - predictions - data frame with coverage predictions
  - Set ...
    - image_ID_column - Column name which exists in both datasets - 'Image_new'
    - classes_to_include - column names of classes to include in the validation (must be the same in field and prediction dataset)
    - names_of_classes - ['Blueberry', 'Lingonberry']
    - colours - ['blue', 'red']

