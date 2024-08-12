# Species Identification and Coverage Estimation from Ground-level Imagery for Vegetation Monitoring 
This repo includes a script to train a YOLO frame detection model and a YOLO species segmentation model based on labelled ground-level imagery. 
Additionally, the script includes a function to use these two models to estimate the percentage coverage per species. 

The code has been tested on a Windows machine and Python 3.10

![image](https://github.com/user-attachments/assets/95afb424-5b7e-4cdf-9b55-7b38a8614afd)


## **1. Label ground-level images**
- This can for example be done on TrainYOLO.com
- YOLO project folder must have specific contents
  - root
    - images
      - img_1.jpeg
      - img_2.jpeg
      - ...
    - labels
      - img_1.txt
      - img_2.txt
      - ...
    - dataset.yaml
      - names:
        - 0: class1
        - 1: class2
        - ...
      - path: /content/data/project_title
      - train: train.txt
      - val: val.txt
    - train.txt
      - ./images/img_1.jpg
      - ...
    - val.txt
      - ./images/img_2.jpg
      - ...


## **2. Clone repo & create environment***
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


