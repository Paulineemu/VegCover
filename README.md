# Species Identification and Coverage Estimation from Ground-level Imagery for Vegetation Monitoring 
This repo includes a script to train a YOLO frame object detection model and a YOLO species instance segmentation model based on labelled ground-level imagery. 
Additionally, the script includes a function to use these two models to estimate the percentage coverage per species. 

The code has been tested on a Windows machine and Python 3.10. 


![Figure1_new](https://github.com/user-attachments/assets/b1288b93-58b9-42f2-a0df-f14faa592346)



## **Label ground-level images**
- This can for example be done on TrainYOLO.com
- YOLO project folder must have specific contents
  - root
    - train
        - img_1.jpeg
        - img_2.jpeg
        - labels_img_1.txt
        - labels_img_2.txt
    - val
        - img_3.jpeg
        - img_4.jpeg
        - labels_img_3.txt
        - labels_img_4.txt
    - dataset.yaml
        - names:
          - 0: Species 1
          - 1: Species 2
        - path: C:\Users\Anonym\VegCover\datasets\Frame_data
        - train: train
        - val: val


## **How to use the demo**
To open the demo.ipynb follow these steps:

```
# clone repo
git clone https://github.com/Paulineemu/VegCover.git
cd VegCover

# create new environment
conda create -n VegCover python=3.10

# activate the created environment
conda activate VegCover

# install requirements
pip install -r requirements.txt

# install jupyter-lab
pip install jupyterlab

# open demo.ipynb
jupyter-lab
```




