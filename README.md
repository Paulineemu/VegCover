# 🌿 Species Identification and Coverage Estimation from Ground-level Imagery for Vegetation Monitoring 📷
This repository contains the data and code used in __[Müller, Puliti, and Breidenbach (2025)](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.70024)__ to train and apply deep learning models for __species coverage estimation__ using ground-level imagery.

It includes: 

✅ A YOLOv8 object detection model for detecting frames in images and one species instance segmentation model for species identification and identifying and segmenting species.

✅ Method to parse instance segmentation masks to species-specific coverage estimates in images.

✅ The data used to train and evaluate the models. The full datasets can be downloaded in this Zenodo repository https://zenodo.org/records/13361905

![Figure1_new](https://github.com/user-attachments/assets/647843f8-7b76-4c51-8ca5-764984b02264)


# 👩‍🚀 Workflow Overview
This demo provides a step-by-step approach for training and applying the models:

1️⃣ __Training:__
- Train two models using labeled images:
  -   __Frame Object Detection__ (dataset: Frame_data)
  -   __Species Instance Segmentation__ (dataset: Species_segmentation_data)
 
2️⃣ __Confidence Optimization:__
- Optimize the confidence threshold based on downstream __cover estimation__ performance.

3️⃣ __Inference:__
- Predict on test images (Species_cover_data_test).

4️⃣ __Evaluation:__
- Compare predictions with field estimates (Field_data_NFI).


📌 The code has been tested on a Windows machine and Python 3.10. 

# 🔧 How to Run the Demo
Follow these steps to set up and run demo.ipynb:

```
# Create a new environment
conda create -n VegCover python=3.10

# Activate the environment
conda activate VegCover

# Install dependencies
pip install -r requirements.txt

# Install Jupyter Lab
pip install jupyterlab

# Open the demo notebook
jupyter-lab

```

# 📖 How to cite

If you use this work, please cite:

__Müller, P., Puliti, S., & Breidenbach, J. (2025).__ Towards Enhancing Field-Based Vegetation Monitoring: A Deep Learning Approach for Species Coverage Estimation from Ground-Level Imagery. _Methods in Ecology and Evolution._




