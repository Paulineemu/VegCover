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
