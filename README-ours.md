# DiffuseLoco: Real-Time Legged Locomotion Control with Diffusion from Offline Datasets

Codebase for the "DiffuseLoco: Real-Time Legged Locomotion Control with Diffusion from Offline Datasets" project. This repository contains the code necessary to train the DiffuseLoco policy from an offline dataset with diverse skills and prepare the model for deployment.


## Installation

### Simulation

First, create the conda environment:

```bash
conda create -n diff python=3.8
```

Then, install the python dependencies:

```bash
pip install -r requirements.txt
```

#### Export the result

```bash
python pytorch_save.py --checkpoint "checkpoints/latest.ckpt" --output_dir ""
```


# Eval

conda config --env --add channels robostack

mamba install ros-noetic-ros-base

```bash
python eval.py --checkpoint=data/outputs/2023.11.04/08.17.36_train_diffusion_unet_lowdim_pusht_lowdim/checkpoints/latest.ckpt -o eval_output_dir/
```

### Real Robot


