## Install Diffusion Policy

We need to use Python version 3.8 since Isaac Gym requires 3.8


```bash
conda create -p ./.conda-env/ python=3.8
```

```bash
conda install llvm-openmp
pip install -r requirements.txt
```

```bash
python pytorch_save.py --checkpoint "checkpoints/latest.ckpt" --output_dir ""
```


# Eval

conda config --env --add channels robostack

mamba install ros-noetic-ros-base

```bash
python eval.py --checkpoint=data/outputs/2023.11.04/08.17.36_train_diffusion_unet_lowdim_pusht_lowdim/checkpoints/latest.ckpt -o eval_output_dir/
```


## FAQ

### libpython3.8 error

```python
ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
```

#### Solution

```bash
export LD_LIBRARY_PATH=/home/tk/Documents/mambaforge/envs/calm/lib:$LD_LIBRARY_PATH
```
