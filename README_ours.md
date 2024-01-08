## Install Diffusion Policy

We need to use Python version 3.8 since Isaac Gym requires 3.8

```bash
conda env create --name diff python=3.8
```

```bash
conda install llvm-openmp
```

```bash
pip install -r ./requirements.txt 
```



## Install AMP for HW

Install isaacgym


```bash
cd /rscratch/tk/diffusion/AMP_for_hardware/rsl_rl
pip install -e .
```

```bash
cd ..
pip install -e .
```

# Eval

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
