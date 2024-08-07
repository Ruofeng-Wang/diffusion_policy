"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

try:
    from isaacgym.torch_utils import *
except ImportError as e:
    print("Error encountered when importing gym. Check gym installation?")
    print(e)
    

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from omegaconf import OmegaConf
import torch.nn.utils.prune as prune
import numpy as np

from diffusion_policy.workspace.base_workspace import BaseWorkspace

from diffusion_policy.env_runner.diffsionrobot_lowdim_cyber_runner import LeggedRunner


import matplotlib.pyplot as plt

def plot_tensor_distribution(tensor, title=""):
    # Calculate the magnitudes of the elements in the tensor
    magnitudes = tensor.abs().flatten().cpu().detach().numpy()

    # Sort the magnitudes
    sorted_magnitudes = np.sort(magnitudes)
    # reverse the order
    sorted_magnitudes = sorted_magnitudes[::-1]
    
    # Calculate the cumulative percentage
    cumulative_percentage = np.arange(1, len(sorted_magnitudes) + 1) / len(sorted_magnitudes) * 100

    # Plot the distribution
    #set x axis range

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_magnitudes, cumulative_percentage, label='Cumulative Distribution')
    plt.xlabel("abs(elem)")
    plt.xlim(-0.01, torch.max(tensor.abs()).item() + 0.01)
    plt.ylabel("Cumulative Percentage")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    # save
    plt.savefig("figs/"+title + ".png")




@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("--device", default="cuda:0")
@click.option("--online", default=True)
@click.option("--generate_data", default=False)
@click.option("--headless", default=False)
def main(checkpoint, output_dir, device, **kwargs):
    # if os.path.exists(output_dir):
    #     click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    # action_steps = 4

    print(cfg['task']['env_runner']['_target_'], '\n\n')
    cfg['task']['env_runner']['_target_'] = 'diffusion_policy.env_runner.diffsionrobot_lowdim_cyber_runner.LeggedRunner'
    print(cfg['task']['env_runner']['n_obs_steps'])

    # cfg['n_action_steps'] = action_steps
    # cfg['task']['env_runner']['n_action_steps'] = action_steps
    # cfg['policy']['n_action_steps'] = action_steps
    OmegaConf.set_struct(cfg, False)

    cfg.task.env_runner['device'] = device
    
    cls = hydra.utils.get_class(cfg._target_)
    print(f"Loading workspace {cls.__name__}")
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    
    sparse_ratio = 0.8
    # plot_tensor_distribution(policy.model.encoder[0].weight, title="policy.model.encoder[0].weight")
    # plot_tensor_distribution(policy.model.encoder[2].weight, title="policy.model.encoder[2].weight")
    

    prune.l1_unstructured(policy.model.encoder[0], name="weight", amount=0.5)
    # prune.l1_unstructured(policy.model.encoder[0], name="bias", amount=sparse_ratio)
    prune.l1_unstructured(policy.model.encoder[2], name="weight", amount=0.5)
    # prune.l1_unstructured(policy.model.encoder[2], name="bias", amount=sparse_ratio)
    prune.remove(policy.model.encoder[0], name="weight")
    # prune.remove(policy.model.encoder[0], name="bias")
    prune.remove(policy.model.encoder[2], name="weight")
    # prune.remove(policy.model.encoder[2], name="bias")


    
    for i in range(6):
        prune.l1_unstructured(policy.model.decoder.layers[i].self_attn.out_proj, name="weight", amount=sparse_ratio)
        # prune.l1_unstructured(policy.model.decoder.layers[i].self_attn.out_proj, name="bias", amount=sparse_ratio)
        prune.l1_unstructured(policy.model.decoder.layers[i].multihead_attn.out_proj, name="weight", amount=0.5)
        # prune.l1_unstructured(policy.model.decoder.layers[i].multihead_attn.out_proj, name="bias", amount=sparse_ratio)
        prune.l1_unstructured(policy.model.decoder.layers[i].linear1, name="weight", amount=0.5)
        # prune.l1_unstructured(policy.model.decoder.layers[i].linear1, name="bias", amount=sparse_ratio)
        prune.l1_unstructured(policy.model.decoder.layers[i].linear2, name="weight", amount=sparse_ratio)
        # prune.l1_unstructured(policy.model.decoder.layers[i].linear2, name="bias", amount=sparse_ratio)
        # prune.l1_unstructured(policy.model.decoder.layers[i].norm1, name="weight", amount=sparse_ratio)
        # prune.l1_unstructured(policy.model.decoder.layers[i].norm1, name="bias", amount=sparse_ratio)
        # prune.l1_unstructured(policy.model.decoder.layers[i].norm2, name="weight", amount=sparse_ratio)
        # prune.l1_unstructured(policy.model.decoder.layers[i].norm2, name="bias", amount=sparse_ratio)
        # prune.l1_unstructured(policy.model.decoder.layers[i].norm3, name="weight", amount=sparse_ratio)
        # prune.l1_unstructured(policy.model.decoder.layers[i].norm3, name="bias", amount=sparse_ratio)
        prune.remove(policy.model.decoder.layers[i].self_attn.out_proj, name="weight")
        # prune.remove(policy.model.decoder.layers[i].self_attn.out_proj, name="bias")
        prune.remove(policy.model.decoder.layers[i].multihead_attn.out_proj, name="weight")
        # prune.remove(policy.model.decoder.layers[i].multihead_attn.out_proj, name="bias")
        prune.remove(policy.model.decoder.layers[i].linear1, name="weight")
        # prune.remove(policy.model.decoder.layers[i].linear1, name="bias")
        prune.remove(policy.model.decoder.layers[i].linear2, name="weight")
        # prune.remove(policy.model.decoder.layers[i].linear2, name="bias")
        # prune.remove(policy.model.decoder.layers[i].norm1, name="weight")
        # prune.remove(policy.model.decoder.layers[i].norm1, name="bias")
        # prune.remove(policy.model.decoder.layers[i].norm2, name="weight")
        # prune.remove(policy.model.decoder.layers[i].norm2, name="bias")
        # prune.remove(policy.model.decoder.layers[i].norm3, name="weight")
        # prune.remove(policy.model.decoder.layers[i].norm3, name="bias")
    
        
    """
    odict_keys(['_dummy_variable', 'model._dummy_variable', 'model.pos_emb', 'model.cond_pos_emb', 'model.mask', 'model.memory_mask', 'model.input_emb.weight', 'model.input_emb.bias', 'model.cond_obs_emb.weight', 'model.cond_obs_emb.bias', 'model.cond_obs_emb_2.weight', 'model.cond_obs_emb_2.bias', 'model.encoder.0.weight', 'model.encoder.0.bias', 'model.encoder.2.weight', 'model.encoder.2.bias', 'model.decoder.layers.0.self_attn.in_proj_weight', 'model.decoder.layers.0.self_attn.in_proj_bias', 'model.decoder.layers.0.self_attn.out_proj.bias', 'model.decoder.layers.0.self_attn.out_proj.weight', 'model.decoder.layers.0.multihead_attn.in_proj_weight', 'model.decoder.layers.0.multihead_attn.in_proj_bias', 'model.decoder.layers.0.multihead_attn.out_proj.bias', 'model.decoder.layers.0.multihead_attn.out_proj.weight', 'model.decoder.layers.0.linear1.bias', 'model.decoder.layers.0.linear1.weight', 'model.decoder.layers.0.linear2.bias', 'model.decoder.layers.0.linear2.weight', 'model.decoder.layers.0.norm1.weight', 'model.decoder.layers.0.norm1.bias', 'model.decoder.layers.0.norm2.weight', 'model.decoder.layers.0.norm2.bias', 'model.decoder.layers.0.norm3.weight', 'model.decoder.layers.0.norm3.bias', 'model.decoder.layers.1.self_attn.in_proj_weight', 'model.decoder.layers.1.self_attn.in_proj_bias', 'model.decoder.layers.1.self_attn.out_proj.bias', 'model.decoder.layers.1.self_attn.out_proj.weight', 'model.decoder.layers.1.multihead_attn.in_proj_weight', 'model.decoder.layers.1.multihead_attn.in_proj_bias', 'model.decoder.layers.1.multihead_attn.out_proj.bias', 'model.decoder.layers.1.multihead_attn.out_proj.weight', 'model.decoder.layers.1.linear1.bias', 'model.decoder.layers.1.linear1.weight', 'model.decoder.layers.1.linear2.bias', 'model.decoder.layers.1.linear2.weight', 'model.decoder.layers.1.norm1.weight', 'model.decoder.layers.1.norm1.bias', 'model.decoder.layers.1.norm2.weight', 'model.decoder.layers.1.norm2.bias', 'model.decoder.layers.1.norm3.weight', 'model.decoder.layers.1.norm3.bias', 'model.decoder.layers.2.self_attn.in_proj_weight', 'model.decoder.layers.2.self_attn.in_proj_bias', 'model.decoder.layers.2.self_attn.out_proj.bias', 'model.decoder.layers.2.self_attn.out_proj.weight', 'model.decoder.layers.2.multihead_attn.in_proj_weight', 'model.decoder.layers.2.multihead_attn.in_proj_bias', 'model.decoder.layers.2.multihead_attn.out_proj.bias', 'model.decoder.layers.2.multihead_attn.out_proj.weight', 'model.decoder.layers.2.linear1.bias', 'model.decoder.layers.2.linear1.weight', 'model.decoder.layers.2.linear2.bias', 'model.decoder.layers.2.linear2.weight', 'model.decoder.layers.2.norm1.weight', 'model.decoder.layers.2.norm1.bias', 'model.decoder.layers.2.norm2.weight', 'model.decoder.layers.2.norm2.bias', 'model.decoder.layers.2.norm3.weight', 'model.decoder.layers.2.norm3.bias', 'model.decoder.layers.3.self_attn.in_proj_weight', 'model.decoder.layers.3.self_attn.in_proj_bias', 'model.decoder.layers.3.self_attn.out_proj.bias', 'model.decoder.layers.3.self_attn.out_proj.weight', 'model.decoder.layers.3.multihead_attn.in_proj_weight', 'model.decoder.layers.3.multihead_attn.in_proj_bias', 'model.decoder.layers.3.multihead_attn.out_proj.bias', 'model.decoder.layers.3.multihead_attn.out_proj.weight', 'model.decoder.layers.3.linear1.bias', 'model.decoder.layers.3.linear1.weight', 'model.decoder.layers.3.linear2.bias', 'model.decoder.layers.3.linear2.weight', 'model.decoder.layers.3.norm1.weight', 'model.decoder.layers.3.norm1.bias', 'model.decoder.layers.3.norm2.weight', 'model.decoder.layers.3.norm2.bias', 'model.decoder.layers.3.norm3.weight', 'model.decoder.layers.3.norm3.bias', 'model.decoder.layers.4.self_attn.in_proj_weight', 'model.decoder.layers.4.self_attn.in_proj_bias', 'model.decoder.layers.4.self_attn.out_proj.bias', 'model.decoder.layers.4.self_attn.out_proj.weight', 'model.decoder.layers.4.multihead_attn.in_proj_weight', 'model.decoder.layers.4.multihead_attn.in_proj_bias', 'model.decoder.layers.4.multihead_attn.out_proj.bias', 'model.decoder.layers.4.multihead_attn.out_proj.weight', 'model.decoder.layers.4.linear1.bias', 'model.decoder.layers.4.linear1.weight', 'model.decoder.layers.4.linear2.bias', 'model.decoder.layers.4.linear2.weight', 'model.decoder.layers.4.norm1.weight', 'model.decoder.layers.4.norm1.bias', 'model.decoder.layers.4.norm2.weight', 'model.decoder.layers.4.norm2.bias', 'model.decoder.layers.4.norm3.weight', 'model.decoder.layers.4.norm3.bias', 'model.decoder.layers.5.self_attn.in_proj_weight', 'model.decoder.layers.5.self_attn.in_proj_bias', 'model.decoder.layers.5.self_attn.out_proj.bias', 'model.decoder.layers.5.self_attn.out_proj.weight', 'model.decoder.layers.5.multihead_attn.in_proj_weight', 'model.decoder.layers.5.multihead_attn.in_proj_bias', 'model.decoder.layers.5.multihead_attn.out_proj.bias', 'model.decoder.layers.5.multihead_attn.out_proj.weight', 'model.decoder.layers.5.linear1.bias', 'model.decoder.layers.5.linear1.weight', 'model.decoder.layers.5.linear2.bias', 'model.decoder.layers.5.linear2.weight', 'model.decoder.layers.5.norm1.weight', 'model.decoder.layers.5.norm1.bias', 'model.decoder.layers.5.norm2.weight', 'model.decoder.layers.5.norm2.bias', 'model.decoder.layers.5.norm3.weight', 'model.decoder.layers.5.norm3.bias', 'model.ln_f.weight', 'model.ln_f.bias', 'model.head.weight', 'model.head.bias', 'mask_generator._dummy_variable', 'normalizer.params_dict.obs.offset', 'normalizer.params_dict.obs.scale', 'normalizer.params_dict.obs.input_stats.max', 'normalizer.params_dict.obs.input_stats.mean', 'normalizer.params_dict.obs.input_stats.min', 'normalizer.params_dict.obs.input_stats.std', 'normalizer.params_dict.action.offset', 'normalizer.params_dict.action.scale', 'normalizer.params_dict.action.input_stats.max', 'normalizer.params_dict.action.input_stats.mean', 'normalizer.params_dict.action.input_stats.min', 'normalizer.params_dict.action.input_stats.std'])
    """

    policy.half()
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy, online=True, generate_data=False)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
