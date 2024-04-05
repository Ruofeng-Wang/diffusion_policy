"""
Usage:
python pytorch_save.py --checkpoint data/outputs/2023.11.06/10.23.44_train_diffusion_transformer_lowdim_pusht_lowdim/checkpoints/epoch=0000-test_mean_score=1.000.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

# from isaacgym.torch_utils import *

import numpy as np
import click
import hydra
import torch
import torch.onnx
import dill
from omegaconf import OmegaConf
import onnxruntime

from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cpu')
def main(checkpoint, output_dir, device):

    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    action_steps = 4            # Maybe you need to change this accordingly
    cfg['task']['env_runner']['_target_'] = 'diffusion_policy.env_runner.diffsionrobot_lowdim_isaac_runner.IsaacHumanoidRunner'
    cfg['n_action_steps'] = action_steps
    cfg['task']['env_runner']['n_action_steps'] = action_steps
    cfg['policy']['n_action_steps'] = action_steps
    OmegaConf.set_struct(cfg, False)



    cfg.task.env_runner['device'] = device
    
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    model = policy.model

    model = model.eval()

    sample = torch.rand((1, 12, 12), dtype=torch.float32, device=device)
    timestep = torch.rand((1, ), dtype=torch.float32, device=device)
    cond = torch.rand((1, 8, 42), dtype=torch.float32, device=device)

    torch_out = model.forward(sample, timestep, cond)

    torch.save(model, "./go1_ckpts/converted_model.pt")


    onnx_file = "./go1_ckpts/model.onnx"

    # model = torch.load("model_full.pt")


    # Export model as ONNX file ----------------------------------------------------
    torch.onnx.export(
        model, 
        (sample, timestep, cond),
        onnx_file, 
        input_names=["sample", "timestep", "cond"], 
        output_names=["action"], 
        do_constant_folding=True, 
        verbose=True, 
        keep_initializers_as_inputs=True, 
        opset_version=17, 
        dynamic_axes={}
        )

    print("Succeeded converting model into ONNX!")

    ort_session = onnxruntime.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])

    # compute ONNX Runtime output prediction
    ort_inputs = {
        ort_session.get_inputs()[0].name: sample.detach().cpu().numpy(),
        ort_session.get_inputs()[1].name: timestep.detach().cpu().numpy(),
        ort_session.get_inputs()[2].name: cond.detach().cpu().numpy(),
        }
    ort_outs = ort_session.run(None, ort_inputs)
    
    np.testing.assert_allclose(torch_out.detach().cpu().numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("test passed")



if __name__ == '__main__':
    main()


