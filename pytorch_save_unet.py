"""
Usage:
python pytorch_save.py --checkpoint data/outputs/2023.11.06/10.23.44_train_diffusion_transformer_lowdim_pusht_lowdim/checkpoints/epoch=0000-test_mean_score=1.000.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os


import numpy as np
import click
import hydra
import torch
import torch.onnx
import dill
from omegaconf import OmegaConf
import onnxruntime
import tensorrt as trt
from cuda import cudart


from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cpu')
@click.option('-n', '--name', default="go1_unet")
def main(checkpoint, output_dir, device, name):

    model_name = name

    onnx_file = f"./checkpoint/{model_name}.onnx"
    trt_file = f"./checkpoint/{model_name}.plan"


    
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
    cond = torch.rand((1, 336), dtype=torch.float32, device=device)

    model_output = model(sample, timestep, 
        local_cond=None, global_cond=cond)
    torch.save(model, f"./checkpoint/{model_name}.pt")

    config_dict = {'horizon': cfg['policy']['horizon'], 
                'n_obs_steps': cfg['policy']['n_obs_steps'],
                'num_inference_steps': cfg['policy']['num_inference_steps'],
                }
    normalizer_ckpt = {k: v for k, v in payload['state_dicts']['model'].items() if "normalizer" in k}
    torch.save((config_dict, normalizer_ckpt), f"./checkpoint/{model_name}_config_dict.pt")



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

    # ort_session = onnxruntime.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])

    # # compute ONNX Runtime output prediction
    # ort_inputs = {
    #     ort_session.get_inputs()[0].name: sample.detach().cpu().numpy(),
    #     ort_session.get_inputs()[1].name: timestep.detach().cpu().numpy(),
    #     ort_session.get_inputs()[2].name: cond.detach().cpu().numpy(),
    #     }
    # ort_outs = ort_session.run(None, ort_inputs)
    
    # np.testing.assert_allclose(torch_out.detach().cpu().numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)

    # print("test passed")



    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True



    # for FP16 mode
    bUseFP16Mode = False
    # for INT8 model
    bUseINT8Mode = False
    nCalibration = 1
    cacheFile = "./int8.cache"

    # os.system("rm -rf ./*.onnx ./*.plan ./*.cache")
    np.set_printoptions(precision=3, linewidth=200, suppress=True)
    cudart.cudaDeviceSynchronize()



    # Parse network, rebuild network and do inference in TensorRT ------------------

    logger = trt.Logger(trt.Logger.VERBOSE)
    # logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    if bUseFP16Mode:
        config.set_flag(trt.BuilderFlag.FP16)
    if bUseINT8Mode:
        import calibrator
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator.MyCalibrator(nCalibration)

    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnx_file):
        print("Failed finding ONNX file!")
        exit()
    print("Succeeded finding ONNX file!")
    with open(onnx_file, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")

    inputTensor_sample = network.get_input(0)
    inputTensor_time = network.get_input(1)
    inputTensor_cond = network.get_input(2)
    opt_shape = [sample.shape[0], sample.shape[1], sample.shape[2]]
    profile.set_shape(inputTensor_sample.name, opt_shape, opt_shape, opt_shape)
    opt_shape = [timestep.shape[0]]
    profile.set_shape(inputTensor_time.name, opt_shape, opt_shape, opt_shape)
    opt_shape = [cond.shape[0], cond.shape[1]]
    profile.set_shape(inputTensor_cond.name, opt_shape, opt_shape, opt_shape)
    config.add_optimization_profile(profile)

    #network.unmark_output(network.get_output(0))  # remove output tensor "y"
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(trt_file, "wb") as f:
        f.write(engineString)




if __name__ == '__main__':
    main()


