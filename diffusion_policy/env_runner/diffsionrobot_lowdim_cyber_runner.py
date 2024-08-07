import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv




from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

import zarr, time

from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

# task = 'hop'
task = 'trot'
# task = 'cyber2_stand_dance_aug'


class LeggedRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            keypoint_visible_rate=1.0,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            n_latency_steps=0,
            fps=10,
            crf=22,
            agent_keypoints=False,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            device=None,
        ):
        super().__init__(output_dir)
        
        env_cfg, train_cfg = task_registry.get_cfgs(name=task)
        # override some parameters for testing
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)

        train_cfg.runner.amp_num_preload_transitions = 1

        # prepare environment        
        self.save_zarr = False
        
        if self.save_zarr:
            env_cfg.env.num_envs = 64
        else: # placeholder
            env_cfg.env.num_envs = 1

        
        # breakpoint()
        env, _ = task_registry.make_env(name=task, args=None, env_cfg=env_cfg)
        
        self.env = env
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
    
    def run(self, policy: BaseLowdimPolicy, online=False, generate_data=False):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        env.max_episode_length = int(env.max_episode_length)

        # plan for rollout
        obs, _ = env.reset()
        past_action = None
        
        expert_policy = torch.load('source_ckpts/{}.pt'.format("amp_policy"), map_location=torch.device('cpu'))
        expert_policy = expert_policy.to(device)

        pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval IsaacGym", 
            leave=False, mininterval=self.tqdm_interval_sec)
        
        history = self.n_obs_steps
        
        state_history = torch.zeros((env.num_envs, history+1, 45), dtype=torch.float32, device=device)
        action_history = torch.zeros((env.num_envs, history, env.num_actions), dtype=torch.float32, device=device)
        
        # state_history[:,:,:] = obs[:,None,:]
        state_history[:,:,:] = env.get_diffusion_observation().to(device)[:, None, :] # (env.num_envs, 1, env.num_observations)
        
        obs_dict = {"obs": state_history[:, :]} #, 'past_action': action_history}
        single_obs_dict = {"obs": state_history[:, -1, :].to("cuda:0")} #, 'past_action': action_history[0]}
        
        
        save_zarr = generate_data or (not online)
        len_to_save = 1200 if not generate_data else 5e5
        print("length to save", len_to_save)
        if save_zarr:
            
            if generate_data:
                zroot = zarr.open_group("recorded_data_{}_{}.zarr".format(task, time.strftime("%H-%M-%S", time.localtime())), "w")
            else:
                file_name = "recorded_data{}_eval.zarr".format(time.strftime("%H-%M-%S", time.localtime()))
                zroot = zarr.open_group(file_name, "w")
            
            zroot.create_group("data")
            zdata = zroot["data"]
            
            zroot.create_group("meta")
            zmeta = zroot["meta"]
            
            zmeta.create_group("episode_ends")
            
            zdata.create_group("action")
            zdata.create_group("state")
            
            recorded_obs = []
            recorded_acs = []
            
            recorded_obs_episode = np.zeros((env.num_envs, env.max_episode_length+2, 45))
            recorded_acs_episode = np.zeros((env.num_envs, env.max_episode_length+3, env.num_actions))
            
            
        episode_ends = []
        action_error = []
        idx = 0    
        saved_idx = 0    
        skip = 5
        while True:
            # run policy
            with torch.no_grad():
                expert_action = expert_policy.act_inference(obs.detach())

                # if (task == 'hop' or task == 'bounce'):
                #     expert_action = env.get_diffusion_action(expert_action)

                # if idx % skip == 4: #not save_zarr and 
                if online:    
                    # if idx < 200:
                    state_history[:, -policy.n_obs_steps-1:-1, 6:9] = 0
                    state_history[:, -policy.n_obs_steps-1:-1, 6] = 0#-0.5
                    state_history[:, -policy.n_obs_steps-1:-1, 7] = 0
                    # print(state_history[:, :, :])
                    obs_dict = {"obs": state_history[:, -policy.n_obs_steps-1:-1, :]}
                    t1 = time.perf_counter()
                    action_dict = policy.predict_action(obs_dict)
                    t2 = time.perf_counter()
                    print("time spent diffusion step: ", t2-t1)

                    # try:
                    #     action_dict = policy.predict_action_init(obs_dict, action_dict["action_pred"][:,8:12,:])
                    # except Exception as e:
                    #     obs_dict = {"obs": state_history[:, -9:-5, :]}
                    #     action_dict = policy.predict_action(obs_dict)
                    
                    pred_action = action_dict["action_pred"]
                    action = pred_action[:,history:history+2,:]
                    # if action.shape[1] == 0:
                    #     action = pred_action[:,-1:,:] # BC policy
                else:
                    action = expert_action[:, None, :]
            if save_zarr:
                curr_idx = np.all(recorded_obs_episode == 0, axis=-1).argmax(axis=-1)
                # curr_idx = idx
                recorded_obs_episode[np.arange(env.num_envs),curr_idx,:] = single_obs_dict["obs"].to("cpu").detach().numpy()
                recorded_acs_episode[np.arange(env.num_envs),curr_idx,:] = expert_action.to("cpu").detach().numpy()

            # step env
            self.n_action_steps = action.shape[1]
            for i in range(self.n_action_steps):
                action_step = action[:, i, :]
                if (task == 'hop' or task == 'bounce'):
                    action_step = env.get_diffusion_action(action_step)
                
                obs, _, rews, done, infos = env.step(action_step)
            
                state_history = torch.roll(state_history, shifts=-1, dims=1)
                action_history = torch.roll(action_history, shifts=-1, dims=1)
                
                state_history[:, -1, :] = env.get_diffusion_observation().to(device)
                action_history[:, -1, :] = action_step
                single_obs_dict = {"obs": state_history[:, -1, :].to("cuda:0")}
            
                idx += 1
            # reset env
            env_ids = torch.nonzero(done, as_tuple=False).squeeze(1).int()
            if len(env_ids) > 0:
                state_history[env_ids,:,:] = single_obs_dict["obs"][env_ids].to(state_history.device)[:,None,:]
                action_history[env_ids,:,:] = 0.0
                
                idx = 0
                
                # flush saved data
                if save_zarr:
                    for i in range(len(env_ids)):
                        epi_len = np.all(recorded_obs_episode[env_ids[i]] == 0, axis=-1).argmax(axis=-1)
                        if epi_len == 0:
                            epi_len = recorded_acs_episode.shape[1]
                        recorded_obs.append(np.copy(recorded_obs_episode[env_ids[i], :epi_len]))
                        recorded_acs.append(np.copy(recorded_acs_episode[env_ids[i], :epi_len]))
                        
                        recorded_obs_episode[env_ids[i]] = 0
                        recorded_acs_episode[env_ids[i]] = 0
                        
                        saved_idx += epi_len
                        episode_ends.append(saved_idx)
                        
                        print("saved_idx: ", saved_idx)
                    
            done = done.cpu().numpy()
            done = np.all(done)
            past_action = action

            # update pbar
            if online:
                pbar.update(action.shape[1])
            else:
                pbar.update(env.num_envs)
            
            if save_zarr and saved_idx >= len_to_save:
                recorded_obs = np.concatenate(recorded_obs)
                recorded_acs = np.concatenate(recorded_acs)
                episode_ends = np.array(episode_ends)
                
                zdata["state"] = recorded_obs
                zdata["action"] = recorded_acs
                zmeta["episode_ends"] = episode_ends
                print(zroot.tree())
                if generate_data:
                    raise StopIteration
                break
            # elif not save_zarr and idx > 300:
            #     break
            
        # clear out video buffer
        _ = env.reset()
        

        return file_name

