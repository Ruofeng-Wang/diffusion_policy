import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

import zarr, time

from baselines.common import tf_util as U
import tensorflow as tf
from cassie_env.cassie_env import CassieEnv
import time
import numpy as np
import argparse
import ppo.policies as policies 
from cassie_env import CASSIE_GYM_ROOT_DIR


library_folder = CASSIE_GYM_ROOT_DIR + '/motions/MotionLibrary/'
model_folder = CASSIE_GYM_ROOT_DIR + '/tf_model/'

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
        
        EP_LEN_MAX = 2000
        
        env = CassieEnv(max_timesteps=EP_LEN_MAX,
                        is_visual=False,
                        ref_file=library_folder+'GaitLibrary.gaitlib',
                        stage='single', 
                        method='baseline')

        env.num_envs = 1
        env.max_episode_length = EP_LEN_MAX

        model_dir = model_folder + 'trial3_baseline_dynrand_perturb_rnds1'
        latest_checkpoint = tf.train.latest_checkpoint(model_dir)
        model_path = latest_checkpoint
        config = tf.ConfigProto(device_count={'GPU': 0})

        ob_space_pol = env.observation_space_pol
        ac_space = env.action_space

        env.num_obs = 62 # obs base shape
        env.num_actions = env.action_space.shape[0]

        ob_space_vf = env.observation_space_vf
        ob_space_pol_cnn = env.observation_space_pol_cnn
        self.pi = policies.MLPCNNPolicy(name='pi', ob_space_vf=ob_space_vf, ob_space_pol=ob_space_pol, ob_space_pol_cnn=ob_space_pol_cnn, 
                                    ac_space=ac_space, hid_size=512, num_hid_layers=2)

        U.make_session(config=config)
        U.load_state(model_path)
        
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
        obs_vf, obs = env.reset()
        expert_policy = self.pi
        past_action = None
        


        pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval IsaacGym", 
            leave=False, mininterval=self.tqdm_interval_sec)
        done = False
        
        history = self.n_obs_steps
        state_history = torch.zeros((env.num_envs, history+1, env.num_obs), dtype=torch.float32, device=device)
        action_history = torch.zeros((env.num_envs, history, env.num_actions), dtype=torch.float32, device=device)
        
        # state_history[:,:,:] = obs[:,None,:]
        # state_history[:,:,:] = torch.from_numpy(obs).to(device)[:, None, :] # (env.num_envs, 1, env.num_observations)
        
        obs_dict = {"obs": state_history[:, :]} #, 'past_action': action_history}
        single_obs_dict = {"obs": state_history[:, -1, :].to("cuda:0")} #, 'past_action': action_history[0]}
        
        
        save_zarr = False #generate_data or (not online)
        len_to_save = 1e6 if not generate_data else 1e6
        print("length to save", len_to_save)
        if save_zarr:
            
            if generate_data:
                zroot = zarr.open_group("recorded_data{}.zarr".format(time.strftime("%H-%M-%S", time.localtime())), "w")
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
            
            recorded_obs_episode = np.zeros((env.num_envs, env.max_episode_length+2, env.num_obs))
            recorded_acs_episode = np.zeros((env.num_envs, env.max_episode_length+3, env.num_actions))
            
            
        episode_ends = []
        action_error = []
        idx = 0    
        saved_idx = 0    
        skip = 5
        t1 = time.perf_counter()
        while True:
            # run policy
            with torch.no_grad():
                expert_action = expert_policy.act(stochastic=False, ob_vf=obs_vf, ob_pol=obs)[0]
                if online:    
                    obs_dict = {"obs": state_history[:, -9:-1, :]}
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
                    action = pred_action[:,history:history+6,:]
                else:
                    action = expert_action[None, None, :]
            if save_zarr:
                curr_idx = np.all(recorded_obs_episode == 0, axis=-1).argmax(axis=-1)
                # curr_idx = idx
                recorded_obs_episode[np.arange(env.num_envs),curr_idx,:] = single_obs_dict["obs"].to("cpu").detach().numpy()
                recorded_acs_episode[np.arange(env.num_envs),curr_idx,:] = expert_action.to("cpu").detach().numpy()
    
            # step env
            self.n_action_steps = action.shape[1]
            for i in range(self.n_action_steps):
                action_step = action[:, i, :]
                action_step = action_step[0]
                _, obs, reward, done, info = env.step(action_step)

                # draw_state = env.render()
                
                state_history = torch.roll(state_history, shifts=-1, dims=1)
                action_history = torch.roll(action_history, shifts=-1, dims=1)
                
                # state_history[:, -1, :] = obs
                # action_history[:, -1, :] = action_step
                single_obs_dict = {"obs": state_history[:, -1, :].to("cuda:0")}
            
                idx += 1
            # reset env
                
            if done:
                env.reset()
            # env_ids = torch.nonzero(done, as_tuple=False).squeeze(1).int()
            # if len(env_ids) > 0:
            #     state_history[env_ids,:,:] = single_obs_dict["obs"][env_ids].to(state_history.device)[:,None,:]
            #     action_history[env_ids,:,:] = 0.0
                
            #     idx = 0
            #     env.reset()
                
            #     # flush saved data
            #     if save_zarr:
            #         for i in range(len(env_ids)):
            #             epi_len = np.all(recorded_obs_episode[env_ids[i]] == 0, axis=-1).argmax(axis=-1)
            #             if epi_len == 0:
            #                 epi_len = recorded_acs_episode.shape[1]
            #             recorded_obs.append(np.copy(recorded_obs_episode[env_ids[i], :epi_len]))
            #             recorded_acs.append(np.copy(recorded_acs_episode[env_ids[i], :epi_len]))
                        
            #             recorded_obs_episode[env_ids[i]] = 0
            #             recorded_acs_episode[env_ids[i]] = 0
                        
            #             saved_idx += epi_len
            #             episode_ends.append(saved_idx)
                        
            #             print("saved_idx: ", saved_idx)
                    
            # done = done.cpu().numpy()
            # done = np.all(done)
            past_action = action
            if idx % 1000 == 0:
                print("time to collect 1000: ", time.perf_counter() - t1)
                t1 = time.perf_counter()

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

