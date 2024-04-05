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
from cassie_run_env.cassie_env import CassieEnv
import time
import numpy as np
import argparse
import  cassie_run_env.ppo.policies as policies 
from cassie_run_env import CASSIE_GYM_ROOT_DIR
import multiprocessing
import random
import os
import pickle

f = open("walk_cmd.pkl", "rb")
cmd_obs = []
while True:
    try:
        cmd_obs.append(pickle.load(f))
    except:
        break
cmd_obs = np.stack(cmd_obs)
cmd_idx = 0

library_folder = CASSIE_GYM_ROOT_DIR + '/motions/MotionLibrary/'
model_folder = CASSIE_GYM_ROOT_DIR + '/tf_model/'
EP_LEN_MAX = 1000
model_name = 'running_400m_stable'


def generate_data_mujoco(recorded_obs, recorded_acs, len_to_save):

    process_pid = os.getpid()
    seed = (os.getpid() * int(time.perf_counter())) % 123456789
    t1 = time.perf_counter()

    random.seed(seed)
    np.random.seed(seed)
    
    perturb_flag = np.random.choice([True, False], p=[0.2, 0.8])
        
    env = CassieEnv(max_timesteps=EP_LEN_MAX, 
                    is_visual=False,
                    ref_file=library_folder+'RunMocapMotion.motionlib', 
                    overwrite_pos_by_cmd=True,
                    stage='single')

    env.num_envs = 1
    env.max_episode_length = EP_LEN_MAX

    obs_vf, obs = env.reset()

    model_dir = model_folder + model_name
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    model_path = latest_checkpoint
    config = tf.ConfigProto(device_count={'GPU': 0})

    ob_space_pol = env.observation_space_pol
    ac_space = env.action_space

    env.num_obs = 64 # obs base shape
    env.num_actions = env.action_space.shape[0]

    ob_space_vf = env.observation_space_vf
    ob_space_pol_cnn = env.observation_space_pol_cnn
    expert_policy = policies.MLPCNNPolicy(name='pi', ob_space_vf=ob_space_vf, ob_space_pol=ob_space_pol, ob_space_pol_cnn=ob_space_pol_cnn, 
                                ac_space=ac_space, hid_size=512, num_hid_layers=2)

    U.make_session(config=config)
    U.load_state(model_path)


    recorded_obs_episode = np.zeros((env.num_envs, env.max_episode_length+2, env.num_obs))
    recorded_acs_episode = np.zeros((env.num_envs, env.max_episode_length+3, env.num_actions))

    idx = 0
    total_idx = 0
            
    print(process_pid, " process seed: ", seed )

    while True:
        # run policy
        single_obs = obs[0][-62:]
        extra_obs = np.array(env.obs_cassie_state.pelvis.rotationalVelocity).flatten()

        single_obs = np.concatenate([single_obs[:,:6], extra_obs[None,:], single_obs[:,6:-1]], axis=-1)

        expert_action = expert_policy.act(stochastic=False, ob_vf=obs_vf, ob_pol=obs)[0]
        recorded_obs_episode[0,idx,:] = single_obs
        recorded_acs_episode[0,idx,:] = expert_action

        action_step = expert_action
        _, obs, reward, done, info = env.step(action_step)
    
        idx += 1
        # reset env
            
        if done:
            env.reset()
            epi_len = np.all(recorded_obs_episode[0] == 0, axis=-1).argmax(axis=-1)
            if epi_len == 0:
                epi_len = recorded_acs_episode.shape[1]
            recorded_obs.append(np.copy(recorded_obs_episode[0, :epi_len]))
            recorded_acs.append(np.copy(recorded_acs_episode[0, :epi_len]))

            total_idx += idx
            idx = 0
            if total_idx >= len_to_save:
                break
            else:
                print(process_pid, " progress: ", total_idx / len_to_save, "remaining time: ", (time.perf_counter() - t1) / (total_idx / len_to_save) - (time.perf_counter() - t1))



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
        self.generate_data = False
        if not self.generate_data:            
            env = CassieEnv(max_timesteps=EP_LEN_MAX, 
                    is_visual=True,
                    ref_file=library_folder+'RunMocapMotion.motionlib', 
                    overwrite_pos_by_cmd=True,
                    stage='single')

            env.num_envs = 1
            env.max_episode_length = EP_LEN_MAX

            model_dir = model_folder + model_name
            latest_checkpoint = tf.train.latest_checkpoint(model_dir)
            model_path = latest_checkpoint
            config = tf.ConfigProto(device_count={'GPU': 0})

            ob_space_pol = env.observation_space_pol
            ac_space = env.action_space

            env.num_obs = 64 # obs base shape
            env.num_actions = env.action_space.shape[0]

            ob_space_vf = env.observation_space_vf
            ob_space_pol_cnn = env.observation_space_pol_cnn
            self.pi = policies.MLPCNNPolicy(name='pi', ob_space_vf=ob_space_vf, ob_space_pol=ob_space_pol, ob_space_pol_cnn=ob_space_pol_cnn, 
                                ac_space=ac_space, hid_size=512, num_hid_layers=2)

            U.make_session(config=config)
            U.load_state(model_path)
        else:
            env = None
        
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

        # plan for rollout
        assert generate_data == self.generate_data, "generate data?"

        save_zarr = generate_data or (not online)
        len_to_save = 1200 if not generate_data else 1e6
        
        if not self.generate_data:
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
            single_obs = obs[0][None, -62:]
            extra_obs = np.array(env.obs_cassie_state.pelvis.rotationalVelocity).flatten()

            single_obs = np.concatenate([single_obs[:,:6], extra_obs[None,:], single_obs[:,6:-1]], axis=-1)
            state_history[:,:,:] = torch.from_numpy(single_obs).to(device)[:, None, :] # (env.num_envs, 1, env.num_observations)
            
            obs_dict = {"obs": state_history[:, :]} #, 'past_action': action_history}
            single_obs_dict = {"obs": state_history[:, -1, :].to("cuda:0")} #, 'past_action': action_history[0]}

        else:
            num_processes = 24  # Define the number of processes

            # Create a manager and a shared list
            manager = multiprocessing.Manager()
            recorded_obs = manager.list()
            recorded_acs = manager.list()

            # Create and start processes
            processes = [multiprocessing.Process(target=generate_data_mujoco, args=(recorded_obs, recorded_acs,len_to_save // num_processes)) for _ in range(num_processes)]
            for process in processes:
                process.start()

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
            zdata.create_group("cmd")
            zdata.create_group("reference")
                       
        if not self.generate_data:
            recorded_obs = []
            recorded_acs = []

            env.max_episode_length = int(env.max_episode_length)
            
            recorded_obs_episode = np.zeros((env.num_envs, env.max_episode_length+2, env.num_obs))
            recorded_acs_episode = np.zeros((env.num_envs, env.max_episode_length+3, env.num_actions))
 
            episode_ends = []
            action_error = []
            idx = 0    
            saved_idx = 0    
            skip_epi = 0
            skip = 0
            t1 = time.perf_counter()
            while True:
                # run policy
                with torch.no_grad():
                    expert_action = expert_policy.act(stochastic=False, ob_vf=obs_vf, ob_pol=obs)[0]
                    if online and idx > skip:    
                        obs_dict = {"obs": state_history[:, -history-1:-1, :]}
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
                        action = pred_action[:,history:history+1,:].cpu().numpy()
                    else:
                        action = expert_action[None, None, :]
                        time.sleep(0.01)
                if save_zarr:
                    curr_idx = np.all(recorded_obs_episode == 0, axis=-1).argmax(axis=-1)
                    # curr_idx = idx
                    recorded_obs_episode[np.arange(env.num_envs),curr_idx,:] = single_obs_dict["obs"].to("cpu").detach().numpy()
                    recorded_acs_episode[np.arange(env.num_envs),curr_idx,:] = expert_action
        
                # step env
                self.n_action_steps = action.shape[1]
                for i in range(self.n_action_steps):
                    action_step = action[:, i, :]
                    action_step = action_step[0]
                    _, obs, reward, done, info = env.step(action_step)

                    draw_state = env.render()
                    
                    state_history = torch.roll(state_history, shifts=-1, dims=1)
                    action_history = torch.roll(action_history, shifts=-1, dims=1)
                    
                    single_obs = obs[0][None, -62:]
                    extra_obs = np.array(env.obs_cassie_state.pelvis.rotationalVelocity).flatten()

                    single_obs = np.concatenate([single_obs[:,:6], extra_obs[None,:], single_obs[:,6:-1]], axis=-1)

                    # import  pickle
                    # if skip_epi > 2:
                    #     print("saving data")
                        # with open("run_cmd.pkl", "ab") as f:
                        #     pickle.dump(single_obs[:,-36:], f)
                    
                    global cmd_idx
                    if idx < 350:
                        single_obs[:,-36:] = cmd_obs[cmd_idx]
                        cmd_idx += 1
                        print('start running')
                    print(idx)

                    state_history[:,-1,:] = torch.from_numpy(single_obs).to(device)[:, None, :] # (env.num_envs, 1, env.num_observations)
                    action_history[:, -1, :] = torch.from_numpy(action_step).to(device)[None, :]
                    single_obs_dict = {"obs": state_history[:, -1, :].to("cuda:0")}
                
                    idx += 1
                # reset env
                    
                if done:

                    cmd_idx = 0
                    idx = 0

                    skip_epi += 1
                    env.reset()
            
                    single_obs = obs[0][None, -62:]
                    extra_obs = np.array(env.obs_cassie_state.pelvis.rotationalVelocity).flatten()

                    single_obs = np.concatenate([single_obs[:,:6], extra_obs[None,:], single_obs[:,6:-1]], axis=-1)

                    state_history[0,:,:] = torch.from_numpy(single_obs).to(device)[:, None, :] # (env.num_envs, 1, env.num_observations)
                    action_history[0,:,:] = 0.0
                
                
                    # flush saved data
                    epi_len = np.all(recorded_obs_episode[0] == 0, axis=-1).argmax(axis=-1)
                    if epi_len == 0:
                        epi_len = recorded_acs_episode.shape[1]
                    recorded_obs.append(np.copy(recorded_obs_episode[0, :epi_len]))
                    recorded_acs.append(np.copy(recorded_acs_episode[0, :epi_len]))
                    
                    recorded_obs_episode[0] = 0
                    recorded_acs_episode[0] = 0
                    
                    saved_idx += epi_len
                    episode_ends.append(saved_idx)
                    
                    print("saved_idx: ", saved_idx)
                    idx = 0

            # update pbar
                if online:
                    pbar.update(action.shape[1])
                else:
                    pbar.update(env.num_envs)
                
                if save_zarr and saved_idx >= len_to_save:
                    recorded_obs = np.concatenate(recorded_obs)
                    recorded_acs = np.concatenate(recorded_acs)
                    episode_ends = np.array(episode_ends)
                   
                    zdata["state"] = recorded_obs[:,:29]
                    zdata["cmd"] = recorded_obs[:,-5:]
                    zdata["reference"] = recorded_obs[:,29:-5]
                    zdata["action"] = recorded_acs
                    zmeta["episode_ends"] = episode_ends
                    print(zroot.tree())
                    break
                                
            # clear out video buffer
            _ = env.reset()
        else:

            for process in processes:
                process.join()

            episode_ends = [l.shape[0] for l in recorded_obs]
            episode_ends = [sum(episode_ends[:i+1]) for i in range(len(episode_ends))]
            # Concatenate all arrays in the shared list
            recorded_obs = np.concatenate(recorded_obs, axis=0)
            recorded_acs = np.concatenate(recorded_acs, axis=0)
            episode_ends = np.array(episode_ends)
            
            zdata["state"] = recorded_obs[:,:29]
            zdata["cmd"] = recorded_obs[:,-5:]
            zdata["reference"] = recorded_obs[:,29:-5]            
            zdata["action"] = recorded_acs
            zmeta["episode_ends"] = episode_ends
            print(zroot.tree())

        return file_name

