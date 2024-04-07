import zarr
import time
import numpy as np

skill_filenames = ['recorded_data_bounce_12-15-35.zarr', 'recorded_data_cyber2_stand_dance_aug_23-39-56.zarr', 'recorded_data_hop_12-18-21.zarr']
zroot = zarr.open_group("recorded_data_{}_{}.zarr".format('combined', time.strftime("%H-%M-%S", time.localtime())), "w")

zroot.create_group("data")
zdata = zroot["data"]

zroot.create_group("meta")
zmeta = zroot["meta"]

zmeta.create_group("episode_ends")

zdata.create_group("action")
zdata.create_group("state")

action = np.zeros((0,12))
state = np.zeros((0,45))
episode_ends = np.zeros((0,))


episode_end = 0
for skill in skill_filenames: 
    print("Processing {}".format(skill))
    zfile = zarr.open(skill, "r")
    action = np.concatenate([action, zfile["data"]["action"]])
    state = np.concatenate([state, zfile["data"]["state"]])
    episode_ends = np.concatenate([episode_ends, np.array(zfile["meta"]["episode_ends"]) + episode_end])
    episode_end = episode_end + zfile["meta"]["episode_ends"][-1]


zdata["action"] = action
zdata["state"] = state
zmeta["episode_ends"] = episode_ends