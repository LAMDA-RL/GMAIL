import os
import random
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
from matplotlib import animation
from gym import make


def make_gridworld_env(env_name, tasks_num, seed=1, env_changing_rate=None):
    env = make(env_name, env_changing_rate=env_changing_rate)
    env.seed(seed)
    tasks = env.sample_tasks(tasks_num)
    env.set_tasks(tasks)
    return env


def seed_torch(seed=1):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def save_path(samples, filename):
    joblib.dump(samples, filename, compress=3)


def restore_traj(dirname, n_path=10, max_steps=None):
    assert os.path.isdir(dirname)
    filenames = get_filenames_origin(dirname, n_path)
    return load_traj(filenames, max_steps)


def get_filenames_origin(dirname, n_path):
    filenames = []
    for _, filename in enumerate(os.listdir(dirname)):
        filenames.append(filename)
    n_path = n_path if n_path is not None else len(filenames)
    filenames = filenames[:n_path]
    
    filepath = []
    for filename in filenames:
        filepath.append(os.path.join(dirname, filename))
    return filepath


def restore_d4rl_traj(filename, max_steps=None, use_absorbing_state=False, max_episode_steps=1000):
    traj = joblib.load(filename)
    traj["obses"] = traj["observations"][:max_steps]
    traj["acts"] = traj["actions"][:max_steps]
    traj["next_obses"] = traj["next_observations"][:max_steps]
    traj["done"] = traj["terminals"][:max_steps]
    if not use_absorbing_state:
        return traj
    
    path_lst = []
    path = {"obs": [], "act": []}
    for i in range(len(traj["obses"])):
        path["obs"].append(traj["obses"][i])
        path["act"].append(traj["acts"][i])
        if traj["done"][i] == 1:
            path["obs"], path["act"] = np.array(path["obs"]), np.array(path["act"])
            path_lst.append(path)
            path = {"obs": [], "act": []}
    
    def get_obs_and_act(path):
        path['done'] = np.hstack([np.zeros((path['obs'].shape[0] - 1)), np.array([1])])
        path['mask'] = 1. - path['done']

        # add an extra indicator dimension that indicates whether the state is absorbing or not, 
        # for absorbing states set the indicator dimension to one and all other dimensions to zero
        path['obs'] = np.hstack([path['obs'], np.zeros([path['obs'].shape[0], 1])])
        ABSORBING_STATE = np.zeros(path['obs'].shape[1], )
        ABSORBING_STATE[-1] = 1.
        ABSORBING_ACTION = np.zeros(path['act'].shape[1])
        # the agent enters absorbing states after the end of episode, 
        # absorbing state transitions to itself for all agent actions
        if path['obs'].shape[0] < max_episode_steps:
            path['obs'] = np.vstack([path['obs'], ABSORBING_STATE, ABSORBING_STATE])
            path['act'] = np.vstack([path['act'], ABSORBING_ACTION, ABSORBING_ACTION])
            path['done'] = np.hstack([path['done'], np.array([0, 0])])
            path['mask'] = np.hstack([path['mask'], np.array([-1, -1])])
        
        obses, actions = path['obs'][:-1], path['act'][:-1]
        dones, masks = path['done'][:-1], path['mask'][:-1]
        next_obses = path['obs'][1:]
        return obses, next_obses, actions, dones, masks

    for i, path in enumerate(path_lst):
        if i == 0:
            obses, next_obses, acts, dones, masks = get_obs_and_act(path)
        else:
            obs, next_obs, act, done, mask = get_obs_and_act(path)
            obses = np.vstack((obs, obses))
            next_obses = np.vstack((next_obs, next_obses))
            acts = np.vstack((act, acts))
            dones = np.hstack((done, dones))
            masks = np.hstack((mask, masks))

    return {'obses': obses, 'next_obses': next_obses, 'acts': acts, 'dones': dones, 'masks': masks}


def restore_latest_n_traj(dirname, n_path=10, max_steps=None, H=1, 
                          use_absorbing_state=False, max_episode_steps=1000):
    assert os.path.isdir(dirname)
    filenames = get_filenames_origin(dirname, n_path)
    return load_trajectories(filenames, max_steps, H, use_absorbing_state, max_episode_steps)


def get_filenames(dirname, n_path=None):
    import re
    itr_reg = re.compile(
        r"step_(?P<step>[0-9]+)_epi_(?P<episodes>[0-9]+)_return_(-?)(?P<return_u>[0-9]+).(?P<return_l>[0-9]+).pkl")

    itr_files = []
    for _, filename in enumerate(os.listdir(dirname)):
        m = itr_reg.match(filename)
        if m:
            itr_count = m.group('step')
            itr_files.append((itr_count, filename))

    n_path = n_path if n_path is not None else len(itr_files)
    itr_files = sorted(itr_files, key=lambda x: int(
        x[0]), reverse=True)[:n_path]
    filenames = []
    for itr_file_and_count in itr_files:
        filenames.append(os.path.join(dirname, itr_file_and_count[1]))
    return filenames


def load_traj(filenames, max_steps=None):
    assert len(filenames) > 0
    paths = []
    for filename in filenames:
        paths.append(joblib.load(filename))
    
    obses, acts = [], []
    for _, path in enumerate(paths):
        obses.append(path['obs'][:-1])
        acts.append(path['act'][:-1])
    obses = np.stack(obses)
    acts = np.stack(acts)
    return {'obs': obses, 'act': acts}
    

def load_trajectories(filenames, max_steps=None, H=1, use_absorbing_state=False, max_episode_steps=1000):
    assert len(filenames) > 0
    paths = []
    for filename in filenames:
        paths.append(joblib.load(filename))

    def get_obs_and_act(path):
        path['done'] = np.hstack([np.zeros((path['obs'].shape[0] - 1)), np.array([1])])
        path['mask'] = 1. - path['done']

        if use_absorbing_state:
            # add an extra indicator dimension that indicates whether the state is absorbing or not, 
            # for absorbing states set the indicator dimension to one and all other dimensions to zero
            path['obs'] = np.hstack([path['obs'], np.zeros([path['obs'].shape[0], 1])])
            ABSORBING_STATE = np.zeros(path['obs'].shape[1], )
            ABSORBING_STATE[-1] = 1.
            ABSORBING_ACTION = np.zeros(path['act'].shape[1])
            # the agent enters absorbing states after the end of episode, 
            # absorbing state transitions to itself for all agent actions
            if path['obs'].shape[0] < max_episode_steps:
                path['obs'] = np.vstack([path['obs'], ABSORBING_STATE, ABSORBING_STATE])
                path['act'] = np.vstack([path['act'], ABSORBING_ACTION, ABSORBING_ACTION])
                path['done'] = np.hstack([path['done'], np.array([0, 0])])
                path['mask'] = np.hstack([path['mask'], np.array([-1, -1])])
        
        obses, actions = path['obs'][:-1], path['act'][:-1]
        dones, masks = path['done'][:-1], path['mask'][:-1]
        if H == 1:
            next_obses = path['obs'][1:]
        else:
            next_obses = []
            for i in range(1, path['obs'].shape[0]):
                if path['obs'].shape[0] - i >= H:
                    next_obs = path['obs'][i: i + H]
                else:
                    next_obs = path['obs'][i:]
                    tab_num = H - (path['obs'].shape[0] - i)
                    tabs = np.repeat(path['obs'][-1].reshape(1, -1), tab_num).reshape(tab_num, -1)
                    next_obs = np.vstack((next_obs, tabs))
                next_obses.append(next_obs)
            next_obses = np.array(next_obses)
        if max_steps is not None:
            return obses[:max_steps], next_obses[:max_steps], actions[:max_steps], dones[:max_steps], masks[:max_steps]
        else:
            return obses, next_obses, actions, dones, masks

    for i, path in enumerate(paths):
        if i == 0:
            obses, next_obses, acts, dones, masks = get_obs_and_act(path)
        else:
            obs, next_obs, act, done, mask = get_obs_and_act(path)
            obses = np.vstack((obs, obses))
            next_obses = np.vstack((next_obs, next_obses))
            acts = np.vstack((act, acts))
            dones = np.hstack((done, dones))
            masks = np.hstack((mask, masks))
    return {'obses': obses, 'next_obses': next_obses, 'acts': acts, 'dones': dones, 'masks': masks}


def frames_to_gif(frames, prefix, save_dir, interval=50, fps=30):
    """
    Convert frames to gif file
    """
    assert len(frames) > 0
    plt.figure(figsize=(frames[0].shape[1] / 72.,
                        frames[0].shape[0] / 72.), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    # TODO: interval should be 1000 / fps ?
    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=interval)
    output_path = "{}/{}.gif".format(save_dir, prefix)
    anim.save(output_path, writer='imagemagick', fps=fps)
