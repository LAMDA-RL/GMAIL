import sys
import os
import gym
import ray
import torch
import numpy as np
import random
import math
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code.algos.ddpg import DDPG
from code.algos.wgail import Posterior


class EnvWorker:
    def __init__(self, parameter, env_name='Hopper-v2', seed=0, policy_type=DDPG,
                 env_decoration=None, env_tasks=None, non_stationary=False, fix_env=None, pos_type=Posterior):
        # init env
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.fix_env_setting = False
        self.set_global_seed(seed)
        self.fix_env = fix_env
        self.non_stationary = non_stationary
        if env_decoration is not None:
            self.env = env_decoration(self.env, log_scale_limit=parameter.env_default_change_range,
                                    rand_params=parameter.varying_params)
        self.non_stationary_period = parameter.non_stationary_period
        self.non_stationary_interval = parameter.non_stationary_interval
        self.observation_space = self.env.observation_space
        self.env_tasks = None
        self.task_ind = -1
        self.env.reset(seed=self.fix_env)
        if env_tasks is not None and isinstance(env_tasks, list) and len(env_tasks) > 0:
            self.env_tasks = env_tasks
            self.task_ind = random.randint(0, len(self.env_tasks) - 1)
            self.env.set_task(self.env_tasks[self.task_ind])
        
        # init policy
        policy_config = policy_type.make_config_from_param(parameter)
        self.policy = policy_type(state_shape=self.observation_space.shape,
                                  action_dim=self.action_space.high.size,
                                  **policy_config)
        if pos_type is not None:
            pos_config = pos_type.make_config_from_param(parameter)
            self.pos = pos_type(state_shape=self.observation_space.shape,
                                action_dim=self.action_space.high.size,
                                encode_dim=parameter.task_num, 
                                **pos_config)
        
        # init results
        self.ep_len = 0
        self.ep_cumrew = 0
        self.ep_len_list = []
        self.ep_cumrew_list = []
        self.ep_rew_list = []
        self.state = self.reset(None)

    def set_weight(self, state_dict):
        self.policy.load_state_dict(state_dict)
        
    def set_pos_weight(self, state_dict):
        self.pos.load_state_dict(state_dict)
        
    def get_task_ind(self):
        return self.task_ind

    def set_global_seed(self, seed):
        import numpy as np
        import torch
        import random
        self.env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.action_space.seed(seed)

    def get_weight(self):
        return self.policy.state_dict()

    def set_fix_env_setting(self, fix_env_setting=True):
        self.fix_env_setting = fix_env_setting

    def change_env_param(self, set_env_ind=None):
        if self.fix_env_setting:
            self.env.set_task(self.env_tasks[self.task_ind])
            return
        if self.env_tasks is not None and len(self.env_tasks) > 0:
            self.task_ind = random.randint(0, len(self.env_tasks) - 1) if set_env_ind is None or set_env_ind >= \
                                                                          len(self.env_tasks) else set_env_ind
            self.env.set_task(self.env_tasks[self.task_ind])
            if self.non_stationary:
                env_param_list = [self.env_tasks[self.task_ind]] + [self.env_tasks[random.randint(0, len(self.env_tasks)-1)] for _ in range(15)]
                self.env.set_nonstationary_para(env_param_list, self.non_stationary_period, self.non_stationary_interval)

    def sample(self, min_batch, deterministic=False, env_ind=None, use_encode=False):
        step_count = 0
        log = {'EpRet': [],
               'EpMeanRew': [],
               'EpLen': []}
        with torch.no_grad():
            while step_count < min_batch:
                state = self.reset(env_ind)
                if use_encode:
                    k_steps = 16
                    traj_states = torch.zeros((k_steps, ) + self.env.observation_space.shape)
                    traj_next_states = torch.zeros((k_steps, ) + self.env.observation_space.shape)
                    traj_actions = torch.zeros((k_steps, ) + self.env.action_space.shape)
                while True:
                    if not use_encode:
                        state_tensor = torch.from_numpy(state).to(torch.get_default_dtype()).unsqueeze(0)
                        action_tensor = self.policy.get_action(state_tensor, deterministic, tensor=True)[0]
                    else:
                        states_tensor = torch.from_numpy(state).to(torch.get_default_dtype()).unsqueeze(0)
                        encode, _ = self.pos.get_encode(traj_states, traj_actions, traj_next_states)
                        actions_tensor = self.policy.get_action(states_tensor, encode, True, True)
                        traj_states = torch.concat((traj_states[1:], states_tensor))
                        traj_actions = torch.concat((traj_actions[1:], actions_tensor))
                        action_tensor = actions_tensor.squeeze()
                    action = action_tensor.numpy()
                    next_state, reward, done, _ = self.env.step(self.env.denormalization(action))
                    if use_encode:
                        traj_next_states = torch.concat((traj_next_states[1:], states_tensor))
                    if self.non_stationary:
                        self.env_param_vector = self.env.env_parameter_vector
                    self.ep_cumrew += reward
                    self.ep_len += 1
                    step_count += 1
                    if done:
                        log['EpMeanRew'].append(self.ep_cumrew / self.ep_len)
                        log['EpLen'].append(self.ep_len)
                        log['EpRet'].append(self.ep_cumrew)
                        break
                    state = next_state
        return log
    
    def sample_encode(self, min_batch, env_ind_list):
        step_count, inner_step_count, k_steps = 0, 0, 16
        params, encodes = [], []
        current_env_ind_count = 0
        with torch.no_grad():
            state = self.reset()
            while step_count < min_batch:
                self.change_env_param(env_ind_list[current_env_ind_count])
                traj_states = torch.zeros((k_steps, ) + self.env.observation_space.shape)
                traj_next_states = torch.zeros((k_steps, ) + self.env.observation_space.shape)
                traj_actions = torch.zeros((k_steps, ) + self.env.action_space.shape)
                while True:
                    states_tensor = torch.from_numpy(state).to(torch.get_default_dtype()).unsqueeze(0)
                    encode, _ = self.pos.get_encode(traj_states, traj_actions, traj_next_states)
                    params.append(env_ind_list[current_env_ind_count])
                    encodes.append(int(encode))
                    actions_tensor = self.policy.get_action(states_tensor, encode, True, True)
                    traj_states = torch.concat((traj_states[1:], states_tensor))
                    traj_actions = torch.concat((traj_actions[1:], actions_tensor))
                    action_tensor = actions_tensor.squeeze()
                    action = action_tensor.numpy()
                    next_state, reward, done, _ = self.env.step(self.env.denormalization(action))
                    traj_next_states = torch.concat((traj_next_states[1:], states_tensor))
                    step_count += 1
                    inner_step_count += 1
                    if done or inner_step_count == 100:
                        inner_step_count = 0
                        current_env_ind_count += 1
                        if done:
                            state = self.reset()
                        break
                    state = next_state
                if current_env_ind_count == len(env_ind_list):
                    break
        return params, encodes

    def get_current_state(self):
        return self.state

    def reset(self, env_ind=None):
        state = self.env.reset(seed=self.fix_env)
        self.change_env_param(env_ind)
        self.env_param_vector = self.env.env_parameter_vector
        self.ep_len = 0
        self.ep_cumrew = 0
        return state

    def step(self, action, env_ind=None, render=False, need_info=False):
        next_state, reward, done, info = self.env.step(self.env.denormalization(action))
        if render:
            self.env.render()
        if self.non_stationary:
            self.env_param_vector = self.env.env_parameter_vector
        current_env_step = self.env._elapsed_steps
        self.state = next_state
        self.ep_len += 1
        self.ep_cumrew += reward
        cur_task_ind = self.task_ind
        cur_env_param = self.env_param_vector
        if done:
            self.ep_len_list.append(self.ep_len)
            self.ep_cumrew_list.append(self.ep_cumrew)
            self.ep_rew_list.append(self.ep_cumrew / self.ep_len)
            self.state = self.reset(env_ind)
        if need_info:
            return next_state, reward, done, self.state, cur_task_ind, cur_env_param, current_env_step, info
        return next_state, reward, done, self.state, cur_task_ind, cur_env_param, current_env_step

    def collect_result(self):
        ep_len_list = self.ep_len_list
        self.ep_len_list = []
        ep_cumrew_list = self.ep_cumrew_list
        self.ep_cumrew_list = []
        ep_rew_list = self.ep_rew_list
        self.ep_rew_list = []
        log = {
        'EpMeanRew': ep_rew_list,
        'EpLen': ep_len_list,
        'EpRet': ep_cumrew_list
        }
        return log


class EnvRemoteArray:
    def __init__(self, parameter, env_name, worker_num=2, seed=None,
                 deterministic=False, use_remote=True, policy_type=DDPG, env_decoration=None,
                 env_tasks=None, non_stationary=False, fix_env=None, pos_type=Posterior):
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.action_space = self.env.action_space
        self.set_seed(seed)
        self.non_stationary = non_stationary
        self.env_tasks = env_tasks
        RemoteEnvWorker = ray.remote(EnvWorker) if use_remote else EnvWorker
        if use_remote:
            self.workers = [RemoteEnvWorker.remote(parameter, env_name, random.randint(0, 10000),
                                                   policy_type, env_decoration, env_tasks,
                                                   non_stationary, fix_env, pos_type) for _ in range(worker_num)]
        else:
            self.workers = [RemoteEnvWorker(parameter, env_name, random.randint(0, 10000),
                                            policy_type, env_decoration, env_tasks,
                                            non_stationary, fix_env, pos_type) for _ in range(worker_num)]

        if env_decoration is not None:
            self.env = env_decoration(self.env, log_scale_limit=parameter.env_default_change_range,
                                    rand_params=parameter.varying_params)
        net_config = policy_type.make_config_from_param(parameter)
        self.policy = policy_type(self.env.observation_space.shape, self.env.action_space.shape[0], **net_config)
        if pos_type is not None:
            self.pos = pos_type(self.env.observation_space.shape, self.env.action_space.shape[0], parameter.task_num)
        self.worker_num = worker_num
        self.env_name = env_name

        self.env.reset()
        if isinstance(env_tasks, list) and len(env_tasks) > 0:
            self.env.set_task(random.choice(env_tasks))
        self.env_parameter_len = self.env.env_parameter_length
        self.running_state = None
        self.deterministic = deterministic
        self.use_remote = use_remote
        self.total_steps = 0

    def set_seed(self, seed):
        if seed is None:
            return
        import numpy as np
        import torch
        import random
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.action_space.seed(seed)
        self.env.seed(seed)

    def set_fix_env_setting(self, fix_env_setting=True):
        if self.use_remote:
            ray.get([worker.set_fix_env_setting.remote(fix_env_setting) for worker in self.workers])
        else:
            for worker in self.workers:
                worker.set_fix_env_setting(fix_env_setting)

    def submit_task(self, min_batch, policy=None, env_ind=None, use_encode=False, pos=None):
        assert not (policy is None and self.policy is None)
        cur_policy = policy if policy is not None else self.policy
        if self.use_remote:
            ray.get([worker.set_weight.remote(cur_policy.state_dict()) for worker in self.workers])
            min_batch_per_worker = min_batch // self.worker_num + 1
            if not use_encode:
                futures = [worker.sample.remote(min_batch_per_worker, self.deterministic, env_ind)
                        for worker in self.workers]
            else:
                cur_pos = pos if pos is not None else self.pos
                ray.get([worker.set_pos_weight.remote(cur_pos.state_dict()) for worker in self.workers])
                futures = [worker.sample.remote(min_batch_per_worker, self.deterministic, env_ind, use_encode)
                           for worker in self.workers]
        else:
            [worker.set_weight(cur_policy.state_dict()) for worker in self.workers]
            min_batch_per_worker = min_batch // self.worker_num + 1
            futures = [worker.sample(min_batch_per_worker, self.deterministic, env_ind)
                    for worker in self.workers]
        return futures
    
    def test_encode(self, min_batch, policy=None, env_ind_list=None, pos=None):
        assert not (policy is None and self.policy is None)
        cur_policy = policy if policy is not None else self.policy
        [worker.set_weight(cur_policy.state_dict()) for worker in self.workers]
        cur_pos = pos if pos is not None else self.pos
        [worker.set_pos_weight(cur_pos.state_dict()) for worker in self.workers]
        min_batch_per_worker = min_batch // self.worker_num + 1

        futures = [worker.sample_encode(min_batch_per_worker, env_ind_list)
                   for worker in self.workers]
        return futures

    def query_sample(self, futures):
        if self.use_remote:
            mem_list_pre = ray.get(futures)
        else:
            mem_list_pre = futures
        logs = {key: [] for key in mem_list_pre[0]}
        for key in logs:
            for item in mem_list_pre:
                logs[key] += item[key]
        return logs

    # always use remote
    def sample(self, min_batch, policy=None, env_ind=None):
        assert not (policy is None and self.policy is None)
        cur_policy = policy if policy is not None else self.policy
        ray.get([worker.set_weight.remote(cur_policy.state_dict()) for worker in self.workers])

        min_batch_per_worker = min_batch // self.worker_num + 1
        futures = [worker.sample.remote(min_batch_per_worker, self.deterministic, env_ind)
                                for worker in self.workers]
        mem_list_pre = ray.get(futures)
        logs = {key: [] for key in mem_list_pre[0][1]}
        for key in logs:
            for _, item in mem_list_pre:
                logs[key] += item[key]
        return logs

    def sample_locally(self, min_batch, policy=None, env_ind=None):
        assert not (policy is None and self.policy is None)
        cur_policy = policy if policy is not None else self.policy
        for worker in self.workers:
            worker.set_weight(cur_policy.state_dict())
        min_batch_per_worker = min_batch // self.worker_num + 1
        mem_list_pre = [worker.sample(min_batch_per_worker, self.deterministic, env_ind)
                                for worker in self.workers]
        logs = {key: [] for key in mem_list_pre[0][1]}
        for key in logs:
            for _, item in mem_list_pre:
                logs[key] += item[key]
        return logs

    def sample1step(self, policy=None, random=False, device=torch.device('cpu'), 
                    env_ind=None, use_encode=False):
        assert not (policy is None and self.policy is None)
        cur_policy = policy if policy is not None else self.policy
        if (not self.use_remote) and self.worker_num == 1:
            return self.sample1step1env(policy, random, device, env_ind, use_encode=use_encode)
        if self.use_remote:
            states = ray.get([worker.get_current_state.remote() for worker in self.workers])
        else:
            states = [worker.get_current_state() for worker in self.workers]

        states = np.array(states)
        with torch.no_grad():
            if random:
                actions = [self.env.normalization(self.action_space.sample()) for item in states]
            else:
                states_tensor = torch.from_numpy(states).to(torch.get_default_dtype()).to(device)
                actions = cur_policy.get_action(states_tensor, self.deterministic, True).to(
                    torch.device('cpu')).squeeze(1).numpy()
        if self.use_remote:
            srd = ray.get([worker.step.remote(action, env_ind, need_info=True) for action, worker in zip(actions, self.workers)])
        else:
            srd = [worker.step(action) for action, worker in zip(actions, self.workers)]

        if self.use_remote:
            logs_ = ray.get([worker.collect_result.remote() for worker in self.workers])
        else:
            logs_ = [worker.collect_result() for worker in self.workers]
        logs = {key: [] for key in logs_[0]}
        for key in logs:
            for item in logs_:
                logs[key] += item[key]
        return logs

    def get_action(self, state, cur_policy, random, device=torch.device("cpu"), encode=None):
        with torch.no_grad():
            if random:
                action = self.env.normalization(self.action_space.sample())
            else:
                if encode is not None:
                    encode = torch.tensor(encode).unsqueeze(0)
                    action = cur_policy.get_action(torch.from_numpy(state[None]).to(device=device,dtype=torch.get_default_dtype()),
                                                   encode, self.deterministic, True)[0].to(torch.device('cpu')).numpy()
                else:  
                    action = cur_policy.get_action(torch.from_numpy(state[None]).to(device=device,dtype=torch.get_default_dtype()),
                                                   self.deterministic, True)[0].to(torch.device('cpu')).numpy()
        return action

    def sample1step1env(self, policy, random=False, device=torch.device('cpu'), env_ind=None, 
                        render=False, need_info=False, use_encode=False):
        cur_policy = policy
        worker = self.workers[0]
        state = worker.get_current_state()
        if use_encode:
            encode = worker.get_task_ind()
            action = self.get_action(state, cur_policy, random, device, encode=encode)
        else:
            action = self.get_action(state, cur_policy, random, device)
        if need_info:
            next_state, reward, done, _, task_ind, env_param, current_steps, info = worker.step(action, env_ind, render, need_info=True)
        else:
            next_state, reward, done, _, task_ind, env_param, current_steps = worker.step(action, env_ind, render, need_info=False)
        mem = {"obs": state, "act": action, "next_obs": next_state, "rew": reward, "done": done}
        if use_encode:
            mem["encode"] = encode
        logs = worker.collect_result()
        self.total_steps += 1
        logs['TotalSteps'] = self.total_steps
        if need_info:
            return mem, logs, info
        return mem, logs

    def collect_samples(self, min_batch, policy=None):
        for i in range(10):
            try:
                logs = self.sample(min_batch, policy)
                break
            except Exception as e:
                print(f'Error occurs while sampling, the error is {e}, tried time: {i}')
        return logs

    def update_running_state(self, state):
        pass

    def make_env_param_dict(self, parameter_name):
        res = {}
        if self.env_tasks is not None:
            for ind, item in enumerate(self.env_tasks):
                res[ind + 1] = item
        res_interprete = {}
        for k, v in res.items():
            if isinstance(v, dict):
                res_interprete[k] = [v[parameter_name][-1]]
            elif isinstance(v, int):
                res_interprete[k] = v
            elif isinstance(v, list):
                res_interprete[k] = math.sqrt(sum([item**2 for item in v]))
            else:
                raise NotImplementedError(f'type({type(v)}) is not implemented.')
        return res_interprete

    def make_env_param_dict_from_params(self, params):
        res_interprete = {}
        for param in params:
            res_ = self.make_env_param_dict(param)
            for k, v in res_.items():
                if k not in res_interprete:
                    res_interprete[k] = v
                else:
                    res_interprete[k] += v

        return res_interprete
    
    