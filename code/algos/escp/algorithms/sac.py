import os, sys
import matplotlib.pyplot as plt
plt.switch_backend('agg')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from log_util.logger import Logger
from models.policy import Policy
from models.value import Value
from parameter.private_config import *
from code.algos.escp.agent.Agent import EnvRemoteArray
import gym
import torch
import numpy as np
import random
from utils.torch_utils import to_device
import time
from utils.timer import Timer
from algorithms.RMDM import RMDMLoss
from utils.visualize_repre import visualize_repre, visualize_repre_real_param
from algorithms.contrastive import ContrastiveLoss
from code.envs.nonstationary_env import NonstationaryEnv
from code.envs.get_env_tasks import get_env_tasks

class SAC:
    def __init__(self, parser, fix_env=None, env_tasks=None):
        # init logger, timer, parameter, replay buffer and device
        self.logger = Logger(parser)
        self.timer = Timer()
        self.parameter = self.logger.parameter
        self.replay_buffer_size = 0
        self.total_iteration = 0
        self.device = torch.device('cuda', index=self.parameter.gpu) if torch.cuda.is_available() else torch.device('cpu')
        self.logger.log(f"torch device is {self.device}")
        self.use_absorbing_state = self.parameter.use_absorbing_state
        
        # init env tasks: train tasks, test tasks, ood tasks
        self.env = NonstationaryEnv(gym.make(self.parameter.env_name), log_scale_limit=self.parameter.env_default_change_range,
                                    rand_params=self.parameter.varying_params)
        self.ood_env = NonstationaryEnv(gym.make(self.parameter.env_name), log_scale_limit=self.parameter.env_ood_change_range,
                                        rand_params=self.parameter.varying_params)
        self.global_seed(np.random, random, self.env, self.ood_env, seed=self.parameter.seed)
        torch.manual_seed(seed=self.parameter.seed)
        if env_tasks is None:
            env_tasks = get_env_tasks(env_name=self.parameter.env_name, varying_param=self.parameter.varying_params[0])
        self.env_tasks = self.env.sample_tasks(self.parameter.task_num, expert_tasks=env_tasks, train_mode=True)
        self.test_tasks = self.env.sample_tasks(self.parameter.test_task_num, expert_tasks=env_tasks)
        self.ood_tasks = self.ood_env.sample_tasks(self.parameter.test_task_num, expert_tasks=env_tasks)
        
        # init agent: train agent, test agent, ood agent, ns agent, ood ns agent
        self.training_agent = EnvRemoteArray(parameter=self.parameter, env_name=self.parameter.env_name,
                                             worker_num=1, seed=self.parameter.seed,
                                             deterministic=False, use_remote=False, policy_type=Policy,
                                             history_len=self.parameter.history_length, env_decoration=NonstationaryEnv,
                                             env_tasks=self.env_tasks,
                                             use_true_parameter=self.parameter.use_true_parameter,
                                             non_stationary=False, fix_env=fix_env)
        self.test_agent = EnvRemoteArray(parameter=self.parameter, env_name=self.parameter.env_name,
                                         worker_num=self.parameter.num_threads, seed=self.parameter.seed + 1,
                                         deterministic=True, use_remote=True, policy_type=Policy,
                                         history_len=self.parameter.history_length, env_decoration=NonstationaryEnv,
                                         env_tasks=self.test_tasks,
                                         use_true_parameter=self.parameter.use_true_parameter,
                                         non_stationary=False, fix_env=fix_env)
        self.ood_agent = EnvRemoteArray(parameter=self.parameter, env_name=self.parameter.env_name,
                                        worker_num=self.parameter.num_threads, seed=self.parameter.seed + 2,
                                        deterministic=True, use_remote=True, policy_type=Policy,
                                        history_len=self.parameter.history_length, env_decoration=NonstationaryEnv,
                                        env_tasks=self.ood_tasks,
                                        use_true_parameter=self.parameter.use_true_parameter,
                                        non_stationary=False, fix_env=fix_env) 
        self.non_station_agent = EnvRemoteArray(parameter=self.parameter, env_name=self.parameter.env_name,
                                                worker_num=self.parameter.num_threads, seed=self.parameter.seed + 3,
                                                deterministic=True, use_remote=True, policy_type=Policy,
                                                history_len=self.parameter.history_length,
                                                env_decoration=NonstationaryEnv, env_tasks=self.test_tasks,
                                                use_true_parameter=self.parameter.use_true_parameter,
                                                non_stationary=True, fix_env=fix_env)
        self.ood_ns_agent = EnvRemoteArray(parameter=self.parameter, env_name=self.parameter.env_name,
                                        worker_num=self.parameter.num_threads, seed=self.parameter.seed + 3,
                                        deterministic=True, use_remote=True, policy_type=Policy,
                                        history_len=self.parameter.history_length, env_decoration=NonstationaryEnv,
                                        env_tasks=self.ood_tasks,
                                        use_true_parameter=self.parameter.use_true_parameter,
                                        non_stationary=True, fix_env=fix_env) 
        self.station_agent_single_thread = EnvRemoteArray(parameter=self.parameter, env_name=self.parameter.env_name,
                                           worker_num=1, seed=self.parameter.seed + 4,
                                           deterministic=True, use_remote=False, policy_type=Policy,
                                           history_len=self.parameter.history_length,
                                           env_decoration=NonstationaryEnv, env_tasks=self.test_tasks,
                                           use_true_parameter=self.parameter.use_true_parameter,
                                           non_stationary=False, fix_env=fix_env)
        self.non_station_agent_single_thread = EnvRemoteArray(parameter=self.parameter, env_name=self.parameter.env_name,
                                                worker_num=1, seed=self.parameter.seed + 4,
                                                deterministic=True, use_remote=False, policy_type=Policy,
                                                history_len=self.parameter.history_length,
                                                env_decoration=NonstationaryEnv, env_tasks=self.test_tasks,
                                                use_true_parameter=self.parameter.use_true_parameter,
                                                non_stationary=True, fix_env=fix_env)
        
        # init config: policy config and value config
        self.policy_config = Policy.make_config_from_param(self.parameter)
        self.value_config = Value.make_config_from_param(self.parameter)
        if self.parameter.use_true_parameter:
            self.policy_config['ep_dim'] = self.training_agent.env_parameter_len
            self.value_config['ep_dim'] = self.training_agent.env_parameter_len
        self.policy_config['logger'] = self.logger
        self.value_config['logger'] = self.logger
        self.loaded_pretrain = not self.parameter.ep_pretrain_path_suffix == 'None'
        self.freeze_ep = False
        self.policy_config['freeze_ep'] = self.freeze_ep
        self.value_config['freeze_ep'] = self.freeze_ep
        
        # init policy: policy, policy for test, policy target
        self.obs_dim = self.training_agent.obs_dim
        self.act_dim = self.training_agent.act_dim
        self.policy = Policy(self.training_agent.obs_dim, self.training_agent.act_dim, **self.policy_config)
        self.policy_for_test = Policy(self.training_agent.obs_dim, self.training_agent.act_dim, **self.policy_config)
        self.policy_target = Policy(self.training_agent.obs_dim, self.training_agent.act_dim, **self.policy_config)

        # init value: value1, value2, target value1, target value2
        self.value1 = Value(self.training_agent.obs_dim, self.training_agent.act_dim, **self.value_config)
        self.value2 = Value(self.training_agent.obs_dim, self.training_agent.act_dim, **self.value_config)
        self.target_value1 = Value(self.training_agent.obs_dim, self.training_agent.act_dim, **self.value_config)
        self.target_value2 = Value(self.training_agent.obs_dim, self.training_agent.act_dim, **self.value_config)

        # init pretrain model
        if not self.parameter.ep_pretrain_path_suffix == 'None':
            pretrain_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(self.logger.model_output_dir)),
                                      '{}-{}'.format(self.parameter.env_name, self.parameter.ep_pretrain_path_suffix),
                                              'model'), 'environment_probe.pt')
            self.logger('load model from {}'.format(pretrain_path))
            _, _, _, _, _ = map(lambda x: x.load(pretrain_path, map_location=torch.device('cpu')), [self.policy.ep, self.value1.ep,
                                                                                    self.value2.ep, self.target_value1.ep,
                                                                                    self.target_value2.ep])

        # init hyperparameters
        self.tau = self.parameter.sac_tau
        self.target_entropy = -self.parameter.target_entropy_ratio * self.act_dim
        self.n_warmup = self.parameter.random_num
        self.rnn_slice_num = self.parameter.rnn_slice_num
        self.rnn_fix_length = self.parameter.rnn_fix_length
        self.update_interval = self.parameter.update_interval
        self.sac_mini_batch_size = self.parameter.sac_mini_batch_size
        self.policy_freq = self.parameter.policy_freq
        self.autoalpha = self.parameter.autoalpha
        
        # init optimizer: policy optimizer, value optimizer, alpha optimizer
        self.policy_parameter = [*self.policy.parameters(True)]
        self.policy_optimizer = torch.optim.Adam(self.policy_parameter, lr=self.parameter.learning_rate)
        self.value_parameter = [*self.value1.parameters(True)] + [*self.value2.parameters(True)]
        if self.parameter.stop_pg_for_ep:
            self.value_parameter = [*self.value1.up.parameters(True)] + [*self.value2.up.parameters(True)]
        self.value_optimizer = torch.optim.Adam(self.value_parameter,
                                                lr=self.parameter.value_learning_rate)
        if self.autoalpha:
            self.log_sac_alpha = (torch.ones((1)).to(torch.get_default_dtype()
                                            ) * np.log(self.parameter.sac_alpha)).to(self.device).requires_grad_(True)
            self.alpha_optimizer = torch.optim.Adam([self.log_sac_alpha], lr=1e-2)
        else:
            self.alpha = self.parameter.alpha
        to_device(self.device, self.policy, self.policy_for_test, self.value1, self.value2, self.target_value1, self.target_value2)
        
        # init loss: rmdm loss, contrastive loss
        self.rmdm_loss = RMDMLoss(max_env_len=self.parameter.task_num, tau=self.parameter.rmdm_tau)
        if self.parameter.use_contrastive:
            self.contrastive_loss = ContrastiveLoss(self.parameter.ep_dim, self.parameter.task_num, ep=self.policy_target.ep)
            to_device(self.device, self.contrastive_loss)
        else:
            self.contrastive_loss = None
        self.logger.log(f'size of parameter of policy: {len(self.policy_parameter)}, '
                        f'size of parameter of value: {len(self.value_parameter)}')
        self.log_consis_w_alpha = (torch.ones((1)).to(torch.get_default_dtype()) * np.log(
            self.parameter.consistency_loss_weight)).to(self.device).requires_grad_(True)
        self.log_diverse_w_alpha = (torch.ones((1)).to(torch.get_default_dtype()) * np.log(
            self.parameter.diversity_loss_weight)).to(self.device).requires_grad_(True)
        self.repre_loss_factor = self.parameter.repre_loss_factor
        self.w_log_max = 5.5
        
        # no use
        self.w_optimizer = torch.optim.SGD([self.log_consis_w_alpha, self.log_diverse_w_alpha], lr=1e-1)
        self.transition = None
        self.transition_optimizer = None
        
        # init some variables
        self.all_repre = None
        self.all_valids = None
        self.all_repre_target = None
        self.all_valids_target = None
        self.all_tasks_validate = None
        
        self.env_param_dict = self.get_env_param_dict(self.env_tasks)
        # self.training_agent.make_env_param_dict_from_params(self.parameter.varying_params)
        self.logger.log('environment parameter dict: ')
        self.logger.log_dict_single(self.env_param_dict)
        self.test_env_param_dict = self.get_env_param_dict(self.test_tasks)
        self.logger.log('test environment parameter dict: ')
        self.logger.log_dict_single(self.test_env_param_dict)
        self.test_ood_env_param_dict = self.get_env_param_dict(self.ood_tasks)
        self.logger.log('test ood environment parameter dict: ')
        self.logger.log_dict_single(self.test_ood_env_param_dict)

    @staticmethod
    def global_seed(*args, seed):
        for item in args:
            item.seed(seed)
    
    @staticmethod    
    def get_env_param_dict(test_envs):
        test_env_param_dict = {}
        for i in range(len(test_envs)):
            test_env_param_dict[i + 1] = []
            for k, v in test_envs[i].items():
                if k == "gravity" or k == "dof_damping" or k == "wind" or k == "body_mass":
                    test_env_param_dict[i + 1].append(v[-1])
                elif k == "geom_friction" or k == "body_inertia":
                    test_env_param_dict[i + 1].append(v[-1][-1])
                else:
                    assert(0)
        return test_env_param_dict

    # train
    def set_replay_buffer_size(self, size):
        self.replay_buffer_size = size

    def value_function_soft_update(self):
        if self.parameter.stop_pg_for_ep:
            self.target_value1.up.copy_weight_from(self.value1.up, self.tau)
            self.target_value2.up.copy_weight_from(self.value2.up, self.tau)
        else:
            self.target_value1.copy_weight_from(self.value1, self.tau)
            self.target_value2.copy_weight_from(self.value2, self.tau)

    def _update_critic(self, state, action, next_state, reward, mask, last_action, valid, task, 
                       policy_hidden=None, value_hidden1=None, value_hidden2=None):
        
        # Calculate target Q
        self.timer.register_point('calculating_target_Q', level=3)
        
        with torch.no_grad():
            # Target actions come from *current* policy
            state_shape = state.shape
            state_dim = len(state_shape)
            if self.parameter.rnn_fix_length is None or self.parameter.rnn_fix_length == 0:
                total_state = torch.cat((state[..., :1, :], next_state), state_dim - 2)
                total_last_action = torch.cat((last_action, action[..., -1:, :]), state_dim - 2)
            else:
                total_state, total_last_action = next_state, action
            _, _, next_action_total, nextact_logprob, _ = self.policy.rsample(total_state, total_last_action, policy_hidden)  
            
            # Target ep come from *current* policy
            ep_total = self.policy.ep_tensor.detach() if self.parameter.stop_pg_for_ep else None
            if self.parameter.stop_pg_for_ep and self.rmdm_loss.history_env_mean is not None:
                ind_ = torch.abs(task[..., -1, 0]-1).to(dtype=torch.int64)
                ep_total = self.rmdm_loss.history_env_mean[ind_].detach()

            # Target Q-values
            if self.parameter.rnn_fix_length and self.parameter.stop_pg_for_ep:
                total_state, total_last_action, next_action_total, ep_total = map(
                lambda x: x[:, -1:, :], [total_state, total_last_action, next_action_total, ep_total])
            if self.use_absorbing_state:
                a_mask = torch.maximum(torch.zeros_like(mask[:, -1:, :]), mask[:, -1:, :])
                target_Q1, _ = self.target_value1.forward(total_state, total_last_action, next_action_total * a_mask, value_hidden1, ep_out=ep_total)
                target_Q2, _ = self.target_value2.forward(total_state, total_last_action, next_action_total * a_mask, value_hidden2, ep_out=ep_total)
            else:
                target_Q1, _ = self.target_value1.forward(total_state, total_last_action, next_action_total, value_hidden1, ep_out=ep_total)
                target_Q2, _ = self.target_value2.forward(total_state, total_last_action, next_action_total, value_hidden2, ep_out=ep_total)

            if self.parameter.rnn_fix_length is None or self.parameter.rnn_fix_length == 0:
                target_Q1, target_Q2, nextact_logprob = target_Q1[..., 1:, :], target_Q2[..., 1:, :], \
                                                        nextact_logprob[..., 1:, :]
                total_reward, total_mask = reward, mask
            else:
                if not self.parameter.stop_pg_for_ep:
                    target_Q1, target_Q2 = target_Q1[..., -1:, :], target_Q2[..., -1:, :]
                nextact_logprob, total_reward, total_mask = nextact_logprob[..., -1:, :], reward[..., -1:, :], mask[..., -1:, :]
            if self.autoalpha:
                self.alpha = self.log_sac_alpha.exp().detach()
            target_v = torch.min(target_Q1, target_Q2) - self.alpha * nextact_logprob
            # target_v = torch.min(target_Q1, target_Q2)  # for TD3
            if self.use_absorbing_state:
                target_Q = (total_reward + self.parameter.gamma * target_v).detach()
            else:
                target_Q = (total_reward + (total_mask * self.parameter.gamma * target_v)).detach()
        self.timer.register_end(level=3)
        
        # Calculate current Q
        self.timer.register_point('calculating_current_Q', level=3)
        
        # Current actions and ep
        _, _, action_rsample, logprob, _ = self.policy.rsample(state, last_action, policy_hidden)
        ep = self.policy.ep_tensor
        ep_current = self.policy.ep_tensor.detach() if self.parameter.stop_pg_for_ep else None
        if self.parameter.stop_pg_for_ep and self.rmdm_loss.history_env_mean is not None:
            ep_current = self.rmdm_loss.history_env_mean[torch.abs(task[..., -1, 0] - 1).to(dtype=torch.int64)].detach()
        
        # Current Q-values
        if self.parameter.rnn_fix_length and self.parameter.stop_pg_for_ep:
            state, last_action, action, ep_current = state[:, -1:, :], last_action[:, -1:, :], action[:, -1:, :], ep_current[:, -1:, :]
        current_Q1, _ = self.value1.forward(state, last_action, action, value_hidden1, ep_out=ep_current)
        current_Q2, _ = self.value2.forward(state, last_action, action, value_hidden2, ep_out=ep_current)
        if self.parameter.rnn_fix_length and not self.parameter.stop_pg_for_ep:
            current_Q1, current_Q2 = current_Q1[:, -1:, :], current_Q2[:, -1:, :]
        self.timer.register_end(level=3)
        
        # Calulate td loss
        if self.parameter.rnn_fix_length is None or self.parameter.rnn_fix_length == 0:
            valid_num = valid.sum()
            q1_loss, q2_loss = map(lambda c: ((c - target_Q) * valid).pow(2).sum() / valid_num, [current_Q1, current_Q2])
        else:
            q1_loss = (current_Q1 - target_Q).pow(2).mean()
            q2_loss = (current_Q2 - target_Q).pow(2).mean()
        # critic_loss = q1_loss  # for DDPG
        critic_loss = q1_loss + q2_loss

        # Update critic
        self.timer.register_point('value_optimization', level=3)
        self.value_optimizer.zero_grad()
        critic_loss.backward()
        if torch.isnan(critic_loss).any().item():
            self.logger.log(f"nan found in critic loss, state: {state.abs().sum()}, "
                            f"last action: {last_action.abs().sum()}, "
                            f"action: {action.abs().sum()}")
            return None
        self.value_optimizer.step()
        self.timer.register_end(level=3)
        
        return critic_loss, action_rsample, logprob, ep_current, ep

    def _update_policy(self, state, action, last_action, valid, task, env_param, action_rsample, logprob, ep_current, ep,
                       policy_hidden=None, value_hidden1=None, value_hidden2=None, can_optimize_ep=True, mask=None):
        # Calculate actor loss
        self.timer.register_point('actor_loss', level=3)
        
        # Calculate current Q-values
        if self.parameter.rnn_fix_length and self.parameter.stop_pg_for_ep:
            state, last_action, action_rsample, ep_current = map(lambda x: x[:, -1:, :], 
                [state, last_action, action_rsample, ep_current])
        actor_q1, _ = self.value1.forward(state, last_action, action_rsample, value_hidden1, ep_out=ep_current)
        actor_q2, _ = self.value2.forward(state, last_action, action_rsample, value_hidden2, ep_out=ep_current)
        if self.parameter.rnn_fix_length:
            actor_q1, actor_q2, logprob = map(lambda x: x[..., -1:, :], [actor_q1, actor_q2, logprob])
        # actor_q = actor_q1  # for DDPG and TD3
        actor_q = torch.min(actor_q1, actor_q2)
        
        # Calculate loss of policy
        if self.parameter.rnn_fix_length is None or self.parameter.rnn_fix_length == 0:
            valid_num = valid.sum()
        else:
            valid_num = state.shape[0]
        if self.autoalpha:
            self.alpha = self.log_sac_alpha.exp().detach()
        if self.parameter.rnn_fix_length and self.use_absorbing_state:
            a_mask = 1.0 - torch.maximum(torch.zeros_like(mask[..., -1:, :]), -mask[..., -1:, :])
            # actor_loss = -torch.sum(actor_q * a_mask) / torch.sum(a_mask)  # for DDPG and TD3
            actor_loss = torch.sum(a_mask * (self.alpha * logprob - actor_q)) / torch.sum(a_mask)
        elif self.parameter.rnn_fix_length:
            # actor_loss = -actor_q.mean()  # for DDPG and TD3
            actor_loss = (self.alpha * logprob - actor_q).mean()
        else:
            actor_loss = ((self.alpha * logprob - actor_q) * valid).sum() / valid_num
        
        # Calculate RMDM loss
        consis_w = torch.exp(self.log_consis_w_alpha)
        diverse_w = torch.exp(self.log_diverse_w_alpha)
        rmdm_loss_tensor = consistency_loss = diverse_loss = None
        batch_task_num = 1
        if self.parameter.ep_smooth_factor > 0:
            ep = self.policy.tmp_ep_res(state, last_action, policy_hidden)
        if self.parameter.use_rmdm and not self.parameter.share_ep and not self.freeze_ep and can_optimize_ep:
            self.timer.register_point('rmdm_loss', level=5)
            if self.parameter.rnn_fix_length:
                ep = ep[..., -1:, :]
            if self.parameter.rnn_fix_length:
                rmdm_loss_tensor, consistency_loss, diverse_loss, batch_task_num, consis_w_loss, diverse_w_loss, \
                    all_repre, all_valids = self.rmdm_loss.rmdm_loss_timing(ep, task, valid, consis_w, diverse_w, True, True,
                                                                            rbf_radius=self.parameter.rbf_radius,
                                                                            kernel_type=self.parameter.kernel_type)
            else:
                rmdm_loss_tensor, consistency_loss, diverse_loss, batch_task_num, consis_w_loss, diverse_w_loss, \
                    all_repre, all_valids = self.rmdm_loss.rmdm_loss(ep, task, valid, consis_w, diverse_w, True, True, 
                                                                     rbf_radius=self.parameter.rbf_radius,
                                                                     kernel_type=self.parameter.kernel_type)
            self.all_repre = [item.detach() for item in all_repre]
            self.all_valids = [item.detach() for item in all_valids]
            self.all_tasks = self.rmdm_loss.lst_tasks
            do_not_train_ep = False
            if self.replay_buffer_size < self.parameter.minimal_repre_rp_size\
                    or len(self.all_tasks) < int(0.5 * self.parameter.task_num):
                do_not_train_ep = True
            if rmdm_loss_tensor is not None and not do_not_train_ep:
                if torch.isnan(consistency_loss).any().item() or torch.isnan(diverse_loss).any().item():
                    self.logger.log(f'rmdm produce nan: consistency: {consistency_loss.item()}, '
                                    f'diverse loss: {diverse_loss.item()}')
                actor_loss = actor_loss + rmdm_loss_tensor * self.repre_loss_factor
                if self.parameter.l2_norm_for_ep > 0.0 :
                    l2_norm_for_ep = 0
                    ep = self.policy.ep if self.parameter.ep_smooth_factor == 0.0 else self.policy.ep_temp
                    for parameter_ in ep.parameters(True): # 8
                        l2_norm_for_ep = l2_norm_for_ep + torch.norm(parameter_).pow(2)
                    actor_loss = actor_loss + l2_norm_for_ep* self.parameter.l2_norm_for_ep
            else:
                pass
            self.timer.register_end(level=5)
        else:
            self.timer.register_point('rmdm_loss', level=5)
            self.timer.register_end(level=5)

        # Calculate uposi loss
        uposi_loss = None
        if self.parameter.use_uposi and not self.parameter.share_ep and not self.freeze_ep:
            target_ep_output = env_param
            if self.parameter.rnn_fix_length:
                target_ep_output = target_ep_output[..., -1:, -2:]
                ep = ep[..., -1:, :]
                uposi_loss = (ep - target_ep_output).pow(2).mean()
            else:
                uposi_loss = ((ep - target_ep_output[..., -2:]).pow(2) * valid).sum() / valid_num
            actor_loss = actor_loss + uposi_loss
            pass
        
        # Caliculate contrastive_loss
        contrastive_loss = None
        if self.parameter.use_contrastive and not self.parameter.share_ep and not self.freeze_ep:
            query_tensor = self.contrastive_loss.get_query_tensor(state, last_action)
            if self.parameter.rnn_fix_length:
                ep = ep[..., -1:, :]
                contrastive_loss = self.contrastive_loss.contrastive_loss(ep, query_tensor, task)
                actor_loss = actor_loss + contrastive_loss
        self.timer.register_end(level=3)
        
        # Update actor
        self.timer.register_point('policy_optimization', level=3)
        self.policy_optimizer.zero_grad()
        if torch.isnan(actor_loss).any().item():
            self.logger.log(f"nan found in actor loss, state: {state.abs().sum()}, "
                            f"last action: {last_action.abs().sum()}, "
                            f"action: {action.abs().sum()}")
            return None
        actor_loss.backward()
        self.policy_optimizer.step()
        if self.parameter.ep_smooth_factor > 0:
            self.policy.apply_temp_ep(self.parameter.ep_smooth_factor)
        self.timer.register_end(level=3)
        
        # Update alpha
        if self.autoalpha: 
            if self.parameter.rnn_fix_length and self.use_absorbing_state:
                a_mask = 1.0 - torch.maximum(torch.zeros_like(mask[..., -1:, :]), -mask[..., -1:, :])
                alpha_loss = (a_mask * self.log_sac_alpha * (logprob + self.target_entropy).detach()).sum() / a_mask.sum()
            elif self.parameter.rnn_fix_length:
                alpha_loss = - (self.log_sac_alpha * ((logprob + self.target_entropy)).detach()).mean()
            else:
                alpha_loss = - (self.log_sac_alpha * ((logprob + self.target_entropy) * valid).detach()).sum() / valid_num
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        if self.autoalpha:
            with torch.no_grad():
                self.log_sac_alpha.clamp_max_(0)
        with torch.no_grad():
            self.log_diverse_w_alpha.clamp_max_(self.w_log_max)
            self.log_consis_w_alpha.clamp_max_(self.w_log_max)
            
        # Update target networks
        self.value_function_soft_update()
        if self.parameter.share_ep:
            self.policy.ep.copy_weight_from(self.value1.ep, tau=0.0)
            
        # Logging
        if self.parameter.rnn_fix_length:
            q_mean1 = actor_q1.mean().item()
            q_mean2 = actor_q2.mean().item()
            logp_pi = logprob.mean().item()
        else:
            q_mean1 = ((actor_q1 * valid).sum() / valid_num).item()
            q_mean2 = ((actor_q2 * valid).sum() / valid_num).item()
            logp_pi = ((logprob * valid).sum() / valid_num).item()
        if self.parameter.use_rmdm:
            rmdm_loss_float = rmdm_loss_tensor.item() if rmdm_loss_tensor is not None else 0
            consistency_loss_float = consistency_loss.item() if consistency_loss is not None else 0
            diverse_loss_float = diverse_loss.item() if diverse_loss is not None else 0
        else:
            rmdm_loss_float = 0.0
            consistency_loss_float = 0.0
            diverse_loss_float = 0.0
            batch_task_num = 0
        if self.contrastive_loss is not None:
            self.contrastive_loss.update_ep(self.policy.ep) 
        
        return actor_loss, q_mean1, q_mean2, logp_pi, \
               rmdm_loss_float, consistency_loss_float, diverse_loss_float, batch_task_num, \
               consis_w, diverse_w, uposi_loss, contrastive_loss

    def _sac_update(self, state, action, next_state, reward, mask, last_action, valid, task, env_param,
                    policy_hidden=None, value_hidden1=None, value_hidden2=None, can_optimize_ep=True):
        critic_loss, action_rsample, logprob, ep_current, ep = self._update_critic(
            state, action, next_state, reward, mask, last_action, valid, task, 
            policy_hidden, value_hidden1, value_hidden2)
        
        actor_loss, q_mean1, q_mean2, logp_pi, rmdm_loss_float, consistency_loss_float, diverse_loss_float, \
        batch_task_num, consis_w, diverse_w, uposi_loss, contrastive_loss = self._update_policy(
            state, action, last_action, valid, task, env_param, action_rsample, logprob, ep_current, ep,
            policy_hidden, value_hidden1, value_hidden2, can_optimize_ep, mask=mask)
        if self.autoalpha:
            alpha = self.log_sac_alpha.item()
        else:
            alpha = self.alpha
        
        return dict(
            CriticLoss=critic_loss.item(),
            ActorLoss=actor_loss.item(),
            Alpha=alpha,
            QMean1=q_mean1,
            QMean2=q_mean2,
            Logp=logp_pi,
            PolicyGradient=0,
            ValueGradient=0,
            rmdmLoss=rmdm_loss_float,
            ConsistencyLoss=consistency_loss_float,
            DiverseLoss=diverse_loss_float,
            BatchTaskNum=batch_task_num,
            w_loss=0,
            consistency_w=consis_w.item(),
            diverse_w=diverse_w.item(),
            UPOSILoss=0 if uposi_loss is None else uposi_loss.item(),
            TransitionLoss=0,
            ContrastiveLoss=0 if contrastive_loss is None else contrastive_loss.item(),
        )
        
    def train(self, states, next_states, actions, last_action, rewards, masks, valid, task, env_param):

        # Numpy to tensor
        self.timer.register_point('from_numpy', level=1)
        if self.parameter.bottle_neck:
            self.policy.set_deterministic_ep(deterministic=False)
        states, next_states, actions, last_action, rewards, masks, valid, task, env_param = \
                map(lambda x: torch.from_numpy(np.array(x)).to(dtype=torch.get_default_dtype(), device=self.device),
                [states, next_states, actions, last_action, rewards, masks, valid, task, env_param])
        self.timer.register_end(level=1)
        
        # Process data
        self.timer.register_point('process_data', level=1)
        
        # Make slice
        self.timer.register_point('making_slice', level=3)
        if self.parameter.rnn_fix_length is None or self.parameter.rnn_fix_length == 0:
            self.timer.register_point('generate_hidden_state', level=4)
            hidden_policy = self.policy.generate_hidden_state(states, last_action,
                                                              slice_num=self.parameter.rnn_slice_num)
            hidden_value1 = self.value1.generate_hidden_state(states, last_action, actions,
                                                              slice_num=self.parameter.rnn_slice_num)
            hidden_value2 = self.value2.generate_hidden_state(states, last_action, actions,
                                                                slice_num=self.parameter.rnn_slice_num)
            self.timer.register_end(level=4)
            
            self.timer.register_point('Policy.slice_tensor', level=4)
            states, next_states, actions, last_action, rewards, masks, valid, task, env_param = \
                map(Policy.slice_tensor, [states, next_states, actions, last_action, rewards, masks, valid, task, env_param],
                    [self.parameter.rnn_slice_num] * 9)
            self.timer.register_end(level=4)

            self.timer.register_point('Policy.merge_slice_tensor', level=4)
            states, next_states, actions, last_action, rewards, masks, valid, task, env_param = Policy.merge_slice_tensor(
                states, next_states, actions, last_action, rewards, masks, valid, task, env_param)
            self.timer.register_end(level=4)
            
            self.timer.register_point('mask_and_sample', level=4)
            mask_for_valid = valid.sum(dim=-2, keepdim=True)[..., 0, 0] > 0
            states, next_states, actions, last_action, rewards, masks, valid, task, env_param = map(
                lambda x: x[mask_for_valid],
                [states, next_states, actions, last_action, rewards, masks, valid, task, env_param])
            hidden_policy, hidden_value1, hidden_value2 = map(
                Policy.hidden_state_mask,
                [hidden_policy, hidden_value1, hidden_value2],
                [mask_for_valid, mask_for_valid, mask_for_valid])
            minibatch_size = self.parameter.sac_mini_batch_size
            traj_num = min(max(minibatch_size // self.parameter.rnn_slice_num, 1), states.shape[0])
            total_inds = np.random.permutation(states.shape[0]).tolist()[:traj_num]
            hidden_policy, hidden_value1, hidden_value2 = \
                map(Policy.hidden_state_sample,
                    [hidden_policy, hidden_value1, hidden_value2],
                    [total_inds, total_inds, total_inds])
            states, next_states, actions, last_action, rewards, masks, valid, task, env_param = \
                map(lambda x: x[total_inds],
                    [states, next_states, actions, last_action, rewards, masks, valid, task, env_param])
            self.timer.register_end(level=4)
        else:
            self.timer.register_point('generate_hidden_state', level=4)
            hidden_policy = self.policy.make_init_state(batch_size=states.shape[0], device=states.device)
            hidden_value1 = self.value1.make_init_state(batch_size=states.shape[0], device=states.device)
            hidden_value2 = self.value2.make_init_state(batch_size=states.shape[0], device=states.device)
            self.timer.register_end(level=4)
        self.timer.register_end(level=3)
        
        # Detach
        self.timer.register_point('detach', level=3)
        states, next_states, actions, last_action, rewards, masks, valid, task, env_param = map(
            lambda x: x.detach(),
            [states, next_states, actions, last_action, rewards, masks, valid, task, env_param])
        hidden_policy, hidden_value1, hidden_value2 = map(
            Policy.hidden_detach,
            [hidden_policy, hidden_value1, hidden_value2])
        self.timer.register_end(level=3)
        self.timer.register_end(level=1)
        
        with torch.set_grad_enabled(True):
            point_num = states.shape[0]
            total_inds = np.random.permutation(point_num).tolist()
            iter_batch_size = states.shape[0] // self.parameter.sac_inner_iter_num
            for i in range(self.parameter.sac_inner_iter_num):
                # Sample from batch
                self.timer.register_point('sample_from_batch', level=1)
                start = i * iter_batch_size
                end = min((i+1) * iter_batch_size, states.shape[0])
                states_batch, next_states_batch, actions_batch, \
                last_action_batch, rewards_batch, masks_batch, valid_batch, task_batch, env_param_batch = \
                map(lambda x: x[start: end], [
                    states, next_states, actions, last_action, rewards, masks, valid, task, env_param])
                data_is_valid = False
                if self.parameter.rnn_fix_length:
                    if valid_batch[..., -1:, :].sum().item() >= 2:
                        data_is_valid = True
                elif valid_batch.sum().item() >= 2:
                    data_is_valid = True
                if not data_is_valid:
                    print('data is not valid!!')
                    continue
                hidden_policy_batch, hidden_value1_batch, hidden_value2_batch = \
                    map(Policy.hidden_state_slice,
                        [hidden_policy, hidden_value1, hidden_value2], [start] * 3, [end] * 3)
                self.timer.register_end(level=1)
                
                # Update
                self.timer.register_point('self.sac_update', level=1)
                can_optimize_ep = self.replay_buffer_size > self.parameter.ep_start_num
                res_dict = self._sac_update(states_batch, actions_batch, next_states_batch, rewards_batch,
                                            masks_batch, last_action_batch, valid_batch, task_batch, env_param_batch,
                                            hidden_policy_batch, hidden_value1_batch, hidden_value2_batch, can_optimize_ep)
                self.timer.register_end(level=1)
        
        self.total_iteration += 1
        
        log = {}
        if res_dict is not None:
            for key in res_dict:
                if key in log:
                    log[key].append(res_dict[key])
                else:
                    log[key] = [res_dict[key]]
        return log

    def get_all_repre(self, tasks_type, step_num):
        if tasks_type == "train":
            env_tasks = self.env_tasks
        elif tasks_type == "test":
            env_tasks = self.test_tasks
        elif tasks_type == "ood":
            env_tasks = self.ood_tasks
        
        agent = EnvRemoteArray(parameter=self.parameter, env_name=self.parameter.env_name,
                               worker_num=1, seed=self.parameter.seed + 4,
                               deterministic=True, use_remote=False, policy_type=Policy,
                               history_len=self.parameter.history_length,
                               env_decoration=NonstationaryEnv, env_tasks=env_tasks,
                               use_true_parameter=self.parameter.use_true_parameter,
                               non_stationary=True, fix_env=0)

        self.policy_for_test.ep.copy_weight_from(self.policy.ep, 0.0)
        self.policy_for_test.up.copy_weight_from(self.policy.up, 0.0)
        self.policy_for_test.to(torch.device('cpu'))
        if self.parameter.bottle_neck:
            self.policy_for_test.set_deterministic_ep(True)

        real_param, ep_traj = [], []
        for _ in range(step_num):
            mem, log, info = agent.sample1step1env(self.policy_for_test, False, render=False, need_info=True)
            real_param.append(info["env_params"])
            ep_traj.append(self.policy_for_test.ep_tensor[:1, ...].squeeze().detach().cpu().numpy())
        
        data = np.concatenate([np.array(real_param)[..., np.newaxis], np.array(ep_traj)], axis=1)
        return data

    # test
    def agent_submit_task(self):
        self.future_test = self.test_agent.submit_task(self.parameter.test_sample_num, self.policy)
        self.future_ood = self.ood_agent.submit_task(self.parameter.test_sample_num, self.policy)
        self.future_ns = self.non_station_agent.submit_task(self.parameter.test_sample_num, self.policy)
        self.future_ood_ns = self.ood_ns_agent.submit_task(self.parameter.test_sample_num, self.policy)

    def agent_query_sample(self):
        batch_test, log_test, mem_test = self.test_agent.query_sample(self.future_test, need_memory=True)
        batch_ood, log_ood, mem_ood = self.ood_agent.query_sample(self.future_ood, need_memory=True)
        batch_non_station, log_non_station, mem_non_station = self.non_station_agent.query_sample(self.future_ns, need_memory=True)
        batch_ood_ns, log_ood_ns, mem_ood_ns = self.ood_ns_agent.query_sample(self.future_ood_ns, need_memory=True)
        self.logger.add_tabular_data(tb_prefix='evaluation', **self.append_key(log_test, "Test"))
        self.logger.add_tabular_data(tb_prefix='evaluation', **self.append_key(log_ood, "OOD"))
        self.logger.add_tabular_data(tb_prefix='evaluation', **self.append_key(log_non_station, "NS"))
        self.logger.add_tabular_data(tb_prefix='evaluation', **self.append_key(log_ood_ns, 'OOD_NS'))
        self.logger.log_tabular('OODDeltaVSTestRet', np.mean(log_test['EpRet']) - np.mean(log_ood['EpRet']), tb_prefix='evaluation')
        self.logger.log_tabular('NSDeltaVSTestRet', np.mean(log_non_station['EpRet']) - np.mean(log_test['EpRet']), tb_prefix='evaluation')

    def visualize_representation(self, step):
        if self.all_repre is not None:
            fig, fig_mean = visualize_repre(self.all_repre, self.all_valids,
                                            os.path.join(self.logger.output_dir, 'visual.png'),
                                            self.env_param_dict, self.all_tasks )
            fig_real_param = visualize_repre_real_param(self.all_repre, self.all_valids, self.all_tasks,
                                                        self.env_param_dict)
            if fig is not None:
                self.logger.tb.add_figure('figs/repre', fig, step)
                self.logger.tb.add_figure('figs/repre_mean', fig_mean, step)
                self.logger.tb.add_figure('figs/repre_real', fig_real_param, step)

    @staticmethod
    def append_key(d, tail):
        res = {}
        for k, v in d.items():
            res[k+tail] = v
        return res

    def update_log(self, step):
        # timestep and period
        self.logger.log_tabular('TotalInteraction', step, tb_prefix='timestep')
        self.logger.log_tabular('ReplayBufferSize', self.replay_buffer_size, tb_prefix='timestep')
        self.logger.add_tabular_data(tb_prefix='period', **self.timer.summary())
        
        # performance
        self.logger.tb_header_dict['EpRet'] = 'performance'
        self.logger.tb_header_dict['EpRetOOD'] = 'performance'
        self.logger.tb_header_dict['EpRetOOD_NS'] = 'performance'
        self.logger.tb_header_dict['EpRetTest'] = 'performance'
        self.logger.tb_header_dict['EpRetNS'] = 'performance'
        
        # representation
        self.logger.tb_header_dict['rmdmLoss'] = 'representations'
        self.logger.tb_header_dict['DiverseLoss'] = 'representations'
        self.logger.tb_header_dict['ConsistencyLoss'] = 'representations'

        self.logger.dump_tabular()

    def save(self):
        self.policy.save(self.logger.model_output_dir)
        self.value1.save(self.logger.model_output_dir, 0)
        self.value2.save(self.logger.model_output_dir, 1)
        self.target_value1.save(self.logger.model_output_dir, "target0")
        self.target_value2.save(self.logger.model_output_dir, "target1")
        torch.save(self.policy_optimizer.state_dict(), os.path.join(self.logger.model_output_dir, 'policy_optim.pt'))
        torch.save(self.value_optimizer.state_dict(), os.path.join(self.logger.model_output_dir, 'value_optim.pt'))
        if self.autoalpha:
            torch.save(self.alpha_optimizer.state_dict(), os.path.join(self.logger.model_output_dir, 'alpha_optim.pt'))

    def load(self, model_dir=None):
        model_dir = self.logger.model_output_dir if model_dir is None else model_dir
        self.policy.load(model_dir, map_location=self.device)
        self.value1.load(model_dir, 0, map_location=self.device)
        self.value2.load(model_dir, 1, map_location=self.device)
        self.target_value1.load(model_dir, "target0", map_location=self.device)
        self.target_value2.load(model_dir, "target1", map_location=self.device)
        self.policy_optimizer.load_state_dict(torch.load(os.path.join(model_dir, 'policy_optim.pt'),
                                                         map_location=self.device))
        self.value_optimizer.load_state_dict(torch.load(os.path.join(model_dir, 'value_optim.pt'),
                                                        map_location=self.device))
        if self.parameter.autoalpha:
            self.alpha_optimizer.load_state_dict(torch.load(os.path.join(model_dir, 'alpha_optim.pt'),
                                                        map_location=self.device))
    def test_non_stationary_repre(self, step):
        self.policy_for_test.ep.copy_weight_from(self.policy.ep, 0.0)
        self.policy_for_test.up.copy_weight_from(self.policy.up, 0.0)
        self.policy_for_test.to(torch.device('cpu'))
        if self.parameter.bottle_neck:
            self.policy_for_test.set_deterministic_ep(True)
        fig, diff_from_expert = self.get_figure(self.policy_for_test, self.non_station_agent_single_thread, 1000)
        self.policy_for_test.to(self.device)
        self.logger.tb.add_figure('figs/policy_behaviour', fig, step)
        self.logger.log_tabular('DiffFromExpert', diff_from_expert[0], tb_prefix='performance')
        self.logger.log_tabular('AtTargetRatio', diff_from_expert[1], tb_prefix='performance')

    def test_non_stationary_repre_clean(self, agent, step_num=1000):
        self.policy_for_test.ep.copy_weight_from(self.policy.ep, 0.0)
        self.policy_for_test.up.copy_weight_from(self.policy.up, 0.0)
        self.policy_for_test.to(torch.device('cpu'))
        if self.parameter.bottle_neck:
            self.policy_for_test.set_deterministic_ep(True)

        real_param, ep_traj = [], []
        for _ in range(step_num):
            mem, log, info = agent.sample1step1env(self.policy_for_test, False, render=False, need_info=True)
            real_param.append(mem.memory[0].env_param)
            ep_traj.append(self.policy_for_test.ep_tensor[:1, ...].squeeze().detach().cpu().numpy())
        
        return real_param, ep_traj

    @staticmethod
    def get_figure(policy, agent, step_num, title=''):
        fig = plt.figure(18, dpi=200)
        plt.cla()
        ep_traj = []
        real_param = []
        action_discrepancy = []
        keep_at_target = []
        done = False
        while not done:
            mem, log, info = agent.sample1step1env(policy, False, render=False, need_info=True)
            done = mem.memory[0].done[0]
        for i in range(step_num):
            mem, log, info = agent.sample1step1env(policy, False, render=False, need_info=True)
            real_param.append(mem.memory[0].env_param)
            ep_traj.append(policy.ep_tensor[:1, ...].squeeze().detach().cpu().numpy())
            if isinstance(info, dict) and 'action_discrepancy' in info and info['action_discrepancy'] is not None:
                action_discrepancy.append(np.array([info['action_discrepancy'][0],
                                                    info['action_discrepancy'][1]]))
                keep_at_target.append(1 if info['keep_at_target'] else 0)
        ep_traj = np.array(ep_traj)
        real_param = np.array(real_param)
        change_inds = np.where(np.abs(np.diff(real_param[:, -1])) > 0)[0] + 1
        # print(np.hstack((ep_traj, real_param)))
        plt.plot(ep_traj[:, 0], label='x')
        plt.plot(ep_traj[:, 1], label='y')
        plt.plot(real_param[:, -1], label='real')
        diff_from_expert = 0
        at_target_ratio = 0
        if len(action_discrepancy) > 0:
            action_discrepancy = np.array(action_discrepancy)
            abs_res = np.abs(action_discrepancy[:, 0]) / 3 + np.abs(action_discrepancy[:, 1]) / 3
            plt.plot(np.arange(action_discrepancy.shape[0]), abs_res, '-*', label='diff')
            plt.title('mean discrepancy: {:.3f}'.format(np.mean(abs_res)))
            diff_from_expert = np.mean(abs_res)
            at_target_ratio = np.mean(keep_at_target)
        else:
            plt.title(title)
        for ind in change_inds:
            plt.plot([ind, ind], [-1.1, 1.1], 'k--', alpha=0.2)
        plt.ylim(bottom=-1.1, top=1.1)
        plt.legend()
        return fig, (diff_from_expert, at_target_ratio)
