import random
import os
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code.experiments.trainer import Trainer
from code.experiments.misc.replay_memory import MemoryArray


class ESCPTrainer(Trainer):
    """
    Trainer class for escp in nonstationary environment

    Command Line Args for Trainer:

        * ``--max-steps`` (int): The maximum steps for training. The default is ``int(1e6)``
        * ``--start-train-num`` (int): Steps to start policy train. The default is ``int(2e4)``
        * ``--test-interval`` (int): Interval to evaluate trained model. The default is ``int(1e3)``
        * ``--save-model-interval`` (int): Interval to save model. The default is ``int(1e4)``
        * ``--repre-interval`` (int): Interval to plot representation figures. The default is ``int(1e4)``

    """
    def __init__(self, policy, args):
        
        if isinstance(args, dict):
            _args = args
            args = policy.__class__.get_argument(Trainer.get_argument())
            args = args.parse_args([])
            for k, v in _args.items():
                if hasattr(args, k):
                    setattr(args, k, v)
                else:
                    raise ValueError(f"{k} is invalid parameter.")
        self._set_from_args(args)
        self._policy = policy
        self._start_train_num = args.start_train_num
        self._repre_interval = args.repre_interval
        
        self.replay_buffer = MemoryArray(policy.rnn_slice_num, max_trajectory_num=5000, 
                                         max_traj_step=1050, fix_length=policy.rnn_fix_length)
        
    def __call__(self):
        """
        Execute training
        """
        
        total_steps = 0
        
        # Init samples
        self._policy.policy.to(device=torch.device('cpu'))
        while self.replay_buffer.size <= self._start_train_num:
            random_action = self.replay_buffer.size < self._policy.n_warmup
            mem, log = self._policy.training_agent.sample1step(self._policy.policy, random_action, 
                                                               device=torch.device('cpu'))
            self.replay_buffer.mem_push(mem)
            total_steps += 1
        
        # Train policy
        while total_steps < self._max_steps + self._start_train_num:
            self._policy.policy.to(device=torch.device('cpu'))
            if total_steps % self._test_interval == 0:
                self._policy.agent_submit_task()
           
            # sample one step
            self._policy.policy.to(self._policy.device)
            random_action = self.replay_buffer.size < self._policy.n_warmup
            mem, log = self._policy.training_agent.sample1step(self._policy.policy, random_action, 
                                                               device=self._policy.device)
            self.replay_buffer.mem_push(mem)
            
            # train policy
            if total_steps % self._policy.update_interval == 0:
                self._policy.policy.to(self._policy.device)
                self._policy.set_replay_buffer_size(self.replay_buffer.size)
                if self._policy.rnn_fix_length:
                    samples = self.replay_buffer.sample_fix_length_sub_trajs(self._policy.sac_mini_batch_size,
                                                                             self._policy.rnn_fix_length)
                else:
                    samples, _ = self.replay_buffer.sample_trajs(self._policy.sac_mini_batch_size,
                                                                 max_sample_size=3e5)
                update_log = self._policy.train(
                    samples.state, samples.next_state, samples.action, samples.last_action, samples.reward,
                    samples.mask, samples.valid, samples.task, samples.env_param)
                log.update(update_log)
            self._policy.logger.add_tabular_data(**log, tb_prefix='training')
            
            # test policy
            if total_steps % self._test_interval == 0:
                self._policy.test_non_stationary_repre(total_steps)
                if total_steps % self._repre_interval == 0:
                    self._policy.visualize_representation(total_steps)
                self._policy.agent_query_sample()
                self._policy.update_log(total_steps)
            
            # save model
            if total_steps % self._save_model_interval == 0:
                self._policy.save()
        
            total_steps += 1

    @staticmethod
    def get_argument(parser=None):
        """
        Create or update argument parser for command line program

        Args:
            parser (argparse.ArgParser, optional): argument parser

        Returns:
            argparse.ArgParser: argument parser
        """
        parser = Trainer.get_argument(parser)
        
        # experiment settings
        parser.add_argument('--start-train-num', type=int, default=int(2e4),
                            help='Steps to start policy train.')
        parser.add_argument('--repre-interval', type=int, default=int(1e4), 
                            help='Interval to plot representation figures.')
        
        return parser


class GMAILTrainer(ESCPTrainer):
    """
    Trainer class for irl and escp in nonstationary environment

    Command Line Args for Trainer:

        * ``--max-steps`` (int): The maximum steps for training. The default is ``int(1e6)``
        * ``--start-train-num`` (int): Steps to start policy train. The default is ``int(2e4)``
        * ``--test-interval`` (int): Interval to evaluate trained model. The default is ``int(1e3)``
        * ``--save-model-interval`` (int): Interval to save model. The default is ``int(1e4)``
        * ``--repre-interval`` (int): Interval to plot representation figures. The default is ``int(1e4)``
        * ``--expert-path-dir`` (int): Path to directory that contains expert trajectories.

    """
    
    def __init__(self, policy, args, irl, 
                 expert_obs, expert_next_obs, expert_act):
        super().__init__(policy, args)

        self._irl = irl
        self._expert_obs = expert_obs
        self._expert_next_obs = expert_next_obs
        self._expert_act = expert_act
        self._random_range = range(expert_obs.shape[0])
        self.use_absorbing_state = args.use_absorbing_state
        self.replay_buffer = MemoryArray(policy.rnn_slice_num, max_trajectory_num=5000, max_traj_step=1050, 
                                         fix_length=policy.rnn_fix_length, use_absorbing_state=self.use_absorbing_state)

    def __call__(self):
        """
        Execute training
        """
        
        total_steps = 0

        def add_replay_buffer(mem):
            import numpy as np
            from code.algos.escp.utils.replay_memory import Memory
            if self.use_absorbing_state:
                assert(len(mem.memory) == 1)
                trainsition = mem.memory[0]
                state = np.hstack([trainsition.state, np.array(0)])
                next_state = np.hstack([trainsition.next_state, np.array(0)])
                if not trainsition.mask[0]:
                    next_state = np.zeros(trainsition.state.shape[0] + 1, )  # absorbing state
                    next_state[-1] = 1.
                mem = Memory()
                mem.push(state, trainsition.action, trainsition.mask, next_state, trainsition.reward, 
                     trainsition.next_action, trainsition.task, trainsition.env_param,
                     trainsition.last_action, trainsition.done, trainsition.valid)
            self.replay_buffer.mem_push(mem)
        
        # Init samples
        self._policy.policy.to(device=torch.device('cpu'))
        while self.replay_buffer.size <= self._start_train_num:
            random_action = self.replay_buffer.size < self._policy.n_warmup
            mem, log = self._policy.training_agent.sample1step(self._policy.policy, random_action, 
                                                               device=torch.device('cpu'))
            add_replay_buffer(mem)
            total_steps += 1
        
        # Train policy
        while total_steps < self._max_steps + self._start_train_num:
            self._policy.policy.to(device=torch.device('cpu'))
            if total_steps % self._test_interval == 0:
                self._policy.agent_submit_task()
           
            # sample one step
            self._policy.policy.to(self._policy.device)
            random_action = self.replay_buffer.size < self._policy.n_warmup
            mem, log = self._policy.training_agent.sample1step(self._policy.policy, random_action, 
                                                               device=self._policy.device)
            add_replay_buffer(mem)
            
            # train
            if total_steps % self._policy.update_interval == 0:
                self._policy.policy.to(self._policy.device)
                
                # sample trajs from replay buffer
                self._policy.set_replay_buffer_size(self.replay_buffer.size)
                if self._policy.rnn_fix_length:
                    samples = self.replay_buffer.sample_fix_length_sub_trajs(self._policy.sac_mini_batch_size,
                                                                             self._policy.rnn_fix_length)
                else:
                    samples, _ = self.replay_buffer.sample_trajs(self._policy.sac_mini_batch_size,
                                                                 max_sample_size=3e5)
                
                # train policy
                batch_size, rnn_length, _ = samples.state.shape
                rew = self._irl.inference(samples.state.reshape((batch_size * rnn_length, -1)), 
                                          samples.action.reshape((batch_size * rnn_length, -1)), 
                                          samples.next_state.reshape((batch_size * rnn_length, -1)))
                rew = rew.reshape((batch_size, rnn_length, 1)).detach().cpu().numpy()
                # rew = torch.tensor(samples.reward)
                update_log = self._policy.train(
                    samples.state, samples.next_state, samples.action, samples.last_action, rew,
                    samples.mask, samples.valid, samples.task, samples.env_param)
                log.update(update_log)
                
                # train IRL
                for _ in range(self._irl.n_training):
                    samples = self.replay_buffer.sample_transitions(self._irl.batch_size)
                    indices = random.sample(self._random_range, self._irl.batch_size)
                    updata_log = self._irl.train(
                        agent_states=samples.state,
                        agent_acts=samples.action,
                        agent_next_states=samples.next_state,
                        expert_states=self._expert_obs[indices],
                        expert_acts=self._expert_act[indices],
                        expert_next_states=self._expert_next_obs[indices], 
                        total_steps = total_steps)
                    log.update(updata_log)
            
            self._policy.logger.add_tabular_data(**log, tb_prefix='training')
            
            # test policy
            if total_steps % self._test_interval == 0:
                self._policy.test_non_stationary_repre(total_steps)
                if total_steps % self._repre_interval == 0:
                    self._policy.visualize_representation(total_steps)
                self._policy.agent_query_sample()
                self._policy.update_log(total_steps)
            
            # save model
            if total_steps % self._save_model_interval == 0:
                self._policy.save()
        
            total_steps += 1
        

    @staticmethod
    def get_argument(parser=None):
        """
        Create or update argument parser for command line program

        Args:
            parser (argparse.ArgParser, optional): argument parser

        Returns:
            argparse.ArgParser: argument parser
        """
        parser = ESCPTrainer.get_argument(parser)
        parser.add_argument('--expert-path-dir', default=None,
                            help='Path to directory that contains expert trajectories.')
        return parser

