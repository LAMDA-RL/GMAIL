import ray
import gym
from code.algos.escp.algorithms.sac import SAC
from code.experiments.gmail_trainer import GMAILTrainer
from code.algos.gaifo import GAIfO
from code.experiments.utils import seed_torch, restore_latest_n_traj, restore_d4rl_traj
# from GAILfOCE.envs.get_env_tasks import get_env_tasks
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


if __name__ == '__main__':
    ray.init(log_to_driver=True)
    parser = GMAILTrainer.get_argument()
    parser.add_argument("--H_step", "-h", type=int, default=1)
    parser.add_argument("--beta", type=float, default=1.0)
    # parser.add_argument('--use_absorbing_state', action='store_true')  # add adsorbing states for Hopper and Walker2d
    parser = GAIfO.get_argument(parser)
    
    # policy: sac
    sac = SAC(parser, fix_env=0)
    args = parser.parse_args()
    seed_torch(seed=args.seed)
    sac.logger.log(sac.logger.parameter)
    
    # get expert data
    if args.expert_path_dir is None:
        print("Please get expert data first!! Run run_escp.py and generate_expert_data.py to train policy model and get \
              expert data with the model separately as follows.")
        print("python run_escp.py  --env_name HalfCheetah-v3 --varying_params gravity --rnn_fix_length 16 \
              --task_num 40 --ep_dim 2 --name_suffix RMDM  --rbf_radius 3000 --use_rmdm --stop_pg_for_ep --bottle_neck --test_task_num 40")
        print("python generate_expert_data.py  --env_name HalfCheetah-v3 --varying_params gravity --rnn_fix_length 16 \
              --task_num 40 --ep_dim 2 --name_suffix TEST --rbf_radius 3000 --use_rmdm --stop_pg_for_ep --bottle_neck --test_task_num 40")
        print("The default expert data directory is `data/expert_data_sac_{0}`".format(args.env_name))
    
    env = gym.make(args.env_name)
    expert_trajs = restore_latest_n_traj(
        args.expert_path_dir, n_path=16, max_steps=1000, H=args.H_step, 
        use_absorbing_state=args.use_absorbing_state, max_episode_steps=env._max_episode_steps)
    # expert_trajs = restore_d4rl_traj(args.expert_path_dir, max_steps=20000, use_absorbing_state=args.use_absorbing_state)
    state_shape = (env.observation_space.shape[0] + 1, ) if args.use_absorbing_state else env.observation_space.shape

    # irl: gailfo
    irl = GAIfO(
        state_shape=state_shape,
        units=[64, 64],
        batch_size=32,
        H=args.H_step,
        beta=args.beta,
        lr=1e-3,
        lr_decay=1.,
        use_gp=True,
        gpu=args.gpu).to(args.gpu)
    
    # train
    trainer = ESCPIrlTrainer(sac, args, irl, expert_trajs["obses"],
                             expert_trajs["next_obses"], expert_trajs["acts"])
    trainer()    
