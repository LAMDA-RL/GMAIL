

"""
You can run this programme with command: 
python generate_expert_data.py  --env_name HalfCheetah-v3 --varying_params gravity --task_num 40 --ep_dim 2 --test_task_num 40 
                                --rnn_fix_length 16 --rbf_radius 3000 --use_rmdm --stop_pg_for_ep --bottle_neck 
                                --name_suffix TEST
"""

def sample1traj(policy, agent, env_ind):
    import numpy as np
    traj = {'obs': [], 'act': []}
    done = False
    task_return = 0
    while not done:
        mem, log, info = agent.sample1step1env(policy, False, env_ind=env_ind, render=False, need_info=True)
        done = mem.memory[0].done[0]
        task_return += mem.memory[0].reward[0]
        traj['obs'].append(np.array(mem.memory[0].state))
        traj['act'].append(np.array(mem.memory[0].action))
    traj['obs'] = np.array(traj['obs'])
    traj['act'] = np.array(traj['act'])
    return traj, task_return


def escp_main():
    import ray
    from code.envs.env_utils import save_path
    from code.experiments.utils import seed_torch
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from code.algos.escp.algorithms.sac import SAC
    ray.init(log_to_driver=True)
    
    # load model
    fix_env = 0
    model = SAC(parser=None, fix_env=fix_env)
    seed_torch(seed=1)
    model.logger.log(model.logger.parameter)
    # model_dir = 'log_file/2022-12-02 16-47-58 HalfCheetah-v3_ESCP/model'
    model_dir = 'trained_escp_model'
    model.policy.load(model_dir, map_location=model.device)
    
    data_dir = "data/expert_data_escp-" + model.parameter.env_name
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # get trajs
    tasks = model.training_agent.env_tasks
    model.station_agent_single_thread.workers[0].env_tasks = tasks
    model.policy.to('cpu')
    model.station_agent_single_thread.workers[0].reset(env_ind=0)
    for task_ind in range(len(tasks)):
        traj, ret = sample1traj(model.policy, model.station_agent_single_thread, env_ind=(task_ind + 1) % 20)
        print("Task: {0}, Return: {1}".format(tasks[task_ind], ret))
        filename = "step_{0:08d}_epi_{1:02d}_return_{2:010}.pkl".format(0, task_ind, ret)
        save_path(traj, os.path.join(data_dir, filename))

 
def sac_main():
    import gym
    import os
    import logging
    import argparse
    import numpy as np
    import torch
    from code.envs.env_utils import save_path
    from code.experiments.utils import seed_torch
    from code.envs.nonstationary_env import NonstationaryEnv
    from stable_baselines3 import SAC
    
    def set_env_tasks(group):
        
        """
        tasks = [  # for gravity
                 {'gravity': np.array([ 0.        ,  0.        , -2.90747556])},
                 {'gravity': np.array([ 0.        ,  0.        , -3.03890879])},
                 {'gravity': np.array([ 0.       ,  0.       , -3.1069304])},
                 {'gravity': np.array([ 0.        ,  0.        , -3.57477312])}, 
                 {'gravity': np.array([ 0.        ,  0.        , -3.73644346])}, 
                 {'gravity': np.array([ 0.        ,  0.        , -5.93762192])},
                 {'gravity': np.array([ 0.        ,  0.        , -6.26256863])}, 
                 {'gravity': np.array([ 0.        ,  0.        , -7.63130717])}, 
                 {'gravity': np.array([  0.        ,   0.        , -11.31560039])}, 
                 {'gravity': np.array([  0.        ,   0.        , -15.39438053])}, 
                 {'gravity': np.array([  0.        ,   0.        , -15.66271324])}, 
                 {'gravity': np.array([  0.        ,   0.        , -17.94187483])}, 
                 {'gravity': np.array([  0.        ,   0.        , -24.61469986])},
                 {'gravity': np.array([ 0.        ,  0.        , -3.63877621])}, 
                 {'gravity': np.array([  0.        ,   0.        , -20.39012771])},   
                 {'gravity': np.array([  0.        ,   0.        , -29.88484933])},                 
                 ]
        """

        """
        tasks = [  # for dof_damping (HalfCheetah)
            {'dof_damping': np.array([ 0.        ,  0.        ,  0.        ,  3.70942174,  1.90543138, 1.1127756 ,  2.09764266,  2.06038708,  1.16686654])},
            {'dof_damping': np.array([ 0.        ,  0.        ,  0.        , 11.02657446, 14.78146576, 5.48681188,  2.63779567,  6.06399203,  0.57132163])},
            {'dof_damping': np.array([ 0.        ,  0.        ,  0.        ,  2.65816568, 12.73197153, 2.07146999,  8.283377  ,  5.19874928,  3.81129331])},
            {'dof_damping': np.array([ 0.        ,  0.        ,  0.        ,  2.48180648, 12.89051274, 4.842493  ,  1.56556483,  5.58508926,  2.78178477])},    
            {'dof_damping': np.array([ 0.        ,  0.        ,  0.        ,  2.92341855, 11.29045818, 0.95013162,  6.81270439,  2.45329541,  1.73021413])},
            {'dof_damping': np.array([ 0.        ,  0.        ,  0.        , 18.74527746,  2.85815835, 4.78982056, 11.2430865 ,  7.83505339,  0.54660139])},
            {'dof_damping': np.array([ 0.        ,  0.        ,  0.        ,  2.25832052,  3.71414597, 9.13909765,  4.87812825,  4.78463211,  0.9575793 ])},
            {'dof_damping': np.array([ 0.        ,  0.        ,  0.        ,  3.58035264,  1.82944816, 0.93177186,  6.95281636,  1.48745092,  0.84796935])},
            {'dof_damping': np.array([ 0.        ,  0.        ,  0.        ,  2.54040635,  5.59201156, 4.87725499,  1.71025129,  2.43398176,  2.40704534])},
            {'dof_damping': np.array([ 0.        ,  0.        ,  0.        ,  8.93733593,  4.66598748, 8.84823721,  5.55471818,  8.0045065 ,  0.62096346])},   
            {'dof_damping': np.array([ 0.        ,  0.        ,  0.        ,  3.4282235 , 11.78922155, 2.51852448, 13.94241896,  4.46483043,  2.01682666])},
            {'dof_damping': np.array([ 0.        ,  0.        ,  0.        ,  7.2606183 ,  3.59877486, 1.58225965, 12.00610562,  3.58894775,  0.44755882])},
            {'dof_damping': np.array([ 0.        ,  0.        ,  0.        , 15.34325248,  3.17988364, 8.10509425,  6.07502882,  0.92376898,  4.26395014])},
            {'dof_damping': np.array([ 0.        ,  0.        ,  0.        ,  1.86587171,  1.42112373, 0.952258  ,  2.42701487,  7.20290859,  1.64861074])},
            {'dof_damping': np.array([ 0.        ,  0.        ,  0.        ,  3.50629382,  5.54397499, 9.40310951,  5.22028054,  0.93014194,  3.11690793])},
            {'dof_damping': np.array([ 0.        ,  0.        ,  0.        , 14.52949459,  8.20934072, 3.43986767,  1.8582758 ,  1.02837939,  0.59706637])},
        ]
        """

        """
        tasks = [  # for dof_damping (Hopper)
            {'dof_damping': np.array([0.        , 0.        , 0.        , 0.61823696, 0.4234292 , 0.3709252 ])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 1.09903551, 0.82153328, 1.56925388])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 1.51393431, 0.81776514, 1.15347608])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 3.12421291, 0.6351463 , 1.59660685])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 0.32582879, 0.44787842, 2.50914372])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 1.0840285 , 1.59487737, 0.6383862 ])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 1.83776241, 3.28477017, 1.82893729])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 0.88094142, 2.70209479, 0.60526217])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 1.5450703 , 0.49581697, 0.5653129 ])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 0.42340106, 1.24266924, 1.62575166])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 0.81157062, 0.33458343, 1.09125503])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 1.23438182, 2.66816883, 0.41397564])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 0.44302761, 2.82932701, 0.69049   ])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 1.35103179, 1.84133418, 0.69239498])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 3.09831532, 1.48827681, 1.34455111])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 1.21010305, 0.79972775, 0.52741988])}
        ]
        """

        """
        tasks = [  # for dof_damping (Walker2d)
            {'dof_damping': np.array([0.        , 0.        , 0.        , 0.0618237 , 0.04234292, 0.03709252, 0.04661428, 0.06867957, 0.0777911 ])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 0.04872364, 0.25089907, 0.03167105, 0.15139343, 0.08177651, 0.11534761])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 0.31242129, 0.06351463, 0.15966069, 0.24984637, 0.26116845, 0.03644009])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 0.03763868, 0.08253658, 0.30463659, 0.10840285, 0.15948774, 0.06383862])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 0.18377624, 0.32847702, 0.18289373, 0.05861768, 0.20213307, 0.03808811])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 0.05967254, 0.0406544 , 0.03105906, 0.15450703, 0.0495817 , 0.05653129])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 0.04234011, 0.12426692, 0.16257517, 0.03800558, 0.08113273, 0.16046969])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 0.1489556 , 0.10368861, 0.29494124, 0.12343818, 0.26681688, 0.04139756])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 0.04430276, 0.2829327 , 0.069049  , 0.18407504, 0.17329164, 0.25408622])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 0.05713706, 0.2619827 , 0.08395082, 0.30983153, 0.14882768, 0.13445511])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 0.1210103 , 0.07997277, 0.05274199, 0.26680235, 0.11963159, 0.02983725])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 0.25572087, 0.07066408, 0.27016981, 0.13500064, 0.0307923 , 0.28426334])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 0.04136344, 0.28645584, 0.16141643, 0.03479033, 0.18616964, 0.18545232])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 0.03109786, 0.03158053, 0.03174193, 0.05393366, 0.24009695, 0.10990738])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 0.05843823, 0.12319944, 0.31343698, 0.11600623, 0.03100473, 0.20779386])},
            {'dof_damping': np.array([0.        , 0.        , 0.        , 0.24215824, 0.18242979, 0.11466226, 0.04129502, 0.03427931, 0.03980442])}
        ]
        """

        """
        tasks = [  # for body_mass (HalfCheetah)
            {'body_mass': np.array([0.        , 10.87080773, 0.4550149 , 0.9773955 ,  
                                    0.45272652,  0.52878632, 0.549513  , 0.58368668])}, 
            {'body_mass': np.array([0.        ,  6.9902102 , 1.26125736, 2.48089614,
                                    0.5209486 ,  3.57678571, 0.37335459, 1.28664652])}, 
            {'body_mass': np.array([0.        ,  7.3364693 , 0.6400741 , 0.75848193, 
                                    2.22231732,  4.45383877, 0.74874296, 1.35690739])}, 
            {'body_mass': np.array([0.        , 16.61113146, 0.55944581, 0.51511575,  
                                    0.4788674 ,  3.57700384, 0.44370397, 0.70145316])}, 
            {'body_mass': np.array([0.        ,  6.89476091, 2.44853236, 1.00925025, 
                                    1.6830702 ,  3.21767224, 0.36518055, 1.56185813])}, 
            {'body_mass': np.array([0.        , 11.63261421, 0.89992681, 3.19560241, 
                                    0.40723448,  1.25585905, 3.18536761, 0.51439383])}, 
            {'body_mass': np.array([0.        ,  2.58574744, 0.47683364, 2.44266336, 
                                    0.53012285,  0.80590299, 1.15492989, 0.28672034])}, 
            {'body_mass': np.array([0.        ,  2.69296339, 1.90780551, 2.57021575, 
                                    0.40635213,  1.15661796, 1.89169882, 0.68972908])}, 
            {'body_mass': np.array([0.        ,  6.94072392, 2.28683791, 1.63925467, 
                                    3.15348402,  1.75971925, 3.14537396, 0.35182525])}, 
            {'body_mass': np.array([0.        , 13.43544404, 1.19693141, 0.70040005,  
                                    3.02508984,  0.98435389, 2.16997082, 1.47275273])}, 
            {'body_mass': np.array([0.        ,  8.59298548, 2.82690469, 1.09463489, 
                                    0.61090406,  3.73479255, 0.98965518, 2.63316355])}, 
            {'body_mass': np.array([0.        ,  8.55176633, 0.60136667, 4.71869765, 
                                    0.94653117,  1.72510774, 0.94275999, 0.44823805])}, 
            {'body_mass': np.array([0.        ,  7.60894406, 0.45807587, 2.10225089, 
                                    0.70128932,  1.52258673, 3.01456853, 0.60055244])}, 
            {'body_mass': np.array([0.        ,  8.58646371, 0.47273817, 4.49403275, 
                                    1.70116643,  4.78012342, 0.531216  , 0.35153525])}, 
            {'body_mass': np.array([0.        , 10.26659091, 0.53411785, 2.94323024,  
                                    1.98283875,  3.98969234, 1.97217272, 0.34070352])}, 
            {'body_mass': np.array([0.        ,  2.00862048, 0.48731741, 0.85265884, 
                                    2.56709405,  1.56682587, 1.34050338, 1.95309839])}
        ]
        """

        """
        tasks = [  # for body_mass (Hopper)
            {'body_mass': np.array([0.        , 6.04067819, 1.16387664, 1.67810286, 2.15499213])}, 
            {'body_mass': np.array([0.        , 1.6474847 , 2.69704039, 2.11151194, 5.59340946])}, 
            {'body_mass': np.array([0.        , 5.54620101, 1.91337296, 6.81024393, 1.61186032])}, 
            {'body_mass': np.array([0.        , 2.89022056, 4.52968999, 1.1316583 , 2.44171378])}, 
            {'body_mass': np.array([0.        , 11.04187987,2.49421368, 4.33372755,12.71563127])}, 
            {'body_mass': np.array([0.        , 1.28789919, 1.27952667, 1.21569254,12.7699861 ])}, 
            {'body_mass': np.array([0.        , 2.91708342,11.96305086, 2.94241764, 8.11693714])}, 
            {'body_mass': np.array([0.        , 5.56351796, 8.86355249, 0.84083789, 9.35307144])}, 
            {'body_mass': np.array([0.        , 6.46399795, 2.30191097, 5.48657073, 1.93844862])}, 
            {'body_mass': np.array([0.        , 9.54999127, 2.376859  , 1.61971337, 2.06905712])}, 
            {'body_mass': np.array([0.        , 5.4607292 , 1.94706871, 1.53444919, 4.98610626])}, 
            {'body_mass': np.array([0.        , 4.23263842, 1.66269207, 3.37302191, 8.27406816])}, 
            {'body_mass': np.array([0.        , 2.8674672 , 6.30162996, 2.2028754 , 1.70282224])}, 
            {'body_mass': np.array([0.        , 5.26452542, 4.07184222, 8.00569642, 6.28223826])}, 
            {'body_mass': np.array([0.        , 1.46311068, 1.63281955, 5.73372856, 3.96785323])}, 
            {'body_mass': np.array([0.        , 9.99966706, 2.71154788, 4.9964153 , 8.81947038])}
        ]
        """

        tasks = [
        # {'body_mass': np.array([0.        , 0.55237808, 0.04683739, 0.04068548, 0.05390866,
        # 0.05201904, 0.03133835, 0.16836338, 0.1126938 , 0.02747063,
        # 0.13199036, 0.03913336, 0.0430438 , 0.18280325])}, 
        # {'body_mass': np.array([0.        , 0.11985637, 0.01135288, 0.08193075, 0.12770464,
        # 0.08973331, 0.11686953, 0.13439901, 0.03321386, 0.07217897,
        # 0.02564541, 0.0512686 , 0.01531807, 0.19148497])}, 
        # {'body_mass': np.array([0.        , 0.26589791, 0.02057122, 0.07108193, 0.05834338,
        # 0.04308459, 0.0113135 , 0.08641882, 0.04791293, 0.04848022,
        # 0.19105648, 0.05677015, 0.02591682, 0.05569191])}, 
        # {'body_mass': np.array([0.        , 0.11226282, 0.05472871, 0.05524656, 0.03208679,
        # 0.01478976, 0.02328142, 0.04659344, 0.04326976, 0.03141578,
        # 0.21296637, 0.01385351, 0.01796523, 0.02847595])}, 
        # {'body_mass': np.array([0.        , 0.17956409, 0.03360655, 0.01958807, 0.02831431,
        # 0.01413712, 0.05335657, 0.02691807, 0.01743585, 0.02650453,
        # 0.14173119, 0.01368789, 0.08299905, 0.02429851])}, 
        {'body_mass': np.array([0.        , 0.30321951, 0.11634267, 0.04707524, 0.1161754 ,
        0.01188908, 0.0215052 , 0.02576561, 0.0222142 , 0.01442733,
        0.04168803, 0.02960963, 0.01263337, 0.10367565])}, 
        # {'body_mass': np.array([0.        , 0.18492749, 0.03859943, 0.01358304, 0.07808398,
        # 0.10365496, 0.02345998, 0.09754339, 0.01489345, 0.06174165,
        # 0.03888825, 0.01687698, 0.04502184, 0.02019722])}, 
        {'body_mass': np.array([0.        , 0.0980767 , 0.05621988, 0.0208459 , 0.1150309 ,
        0.11229035, 0.01979539, 0.07812404, 0.04563152, 0.04348664,
        0.03309355, 0.10974105, 0.03207404, 0.15077109])}, 
        # {'body_mass': np.array([0.        , 0.19992226, 0.07826374, 0.02835774, 0.16404934,
        # 0.04445155, 0.09232936, 0.10369065, 0.0630972 , 0.03659464,
        # 0.19687695, 0.05177865, 0.03030871, 0.08408728])}, 
        {'body_mass': np.array([0.        , 0.20194495, 0.05385788, 0.02188897, 0.08649873,
        0.03067319, 0.01502724, 0.03973713, 0.04324536, 0.04550191,
        0.07777661, 0.05295198, 0.05281078, 0.05493652])}, 
        # {'body_mass': np.array([0.        , 0.23711087, 0.03120732, 0.09464636, 0.13671915,
        # 0.0599013 , 0.01379237, 0.18010433, 0.06142913, 0.12276479,
        # 0.027666  , 0.08932251, 0.01604819, 0.08598348])}, 
        # {'body_mass': np.array([0.        , 0.76307264, 0.07703984, 0.04315454, 0.05179115,
        # 0.01278859, 0.05896729, 0.05797443, 0.0626081 , 0.0889444 ,
        # 0.20641056, 0.08668449, 0.01112041, 0.04617225])}, 
        # {'body_mass': np.array([0.        , 0.14721084, 0.03839233, 0.01233545, 0.03128621,
        # 0.01130612, 0.07452875, 0.0331615 , 0.02503946, 0.10334905,
        # 0.10673193, 0.01167841, 0.01613436, 0.08723054])}, 
        # {'body_mass': np.array([0.        , 0.1729617 , 0.10490254, 0.04813142, 0.07078946,
        # 0.04539546, 0.06384886, 0.04108013, 0.02847633, 0.01800754,
        # 0.03025303, 0.10752732, 0.06533036, 0.06342203])}, 
        # {'body_mass': np.array([0.        , 0.18002988, 0.01244672, 0.03109756, 0.04106523,
        # 0.0588118 , 0.02709301, 0.02977192, 0.01147674, 0.01272908,
        # 0.1004287 , 0.03259093, 0.03987181, 0.17038169])}, 
        {'body_mass': np.array([0.        , 0.16434891, 0.05423981, 0.0205096 , 0.02022394,
        0.06839236, 0.02354278, 0.04888717, 0.04521989, 0.08161814,
        0.08883749, 0.09031115, 0.0210259 , 0.13403601])}
    ]

        if group == 0:
            return tasks[0: 2].copy()
        elif group == 1:
            return tasks[2: 4].copy()
        elif group == 2:
            return tasks[8: 12].copy()
        elif group == 3:
            return tasks[12: 16].copy()
    
    seed_torch(seed=0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_tasks', type=int, default=int(0))
    parser.add_argument('--total_steps', type=int, default=int(1e6))
    parser.add_argument('--gpu', type=int, default=int(0))
    args = parser.parse_args()

    varing_params = "body_mass"
    varing_params_sn = "mass"
    env_name = "Ant"
    log_dir = "expert_data/sac_{0}_{1}".format(env_name, varing_params_sn)
    model_dir = "expert_data/expert_models/sac_{0}_{1}".format(env_name, varing_params_sn)
    data_dir = "expert_data/sac_{0}_{1}".format(env_name, varing_params_sn)

    logging.basicConfig(filename=os.path.join(log_dir, 'env_tasks_{}.log'.format(args.env_tasks)),
                        level=logging.INFO)
    env_tasks = set_env_tasks(args.env_tasks)
    
    torch.cuda.set_device(args.gpu)
    for i in range(len(env_tasks)):
        env = NonstationaryEnv(gym.make(env_name + "-v2"), rand_params=[varing_params], log_scale_limit=3.0)
        env.set_task(env_tasks[i])
        
        model = SAC("MlpPolicy", env, verbose=1)
        model_save_name = "sac_{0}_{1}_{2}".format(env_name, varing_params_sn, str(env_tasks[i][varing_params][-1]).replace(".", "d"))
        model_save_path = os.path.join(model_dir, model_save_name)
        # model = SAC.load(model_save_path, env)
        model.learn(total_timesteps=args.total_steps)
        model.save(model_save_path)
        logging.info("Save model to {}".format(model_save_path))

        traj = {'obs': [], 'act': []}
        ret = 0
        obs = env.reset()
        env.set_task(env_tasks[i])
        for _ in range(1000 * 4):  # sample 1 trajectories
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            traj['obs'].append(np.array(obs))
            traj['act'].append(np.array(action))
            ret += reward
            if done:
                traj['obs'] = np.array(traj['obs'])
                traj['act'] = np.array(traj['act'])
                filename = "{0}_{1}_return_{2}.pkl".format(
                    varing_params_sn, str(env_tasks[i][varing_params][-1]).replace(".", "d"), int(ret)
                )
                logging.info("Save trajectory to {}".format(os.path.join(data_dir, filename)))
                save_path(traj, os.path.join(data_dir, filename))
                
                traj = {'obs': [], 'act': []}
                ret = 0
                obs = env.reset()
                env.set_task(env_tasks[i])
                
        
    
if __name__ == '__main__':
    sac_main()
    # escp_main()