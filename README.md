# GMAIL

Author's official implementation of TPAMI paper "Generalizable Multi-modal Adversarial Imitation Learning for Non-stationary Dynamics"

## Run Code

1. Code of the discriminator is in the file `GMAIL/algps/gaifo.py`, code of the generator is in the directory `GMAIL/algos/escp`, and code of the non-stationary environment is in the directory `GMAIL/envs`.

2. Get demonstrations with the following command. The expert trajectories will be stored in the directory `expert_data/sac_HalfCheetah_gravity` if you set the environment name as `HalfCheetah` and the changing parameter as `gravity`.

    ```bash
    python generate_expert_data.py
    ```

3. Run GMAIL in HalfCheetah with `gravity` as the changing parameter with the following command. Demonstrations are in the directory `expert_data/sac_HalfCheetah_gravity`, so please collect demonstrations before run the code.

    ```bash
    python run_gmail.py --env_name HalfCheetah-v3 --varying_params gravity --expert-path-dir expert_data/sac_HalfCheetah_gravity --H_step 4 --use_rmdm --stop_pg_for_ep --bottle_neck --rbf_radius 3000 --name_suffix GMAIL --rnn_fix_length 16 --autoalpha
    ```

4. Modify the parameters `env_name` and `varying_params` to run GMAIL in other tasks and with other varying params. For example, run GMAIL in Hopper with `body mass` as the changing parameter with the following command. Here, demonstrations are in the directory `expert_data/sac_Hopper_mass`.

    ```bash
    python run_gmail.py --env_name Hopper-v3 --varying_params body_mass --expert-path-dir expert_data/sac_Hopper_mass --use_absorbing_state --H_step 4 --use_rmdm --stop_pg_for_ep --bottle_neck --rbf_radius 3000 --name_suffix GMAIL --rnn_fix_length 2
    ```
