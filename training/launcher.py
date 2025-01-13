"""
This script launches experiments for training agents with different parameters.

The launcher is configured to run experiments with specified settings and parameters.
It uses the `experiment_launcher` module to manage and execute the experiments.

Main steps:
1. Determine if the script is running locally.
2. Set up the launcher with experiment configurations.
3. Define default parameters for the experiments.
4. Add experiments with varying reward ratios.
5. Run the experiments.

Parameters:
- LOCAL (bool): Indicates if the script is running locally.
- TEST (bool): Indicates if the script is in test mode.
- USE_CUDA (bool): Indicates if CUDA should be used for training.
- N_SEEDS (int): Number of seeds for experiment reproducibility.
- default_params (dict): Default parameters for the experiments.
- reward_ratios (list): List of reward ratios to use in the experiments.

Usage:
Run this script as the main module to launch the experiments.
"""
from experiment_launcher import Launcher
from experiment_launcher.utils import is_local


if __name__ == '__main__':
    LOCAL = is_local()
    TEST = False
    USE_CUDA = False

    N_SEEDS = 1

    launcher = Launcher(
        exp_name='13_speeds_reward_ratio',
        exp_file='experiment',
        n_seeds=N_SEEDS,
        n_exps_in_parallel=1,
        use_timestamp=True,)

    default_params = dict(
        n_epochs=4000,
        n_steps_per_epoch=5000,
        n_epochs_save=100,
        n_eval_episodes=5,
        n_steps_per_fit=1000,
        use_cuda=USE_CUDA,
        env_id="HumanoidTorque.walk",
        curriculum='progression',)

    reward_ratios = [0.3]

    for reward_ratio in reward_ratios:
        launcher.add_experiment(reward_ratio__=reward_ratio, **default_params)

    launcher.run(LOCAL, TEST)
