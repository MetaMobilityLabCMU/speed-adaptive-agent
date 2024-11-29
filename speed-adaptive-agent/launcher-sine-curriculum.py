from experiment_launcher import Launcher
from experiment_launcher.utils import is_local


if __name__ == '__main__':
    LOCAL = is_local()
    TEST = False
    USE_CUDA = False

    N_SEEDS = 1

    launcher = Launcher(exp_name='14_speeds_reward_ratio_055_185',
                        exp_file='experiment-sine-curriculum',
                        n_seeds=N_SEEDS,
                        # n_cores=1,  # only used for slurm
                        # memory_per_core=1500,   # only used for slurm
                        n_exps_in_parallel=2,  # should not be used in slurm
                        # days=2,     # only used for slurm
                        # hours=0,    # only used for slurm
                        # minutes=0,  # only used for slurm
                        use_timestamp=True,
                        )

    default_params = dict(n_epochs=4000,
                          n_steps_per_epoch=5000,
                          n_epochs_save=100,
                          n_eval_episodes=5,
                          n_steps_per_fit=1000,
                          use_cuda=USE_CUDA,
                          env_id="HumanoidTorque.walk",)

    reward_ratios = [0.3]
    # reward_ratios = [0, 0.1, 0.2, 0.6, 0.7, 0.8, 0.9, 1]

    for reward_ratio in reward_ratios:
        launcher.add_experiment(reward_ratio__=reward_ratio, **default_params)

    launcher.run(LOCAL, TEST)
