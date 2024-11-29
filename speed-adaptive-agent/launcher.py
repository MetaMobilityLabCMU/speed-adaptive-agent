from experiment_launcher import Launcher
from experiment_launcher.utils import is_local


if __name__ == '__main__':
    LOCAL = is_local()
    TEST = False
    USE_CUDA = False

    N_SEEDS = 1

    launcher = Launcher(exp_name='13_speeds_reward_ratio_065_185',
                        exp_file='experiment',
                        n_seeds=N_SEEDS,
                        n_exps_in_parallel=2, 
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

    for reward_ratio in reward_ratios:
        launcher.add_experiment(reward_ratio__=reward_ratio, **default_params)

    launcher.run(LOCAL, TEST)
