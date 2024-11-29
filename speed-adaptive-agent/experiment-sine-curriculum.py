import os
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from experiment_launcher import run_experiment
from mushroom_rl.core import Agent
from custom_core import SpeedCore
from mushroom_rl.utils.dataset import compute_J, compute_episodes_length
from mushroom_rl.core.logger.logger import Logger

from imitation_lib.utils import BestAgentSaver

from loco_mujoco import LocoEnv
from utils import get_agent, SineCurriculum, compute_mean_speed
from tqdm import tqdm

from custom_env_wrapper import SpeedDomainRandomizationWrapper

def experiment(fix_training_epoch: int = 0, 
               reward_ratio: float = 0.3, 
               env_id: str = None,
               n_epochs: int = 500,
               n_steps_per_epoch: int = 10000,
               n_steps_per_fit: int = 1024,
               n_eval_episodes: int = 50,
               n_epochs_save: int = 500,
               gamma: float = 0.99,
               results_dir: str = './logs',
               use_cuda: bool = False,
               seed: int = 0):

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    results_dir = os.path.join(results_dir, str(seed))

    # logging
    sw = SummaryWriter(log_dir=results_dir)     # tensorboard
    logger = Logger(results_dir=results_dir, log_name="logging", seed=seed, append=True)    # numpy
    agent_saver = BestAgentSaver(save_path=results_dir, n_epochs_save=n_epochs_save)

    # speed_range = np.round(np.linspace(0.50, 1.85, 55), 3)
    # speed_range = np.round(np.linspace(0.5, 1.85, 28), 2)
    speed_range = np.round(np.linspace(0.55, 1.85, 14), 2)


    rng = np.random.default_rng()
    print(f'speed range: {speed_range}')
    curriculum = SineCurriculum(speed_range)

    print(f"Starting training {env_id}...")
    # create environment, agent and core
    mdp = LocoEnv.make(env_id, headless=True)
    mdp = SpeedDomainRandomizationWrapper(mdp, (speed_range[0], speed_range[-1]))
    _ = mdp.reset()

    agent = get_agent(env_id, mdp, use_cuda, sw)
    agent._env_reward_frac = reward_ratio
    print(f'env_reward_frac = {agent._env_reward_frac}')
    print(f'fix training epoch = {fix_training_epoch}')
    # agent = Agent.load("logs/locomujoco_speed_domain_randomization_large_speed_range/env_id___HumanoidTorque.walk/0/agent_epoch_3914_J_962.670273.msh")
    core = SpeedCore(agent, mdp)

    for epoch in tqdm(range(n_epochs)):
        if epoch < fix_training_epoch:
            target_speed = 1.25
        else:
            target_speed = curriculum.get_target_speed()
            # target_speed = rng.choice(speed_range, 1)[0]

        # update the environment target speed
        mdp.set_operate_speed(target_speed)
        # reward_params = dict(target_velocity=target_speed)
        # mdp._reward_function = mdp._get_reward_function(reward_type="target_velocity", reward_params=reward_params)

        # train
        core.learn(n_steps=n_steps_per_epoch, n_steps_per_fit=n_steps_per_fit, quiet=True, render=False, target_speed=target_speed)
        # evaluate
        dataset = core.evaluate(n_episodes=n_eval_episodes, target_speed=target_speed)
        R_mean = np.mean(compute_J(dataset))
        J_mean = np.mean(compute_J(dataset, gamma=gamma))
        L = np.mean(compute_episodes_length(dataset))
        S_mean = compute_mean_speed(mdp, dataset)
        logger.log_numpy(Epoch=epoch, R_mean=R_mean, J_mean=J_mean, L=L, S_mean=S_mean, target_speed=target_speed)
        sw.add_scalar("Eval_R-stochastic", R_mean, epoch)
        sw.add_scalar("Eval_J-stochastic", J_mean, epoch)
        sw.add_scalar("Eval_L-stochastic", L, epoch)
        sw.add_scalar("Eval_S-stochastic", S_mean, epoch)
        sw.add_scalar("target_speed", target_speed, epoch)
        agent_saver.save(core.agent, R_mean)

    agent_saver.save_curr_best_agent()
    print("Finished.")


if __name__ == "__main__":
    run_experiment(experiment)
