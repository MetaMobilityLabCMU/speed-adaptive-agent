"""
This module provides utility functions for data interpolation and metric calculation
used in training and evaluating speed-adaptive agents.

Functions:
- interpolate_data: Interpolates the given data to a specified mean length.
- calculate_metrics: Calculates evaluation metrics such as RMSE and R2 score.

Usage:
Import the required functions and use them to preprocess data and evaluate model performance.

Example:
    from utils import interpolate_data, calculate_metrics

    interpolated_data = interpolate_data(data, mean_length=100)
    metrics = calculate_metrics(eval_data, ground_truth)
"""
import pickle
from copy import deepcopy
import yaml
import numpy as np
import torch.nn.functional as F
from torch import optim
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.core import Core, Agent
from imitation_lib.imitation import VAIL_TRPO
from imitation_lib.utils import FullyConnectedNetwork, NormcInitializer, Standardizer, VariationalNet, VDBLoss
import mujoco
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from tqdm import tqdm
from sklearn.metrics import root_mean_squared_error, r2_score
from speed_vail import SpeedVAIL

def get_agent(env_id, mdp, use_cuda, sw, conf_path=None):

    if conf_path is None:
        conf_path = 'confs.yaml'    # use default one

    with open(conf_path, 'r') as f:
        confs = yaml.safe_load(f)

    # get conf for environment
    try:
        # get the default conf (task agnostic)
        env_id_short = env_id.split('.')[0]
        conf = confs[env_id_short]
    except KeyError:
        # get the conf for the specific environment and task
        env_id_short = ".".join(env_id.split('.')[:2])
        conf = confs[env_id_short]

    if conf["algorithm"] == "VAIL":
        agent = create_vail_agent(mdp, sw, use_cuda, **conf["algorithm_config"])
    elif conf["algorithm"] == "SpeedVAIL":
        agent = create_speed_vail_agent(mdp, sw, use_cuda, **conf["algorithm_config"])
    else:
        raise ValueError(f"Invalid algorithm: {conf['algorithm']}")

    return agent

def create_vail_agent(mdp, sw, use_cuda, std_0, info_constraint, lr_beta, z_dim, disc_only_states,
                      disc_use_next_states, train_disc_n_th_epoch, disc_batch_size, learning_rate_critic,
                      learning_rate_disc, policy_entr_coef, max_kl, n_epochs_cg, use_noisy_targets,
                      last_policy_activation):

    mdp_info = deepcopy(mdp.info)
    expert_data = mdp.create_dataset()

    trpo_standardizer = Standardizer(use_cuda=use_cuda)
    policy_params = dict(network=FullyConnectedNetwork,
                         input_shape=mdp_info.observation_space.shape,
                         output_shape=mdp_info.action_space.shape,
                         std_0=std_0,
                         n_features=[512, 256],
                         initializers=[NormcInitializer(1.0), NormcInitializer(1.0), NormcInitializer(0.001)],
                         activations=['relu', 'relu', last_policy_activation],
                         standardizer=trpo_standardizer,
                         use_cuda=use_cuda)

    critic_params = dict(network=FullyConnectedNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': learning_rate_critic,
                                               'weight_decay': 0.0}},
                         loss=F.mse_loss,
                         batch_size=256,
                         input_shape=mdp_info.observation_space.shape,
                         activations=['relu', 'relu', 'identity'],
                         standardizer=trpo_standardizer,
                         squeeze_out=False,
                         output_shape=(1,),
                         initializers=[NormcInitializer(1.0), NormcInitializer(1.0), NormcInitializer(0.001)],
                         n_features=[512, 256],
                         use_cuda=use_cuda)

    discrim_obs_mask = mdp.get_kinematic_obs_mask()
    discrim_act_mask = [] if disc_only_states else np.arange(mdp_info.action_space.shape[0])
    discrim_input_shape = (len(discrim_obs_mask) + len(discrim_act_mask),) if not disc_use_next_states else \
        (2 * len(discrim_obs_mask) + len(discrim_act_mask),)
    discrim_standardizer = Standardizer()
    z_size = z_dim
    encoder_net = FullyConnectedNetwork(input_shape=discrim_input_shape, output_shape=(128,), n_features=[256],
                                        activations=['relu', 'relu'], standardizer=None,
                                        squeeze_out=False, use_cuda=use_cuda)
    decoder_net = FullyConnectedNetwork(input_shape=(z_size,), output_shape=(1,), n_features=[],
                                        # no features mean no hidden layer -> one layer
                                        activations=['identity'], standardizer=None,
                                        initializers=[NormcInitializer(std=0.1)],
                                        squeeze_out=False, use_cuda=use_cuda)

    discriminator_params = dict(optimizer={'class': optim.Adam,
                                           'params': {'lr': learning_rate_disc,
                                                      'weight_decay': 0.0}},
                                batch_size=disc_batch_size,
                                network=VariationalNet,
                                input_shape=discrim_input_shape,
                                output_shape=(1,),
                                z_size=z_size,
                                encoder_net=encoder_net,
                                decoder_net=decoder_net,
                                use_next_states=disc_use_next_states,
                                use_actions=not disc_only_states,
                                standardizer=discrim_standardizer,
                                use_cuda=use_cuda)

    alg_params = dict(train_D_n_th_epoch=train_disc_n_th_epoch,
                      state_mask=discrim_obs_mask,
                      act_mask=discrim_act_mask,
                      n_epochs_cg=n_epochs_cg,
                      trpo_standardizer=trpo_standardizer,
                      D_standardizer=discrim_standardizer,
                      loss=VDBLoss(info_constraint=info_constraint, lr_beta=lr_beta),
                      ent_coeff=policy_entr_coef,
                      use_noisy_targets=use_noisy_targets,
                      max_kl=max_kl,
                      use_next_states=disc_use_next_states)

    agent = VAIL_TRPO(mdp_info=mdp_info, policy_class=GaussianTorchPolicy, policy_params=policy_params, sw=sw,
                      discriminator_params=discriminator_params, critic_params=critic_params,
                      demonstrations=expert_data, **alg_params)
    return agent
def create_speed_vail_agent(mdp, sw, use_cuda, std_0, info_constraint, lr_beta, z_dim, disc_only_states,
                      disc_use_next_states, train_disc_n_th_epoch, disc_batch_size, learning_rate_critic,
                      learning_rate_disc, policy_entr_coef, max_kl, n_epochs_cg, use_noisy_targets,
                      last_policy_activation):
    mdp_info = deepcopy(mdp.info)

    with open('locomujoco_13_speeds_dataset.pkl', 'rb') as f:
        expert_data = pickle.load(f)

    trpo_standardizer = Standardizer(use_cuda=use_cuda)
    policy_params = dict(network=FullyConnectedNetwork,
                         input_shape=mdp_info.observation_space.shape,
                         output_shape=mdp_info.action_space.shape,
                         std_0=std_0,
                         n_features=[512, 256],
                         initializers=[NormcInitializer(1.0), NormcInitializer(1.0), NormcInitializer(0.001)],
                         activations=['relu', 'relu', last_policy_activation],
                         standardizer=trpo_standardizer,
                         use_cuda=use_cuda)

    critic_params = dict(network=FullyConnectedNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': learning_rate_critic,
                                               'weight_decay': 0.0}},
                         loss=F.mse_loss,
                         batch_size=256,
                         input_shape=mdp_info.observation_space.shape,
                         activations=['relu', 'relu', 'identity'],
                         standardizer=trpo_standardizer,
                         squeeze_out=False,
                         output_shape=(1,),
                         initializers=[NormcInitializer(1.0), NormcInitializer(1.0), NormcInitializer(0.001)],
                         n_features=[512, 256],
                         use_cuda=use_cuda)

    discrim_obs_mask = mdp.get_kinematic_obs_mask()
    discrim_act_mask = [] if disc_only_states else np.arange(mdp_info.action_space.shape[0])
    discrim_input_shape = (len(discrim_obs_mask) + len(discrim_act_mask),) if not disc_use_next_states else \
        (2 * len(discrim_obs_mask) + len(discrim_act_mask),)
    discrim_standardizer = Standardizer()
    z_size = z_dim
    encoder_net = FullyConnectedNetwork(input_shape=discrim_input_shape, output_shape=(128,), n_features=[256],
                                        activations=['relu', 'relu'], standardizer=None,
                                        squeeze_out=False, use_cuda=use_cuda)
    decoder_net = FullyConnectedNetwork(input_shape=(z_size,), output_shape=(1,), n_features=[],
                                        # no features mean no hidden layer -> one layer
                                        activations=['identity'], standardizer=None,
                                        initializers=[NormcInitializer(std=0.1)],
                                        squeeze_out=False, use_cuda=use_cuda)

    discriminator_params = dict(optimizer={'class': optim.Adam,
                                           'params': {'lr': learning_rate_disc,
                                                      'weight_decay': 0.0}},
                                batch_size=disc_batch_size,
                                network=VariationalNet,
                                input_shape=discrim_input_shape,
                                output_shape=(1,),
                                z_size=z_size,
                                encoder_net=encoder_net,
                                decoder_net=decoder_net,
                                use_next_states=disc_use_next_states,
                                use_actions=not disc_only_states,
                                standardizer=discrim_standardizer,
                                use_cuda=use_cuda)

    alg_params = dict(train_D_n_th_epoch=train_disc_n_th_epoch,
                      state_mask=discrim_obs_mask,
                      act_mask=discrim_act_mask,
                      n_epochs_cg=n_epochs_cg,
                      trpo_standardizer=trpo_standardizer,
                      D_standardizer=discrim_standardizer,
                      loss=VDBLoss(info_constraint=info_constraint, lr_beta=lr_beta),
                      ent_coeff=policy_entr_coef,
                      use_noisy_targets=use_noisy_targets,
                      max_kl=max_kl,
                      use_next_states=disc_use_next_states)

    agent = SpeedVAIL(mdp_info=mdp_info, policy_class=GaussianTorchPolicy, policy_params=policy_params, sw=sw,
                      discriminator_params=discriminator_params, critic_params=critic_params,
                      demonstration_collections=expert_data, **alg_params)
    return agent

class ProgressionCurriculum():
    def __init__(self, speed_range):
        self.speed_range = speed_range
        self.idx = round(len(self.speed_range)/2)
        self.direction = 1
    def get_target_speed(self):
        target_speed = self.speed_range[self.idx]
        if self.idx == len(self.speed_range)-1 and self.direction == 1:
            self.direction = -1
        if self.idx == 0 and self.direction == -1:
            self.direction = 1
        self.idx = self.idx + self.direction
        return target_speed

class RandomCurriculum():
    def __init__(self, speed_range):
        self.speed_range = speed_range
        self.rng = np.random.default_rng()
    def get_target_speed(self):
        target_speed = self.rng.choice(self.speed_range, 1)[0]
        return target_speed

def compute_mean_speed(env, dataset):
    x_vel_idx = env.get_obs_idx("dq_pelvis_tx")
    speeds = []
    for i in range(len(dataset)):
        speeds.append(dataset[i][0][x_vel_idx])
    return np.mean(speeds)

def process_data(
        agent,
        env,
        n_trials=5,
        n_episodes=1,
        cycle_length_cutoff=60,
        record=False,
        is_wrapped=False,
        min_peak=True):
    core = Core(agent, env)
    datasets = []
    for _ in range(n_trials):
        dataset = core.evaluate(n_episodes=n_episodes, render=record, record=record)
        datasets.append(dataset)

    if is_wrapped:
        motor_indices = env.env._action_indices
        motor_names = [mujoco.mj_id2name(env.env._model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx) for idx in motor_indices]
    else:
        motor_indices = env._action_indices
        motor_names = [mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx) for idx in motor_indices]
    obs_keys = dict(
        q_hip_flexion_l=9,
        q_knee_angle_l=12,
        q_ankle_angle_l=13,
        dq_hip_flexion_l=28,
        dq_knee_angle_l=31,
        dq_ankle_angle_l=32,
    )
    act_keys = dict(
        mot_hip_flexion_l=motor_names.index('mot_hip_flexion_l'),
        mot_knee_angle_l=motor_names.index('mot_knee_angle_l'),
        mot_ankle_angle_l=motor_names.index('mot_ankle_angle_l'),
    )

    datas = []
    for dataset in datasets:
        data = {}
        for i in range(len(dataset)):
            for key, value in obs_keys.items():
                if key not in data:
                    data[key] = []
                data[key].append(dataset[i][0][value])
            for key, value in act_keys.items():
                if key not in data:
                    data[key] = []
                data[key].append(dataset[i][1][value])
        for key in obs_keys.keys():
            data[key] = np.array(data[key])
        for key in act_keys.keys():
            data[key] = np.array(data[key])
        datas.append(data)
    cycless = []
    for data in datas:
        if min_peak:
            heel_strikes, _ = find_peaks(-1*data['q_hip_flexion_l'], height=0.2, distance=40)
        else:
            heel_strikes, _ = find_peaks(data['q_hip_flexion_l'], height=np.deg2rad(8),  distance=40)
        cycles = {}
        cycle_lengths = np.diff(heel_strikes)
        for i, cycle_length in enumerate(cycle_lengths):
            if cycle_length > cycle_length_cutoff:
                for key in data.keys():
                    if key not in cycles:
                        cycles[key] = []
                    cycles[key].append(data[key][heel_strikes[i]:heel_strikes[i+1]])
        print(f'Number of recorded cycle: {len(cycle_lengths)}')
        print(f'Number of effective cycle: {len(cycles[key])}')
        cycless.append(cycles)
    x_vel_idx = env.get_obs_idx("dq_pelvis_tx")
    speedss = []
    for dataset in datasets:
        speeds = []
        for i in range(len(dataset)):
            speeds.append(dataset[i][0][x_vel_idx])
        speeds = np.array(speeds)
        print(f'Average speed: {np.mean(speeds)}')
        speedss.append(speeds)
    interpolated_cycless = []
    mean_lengths = []
    for cycles in cycless:
        cycles_lengths = [len(cycle) for cycle in cycles[key]]
        mean_length = round(np.mean(cycles_lengths))
        mean_lengths.append(mean_length)
        interpolated_cycles = {}
        for key, cycle in cycles.items():
            for i in range(len(cycle)):
                cycle_len = len(cycle[i])
                x = np.linspace(0, cycle_len-1, num=cycle_len)
                xnew = np.linspace(0, cycle_len-1, num=mean_length)
                y = np.array([cycle[i][j] for j in range(cycle_len)])
                spl = CubicSpline(x, y)
                if key not in interpolated_cycles:
                    interpolated_cycles[key] = []
                interpolated_cycles[key].append(np.array(spl(xnew)))
        for key in interpolated_cycles.keys():
            interpolated_cycles[key] = np.array(interpolated_cycles[key])
        interpolated_cycless.append(interpolated_cycles)

    return interpolated_cycless, mean_lengths, speedss

def interpolate_data(data, mean_length):
    """
    Interpolates the given data to a specified mean length.

    Args:
        data (dict): A dictionary containing the data to be interpolated. The keys are speeds,
                     and the values are dictionaries with joint names as keys and lists of cycles as values.
        mean_length (int): The target length for interpolation.

    Returns:
        dict: A dictionary with the same structure as the input data, but with interpolated cycles.
    """
    interpolated_data = dict()
    for speed in tqdm(data, desc='Interpolating Data'):
        interpolated_cycles = {}
        for joint in data[speed]:
            interpolated_cycles[joint] = []
            for joint_cycle in data[speed][joint]:
                x = np.linspace(0, len(joint_cycle)-1, num=len(joint_cycle))
                xnew = np.linspace(0, len(joint_cycle)-1, num=mean_length)
                spl = CubicSpline(x, joint_cycle)
                interpolated_cycles[joint].append(spl(xnew))
        for joint in interpolated_cycles:
            interpolated_cycles[joint] = np.array(interpolated_cycles[joint])
        interpolated_data[speed] = {
            'cycles': interpolated_cycles,
            'mean_length': mean_length,}

    return interpolated_data

def calculate_metrics(eval_data, ground_truth):
    """
    Calculates evaluation metrics such as RMSE and R2 score.

    Args:
        eval_data (dict): A dictionary containing the evaluation data.
        ground_truth (dict): A dictionary containing the ground truth data.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    metric = {}
    for i, eval_data_ in enumerate(eval_data):
        metric[i] = {}
        for speed in eval_data_:
            metric[i][speed] = {}
            RMSE, R2 = 0, 0
            for data_joint, eval_joint in zip(['q_hip_flexion_r', 'q_knee_angle_r', 'q_ankle_angle_r'], ['Hip', 'Knee', 'Ankle']):
                avg_data = np.mean(ground_truth[speed]['cycles'][data_joint], axis=0)
                avg_eval = np.mean(eval_data_[speed]['data'][eval_joint]['Angle (deg)'], axis=0)
                avg_eval_interpolated = np.interp(np.linspace(0, 1, num=len(avg_data)), np.linspace(0, 1, num=len(avg_eval)), avg_eval)
                RMSE += root_mean_squared_error(avg_data, avg_eval_interpolated)
                R2 += r2_score(avg_data, avg_eval_interpolated)
            metric[i][speed]['RMSE'] = RMSE/3
            metric[i][speed]['R2'] = R2/3
            metric[i][speed]['actual_speed'] = eval_data_[speed]['mean_speed']

    return metric

def eval_model(mdp, model_file, speed_range, mass, n_trials=5, n_episodes=1, cycle_length_cutoff=40, record=False):
    agent = Agent.load(model_file)
    data_dict = {}
    for target_speed in speed_range:
        data_dict[target_speed] = {}
        mdp.set_operate_speed(target_speed)
        _ = mdp.reset()
        data, mean_length, speeds = process_data(agent, mdp, n_trials=n_trials, n_episodes=n_episodes, cycle_length_cutoff=cycle_length_cutoff, record=record, is_wrapped=True)
        data_dict[target_speed]['data'] = data
        data_dict[target_speed]['mean_length'] = mean_length
        data_dict[target_speed]['speeds'] = speeds
    processed_datas = []
    for i in range(n_trials):
        processed_data = {}
        HIP_GEAR, KNEE_GEAR, ANKLE_GEAR = 275, 600, 500
        for speed in data_dict:
            processed_data[speed] = {
                'data': {
                    'Hip': {},
                    'Knee': {},
                    'Ankle': {}},
                'idxs': None,}
            processed_data[speed]['data']['Hip']['Angle (deg)'] = np.rad2deg(data_dict[speed]['data'][i]['q_hip_flexion_l'])
            processed_data[speed]['data']['Knee']['Angle (deg)'] = -1*np.rad2deg(data_dict[speed]['data'][i]['q_knee_angle_l'])
            processed_data[speed]['data']['Ankle']['Angle (deg)'] = np.rad2deg(data_dict[speed]['data'][i]['q_ankle_angle_l'])
            processed_data[speed]['data']['Hip']['Torque (Nm/kg)'] = -1*data_dict[speed]['data'][i]['mot_hip_flexion_l'] * HIP_GEAR / mass
            processed_data[speed]['data']['Knee']['Torque (Nm/kg)'] = data_dict[speed]['data'][i]['mot_knee_angle_l'] * KNEE_GEAR / mass
            processed_data[speed]['data']['Ankle']['Torque (Nm/kg)'] = -1*data_dict[speed]['data'][i]['mot_ankle_angle_l'] * ANKLE_GEAR / mass
            processed_data[speed]['data']['Hip']['Power (W/kg)'] = np.multiply(data_dict[speed]['data'][i]['dq_hip_flexion_l'], data_dict[speed]['data'][i]['mot_hip_flexion_l'] * HIP_GEAR) / mass
            processed_data[speed]['data']['Knee']['Power (W/kg)'] = np.multiply(data_dict[speed]['data'][i]['dq_knee_angle_l'], data_dict[speed]['data'][i]['mot_knee_angle_l'] * KNEE_GEAR) / mass
            processed_data[speed]['data']['Ankle']['Power (W/kg)'] = np.multiply(data_dict[speed]['data'][i]['dq_ankle_angle_l'], data_dict[speed]['data'][i]['mot_ankle_angle_l'] * ANKLE_GEAR) / mass
            processed_data[speed]['idxs'] = np.linspace(0, 1, data_dict[speed]['mean_length'][i])*100
            processed_data[speed]['mean_speed'] = round(np.mean(data_dict[speed]['speeds'][i]), 2)
        processed_datas.append(processed_data)
    return processed_datas
