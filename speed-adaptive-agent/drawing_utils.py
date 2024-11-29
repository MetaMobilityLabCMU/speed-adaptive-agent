
from mushroom_rl.core import Core, Agent
from loco_mujoco import LocoEnv
import mujoco
import numpy as np
from numpy import linalg as LA
from operator import itemgetter 
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from dm_control.mujoco import *

CYCLE_LENGTH_CUTOFF = 40
HIP_GEAR = 275


class ExoCallbackHandler:
    def __init__(self, exo_env, is_exo, is_bio_torque=False):
        self.exo_env = exo_env
        self.exo_step = 0
        self.exo_l_foot_contact = False
        self.exo_heel_strikes = []
        self.exo_grfs = []
        self.exo_torques = []
        self.is_exo = is_exo
        self.is_bio_torque = is_bio_torque

    def exo_get_heelstrike_and_grf_and_exotorque(self, sample):
        # Heel strike detection
        contact_group = []
        for coni in range(self.exo_env._data.ncon):
            con = self.exo_env._data.contact[coni]
            geom1 = mujoco.mj_id2name(self.exo_env._model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
            geom2 = mujoco.mj_id2name(self.exo_env._model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
            contact_group.append(set([geom1, geom2]))

        if set(['floor', 'foot_box_l']) in contact_group:
            if not self.exo_l_foot_contact:
                self.exo_l_foot_contact = True
                self.exo_heel_strikes.append(self.exo_step)
        else:
            self.exo_l_foot_contact = False

        # Ground Reaction Forces (GRF)
        exo_l_grf = self.exo_env._get_ground_forces()[-3:]
        self.exo_grfs.append(exo_l_grf)

        # Exo Torque
        if self.is_exo:
            l_exo_torque = self.exo_env._data.actuator('mot_exo_l').ctrl[0]
            self.exo_torques.append(l_exo_torque)
        if self.is_bio_torque:
            l_exo_torque = self.exo_env.exo_torque[1]

        if self.is_bio_torque and not self.is_exo:
            AssertionError("Bio torque is enabled but Exo torque is not enabled")

        # Increment step
        self.exo_step += 1


def plot_gait(data_dict, mean_length, alpha=0.4, ylabel='Angle', title='test', save=False):
    idxs = np.linspace(0, 1, mean_length)*100
    
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")

    for label, data in data_dict.items():   
        ax.plot(idxs,np.mean(data, axis=0), label=label)
        ax.fill_between(idxs, np.mean(data, axis=0) - np.std(data, axis=0), np.mean(data, axis=0) + np.std(data, axis=0), alpha=alpha)
    
    plt.xticks([0, 20, 40, 60, 80, 100])
    plt.xlim(0, 100)
    plt.xlabel("Gait Phase (%)")
    plt.ylabel(ylabel)
    ax.spines[['right', 'top']].set_visible(False)
    plt.axhline(y = 0, color = 'black', alpha=0.1)
    plt.title(title)
    plt.legend()
    if save:
        plt.savefig(f'{title}.png', dpi=300)
    else:
        plt.show()


def smooth_1d(data, window_size=5):
    kernel = np.ones(window_size) / window_size
    length = len(data)
    
    smooth_data = np.convolve(data, kernel, 'valid')
    x = np.linspace(0, len(smooth_data)-1, num=len(smooth_data))
    xnew = np.linspace(0, len(smooth_data)-1, num=len(data))
    spl = CubicSpline(x, smooth_data)
    smooth_data = spl(xnew)
    return smooth_data

def smooth_2d(data, window_size=5):
    smooth_data = []
    for cycle in data:
        smooth_data.append(smooth_1d(cycle, window_size=window_size))
    smooth_data = np.array(smooth_data)
    return smooth_data

def process_data(agent, exo_env, is_exo= False, is_bio_torque=False, record=False):

    # grf
    # Initialize the handler with your environment
    callback_handler = ExoCallbackHandler(exo_env, is_exo=is_exo, is_bio_torque=is_bio_torque)

    # Use the Core with the class method as a callback
    core = Core(agent, exo_env, callback_step=callback_handler.exo_get_heelstrike_and_grf_and_exotorque)
    exo_dataset = core.evaluate(n_episodes=5, render=record, record=record)

    motor_indices = exo_env._action_indices
    motor_names = [mujoco.mj_id2name(exo_env._model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx) for idx in motor_indices]

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

    if is_exo:
        act_keys['mot_exo_l'] = motor_names.index('mot_exo_l')

    processed_exo_dataset = []
    # format: obs(6), act(3), grf(3), exo_torque(1)
    data = {}
    data['grf'] = []

    for i in range(len(exo_dataset)):
        for key, value in obs_keys.items():
            if key not in data:
                data[key] = []
            data[key].append(exo_dataset[i][0][value])

        for key, value in act_keys.items():
            if key not in data:
                data[key] = []
            data[key].append(exo_dataset[i][1][value])
        
        data['grf'].extend(callback_handler.exo_grfs[i])

        # for j in obs_keys.values():
        #     data.append(exo_dataset[i][0][j])
        # for k in act_keys.values():
        #     data.append(exo_dataset[i][1][k])
        # data.extend(callback_handler.exo_grfs[i])
        # if is_exo:
        #     data.append(exo_dataset[i][1][motor_names.index('mot_exo_l')])
        #     # data.append(callback_handler.exo_torques[i])
        #     # data.append(callback_handler.exo_torques[i])
        
        # processed_exo_dataset.append(np.array(data))
    for key in obs_keys.keys():
        data[key] = np.array(data[key])
    for key in act_keys.keys():
        data[key] = np.array(data[key])
    data['grf'] = np.array(data['grf'])


    # processed_exo_dataset = np.array(processed_exo_dataset)

    exo_cycles = {}
    exo_cycle_lengths = np.diff(callback_handler.exo_heel_strikes)
    for i, exo_cycle_length in enumerate(exo_cycle_lengths):
        if exo_cycle_length > CYCLE_LENGTH_CUTOFF:
            for key in data.keys():
                if key not in exo_cycles:
                    exo_cycles[key] = []
                exo_cycles[key].append(data[key][callback_handler.exo_heel_strikes[i]:callback_handler.exo_heel_strikes[i+1]])

            # exo_cycles.append(processed_exo_dataset[callback_handler.exo_heel_strikes[i]:callback_handler.exo_heel_strikes[i+1]])
    
    x_vel_idx = exo_env.get_obs_idx("dq_pelvis_tx")
    speeds = []
    for i in range(len(exo_dataset)):
        speeds.append(exo_dataset[i][0][x_vel_idx])
    speeds = np.array(speeds)

    print(f'Number of recorded exo_cycle: {len(exo_cycle_lengths)}')
    print(f'Cycle length cutff: {CYCLE_LENGTH_CUTOFF}')
    print(f'Number of effective exo_cycle: {len(exo_cycles[key])}')
    print(f'Average speed: {np.mean(speeds)}')

    # interpolate exo cycles as they have different lengths
    exo_cycles_lengths = [len(cycle) for cycle in exo_cycles[key]]
    exo_mean_length = round(np.mean(exo_cycles_lengths))
    exo_interpolated_cycles = {}
    for key, cycle in exo_cycles.items():
        
        for i in range(len(cycle)):
            cycle_len = len(cycle[i])
            x = np.linspace(0, cycle_len-1, num=cycle_len)
            xnew = np.linspace(0, cycle_len-1, num=exo_mean_length)
            y = np.array([cycle[i][j] for j in range(cycle_len)])
            spl = CubicSpline(x, y)
            if key not in exo_interpolated_cycles:
                exo_interpolated_cycles[key] = []
            exo_interpolated_cycles[key].append(np.array(spl(xnew)))

    for key in exo_interpolated_cycles.keys():
        exo_interpolated_cycles[key] = np.array(exo_interpolated_cycles[key])

    return exo_interpolated_cycles, exo_mean_length, speeds
