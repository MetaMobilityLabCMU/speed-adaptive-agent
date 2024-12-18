from mushroom_rl.core import Core
import mujoco
import numpy as np
from scipy.interpolate import CubicSpline

from dm_control.mujoco import *

from scipy.signal import find_peaks

def process_data(agent, env, n_trials=5, n_episodes=1, cycle_length_cutoff=40, record=False, is_wrapped=False, min_peak=True):
    # Use the Core with the class method as a callback
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

 
    # format: obs(6), act(3), grf(3), exo_torque(1)
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
    

    # interpolate exo cycles as they have different lengths
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

if __name__ == '__main__':
    # load model
    # inference
    # plot biomechanics
    # calculate RMSE
    # add these to training script?
    # generate videos?
    # save ground truth as a separate file for faster loading?
    speed_range = np.round(np.linspace(0.65, 1.85, 13), 2)
    mdp = LocoEnv.make("HumanoidTorque.walk", headless=True, default_camera_mode='static')
    mdp = SpeedDomainRandomizationWrapper(mdp, (speed_range[0], speed_range[-1]))
    _ = mdp.reset()
    import os
    import loco_mujoco
    library_path = os.path.dirname(loco_mujoco.__file__)
    file_path = os.path.join(library_path, 'environments', 'data', 'humanoid', 'humanoid_torque.xml')

    physics = Physics.from_xml_path(file_path)
    mass = np.sum(physics.named.model.body_mass._field)
    agent = Agent.load("../../exo-rl/speed-domain-randomization/logs/55_speeds_reward_ratio_2024-11-22_22-00-50/reward_ratio___0.3/0/agent_epoch_3997_J_981.966267.msh")

    data_dict = {}
    for target_speed in speed_range:
        data_dict[target_speed] = {}

        mdp.set_operate_speed(target_speed)
        _ = mdp.reset()
        
        data, mean_length, speeds = process_data(agent, mdp, 0.05, cycle_length_cutoff=60, is_exo=False, is_bio_torque=False, record=False, is_wrapped=True)
        data_dict[target_speed]['data'] = data
        data_dict[target_speed]['mean_length'] = mean_length
        print(mean_length)
        data_dict[target_speed]['speeds'] = speeds