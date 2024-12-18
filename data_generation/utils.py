import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import CubicSpline, PchipInterpolator
import os
import pickle 
from scipy import stats
from tqdm import tqdm
import copy
from loco_mujoco import LocoEnv
import mujoco
from scipy.signal import find_peaks

def extract_locomujoco_data():
    mdp = LocoEnv.make("HumanoidTorque.walk.real", headless=True, random_start=False, init_step_no=0)
    num_steps = 89800-1
    samples = []
    obss = []
    mdp._init_step_no = 0
    mdp.reset()
    sample = mdp.trajectories.get_current_sample()
    obs = mdp._create_observation(np.concatenate(sample))

    samples.append(sample)
    obss.append(obs)

    for j in tqdm(range(num_steps), desc='Loading LocoMuJoCo Data'):

        mdp.set_sim_state(sample)

        mdp._simulation_pre_step()
        mujoco.mj_forward(mdp._model, mdp._data)
        mdp._simulation_post_step()
            
        sample = mdp.trajectories.get_next_sample()
        obs = mdp._create_observation(np.concatenate(sample))
        samples.append(sample)
        obss.append(obs)

    mdp.reset()
    mdp.stop()
    joints = [joint[0] for joint in mdp.obs_helper.observation_spec]
    np_obss = np.array(obss)
    np_samples = np.array([np.array(sample).flatten() for sample in samples])
    df = pd.DataFrame()
    df['Timestep'] = np.arange(len(np_samples))
    for joint in joints:
        # skip velocity fields
        if joint[0] == 'd':
            continue
        if joint == 'q_pelvis_tx':
            df[joint] = np_samples[:, 0].flatten()
        elif joint == 'q_pelvis_tz':
            df[joint] = np_samples[:, 1].flatten()
        else:
            joint_idx = mdp.get_obs_idx(joint)
            df[joint] = np_obss[:, joint_idx].flatten()
    not_angle_joints = {'Timestep', 'q_pelvis_tx', 'q_pelvis_tz', 'q_pelvis_ty'}
    for column in df.columns:
        if column not in not_angle_joints:
            df[column] = np.rad2deg(df[column])
    peaks, _ = find_peaks(df['q_hip_flexion_r'], height=10, distance=80)
    heelstrike = [False] * 89800
    for peak in peaks:
        heelstrike[peak] = True

    df['Heelstrike'] = np.array(heelstrike)
    cycle_idx = 0
    cycle_idxs = []
    for heelstrike in df['Heelstrike']:
        if heelstrike == True:
            cycle_idx += 1
        cycle_idxs.append(cycle_idx)
    df = df.assign(Cycle_Idx = cycle_idxs)
    counter = Counter(cycle_idxs)
    proper_cycles_idxs = [idx for idx in counter if counter[idx] > 1]

    cycles = {}
    for idx in proper_cycles_idxs:
        df_ = df[df['Cycle_Idx'] == idx]
        
        for key in joints:
            if key[0] != 'd':
                y = df_[key].to_numpy()
                if key not in cycles:
                    cycles[key] = []
                cycles[key].append(y)
    return cycles

def to_training_format(data):
    dt = 0.01
    simulation_data = {}
    for speed in tqdm(data, desc='Formatting Data'):
        simulation_data[speed] = {}
        for joint in data[speed]:
            concat_data = np.concatenate(data[speed][joint])
            if joint not in ['q_pelvis_tx', 'q_pelvis_tz', 'q_pelvis_ty']:
                concat_data = np.deg2rad(concat_data)
            if 'knee' in joint:
                concat_data = -1*concat_data
            
            simulation_data[speed][joint] = concat_data   
            simulation_data[speed]['d'+joint] = np.gradient(concat_data, dt)
    ignore_keys = ["q_pelvis_tx", "q_pelvis_tz"]
    mdp = LocoEnv.make("HumanoidTorque.walk.real", headless=True, random_start=False, init_step_no=0)
    joints = [joint[0] for joint in mdp.obs_helper.observation_spec]
    training_data = {}
    for speed in simulation_data:
        training_data[speed] = {}
        all_states = np.stack([simulation_data[speed][joint] for joint in joints if joint not in ignore_keys]).T
        training_data[speed]['states'] = all_states[:-1, :]
        training_data[speed]['next_states'] = all_states[1:, :]
        training_data[speed]['absorbing'] = np.zeros(89799)
        training_data[speed]['last'] = np.concatenate([np.zeros(89799), np.ones(1)])
    return training_data

def joint_peaks(cycle, joint, normalize=False):
    cycle_length = len(cycle)
    peaks = [round(cycle_length * 0.05 * x) for x in range(20)]
    peaks.append(cycle_length -1)

    if normalize:
        peaks = [peak*100/cycle_length for peak in peaks]

    return peaks

def interpolate_data(data):
    interpolated_data = dict()
    
    for speed in tqdm(data, desc='Interpolating Data'):
        mean_length = round(np.mean([data[speed][subject]['mean_length'] for subject in data[speed]]))
        
        interpolated_cycles = {}
        for subject in data[speed]:
            joints = data[speed][subject]['cycles'].keys()
            
            for joint in joints:
                joint_cycles = data[speed][subject]['cycles'][joint]
                if joint not in interpolated_cycles:
                    interpolated_cycles[joint] = []
    
                for joint_cycle in joint_cycles:
                    x = np.linspace(0, len(joint_cycle)-1, num=len(joint_cycle))
                    xnew = np.linspace(0, len(joint_cycle)-1, num=mean_length)
                    spl = CubicSpline(x, joint_cycle) 
                    interpolated_cycles[joint].append(spl(xnew))
                    
        for joint in interpolated_cycles.keys():
            interpolated_cycles[joint] = np.array(interpolated_cycles[joint])
    
        interpolated_data[speed] = {
            'cycles': interpolated_cycles,
            'mean_length': mean_length,
        }
    return interpolated_data

def load_gatech_data(root):
    subjects = [fname for fname in os.listdir(root) if 'AB' in fname]
    all_data = dict()
    for subject in tqdm(subjects, desc='Loading GaTech data'):
        data, joints = load_subject(root=root, subject=subject, plot=False, save=False)
        for _data in data:
            if _data['speed'] not in all_data:
                all_data[_data['speed']] = dict()
                
            all_data[_data['speed']][subject] = {
                'cycles': _data['cycles'],
                'mean_length': _data['mean_length'],
            }
    return all_data, joints

def load_subject(root='../dataset_csv', subject='AB08', activity='treadmill', plot=False, save=False, debug=False):
    # path to files
    root = root
    subject = subject
    activity = activity
    files = os.listdir(f'{root}/{subject}/{activity}/ik')

    # process data
    data = []
    for file in files:
        if debug:
            print(file)
        ik = pd.read_csv(f'{root}/{subject}/{activity}/ik/{file}')
        conditions = pd.read_csv(f'{root}/{subject}/{activity}/conditions/{file}')
        gcRight = pd.read_csv(f'{root}/{subject}/{activity}/gcRight/{file}')
        _data = process_csv(ik, conditions, gcRight, debug=debug)
        data.extend(_data)
    
    data.sort(key=lambda _data: _data['speed'])

    joints = ik.columns.tolist()
    joints.remove('Header')
    return data, joints
    
def process_csv(ik, conditions, gcRight, debug=False):
    # helper function
    def remove_items(test_list, item): 
        res = [i for i in test_list if i != item] 
        return res 

    columns = ik.columns.tolist()
    columns.remove('Header')
    
    # Construct dataframe with hip angle, speed and gait
    df = pd.DataFrame()
    df['Time'] = ik['Header']
    for key in columns:
        if 'knee' in key:
            df[key] = ik[key] * -1
        else:
            df[key] = ik[key]
    df['Gait'] = gcRight[gcRight['Header'].isin(df['Time'])]['HeelStrike'].to_numpy()
    df['Speed'] = conditions[conditions['Header'].isin(df['Time'])]['Speed'].to_numpy()

    # Get Distinct Speeds
    speeds = df['Speed'].to_numpy()
    distinct_speeds = Counter(speeds).most_common(6)
    distinct_speeds = [s_[0] for s_ in distinct_speeds if s_[1] > 4000]

    # Extract data by speed
    data = []
    for speed in distinct_speeds:
        df_speed = df[df['Speed'] == speed]
        gait_percentages = df_speed['Gait'].to_numpy()
        cycle_idx = 0
        cycle_idxs = []
        for gait_percentage in gait_percentages:
            if gait_percentage == 0:
                cycle_idx += 1
            cycle_idxs.append(cycle_idx)
        df_speed = df_speed.assign(Cycle_Idx = cycle_idxs)

        # get rid of the first and last cycles
        cycle_idxs = remove_items(cycle_idxs, 0)
        cycle_idxs = remove_items(cycle_idxs, cycle_idx)
        
        # get cycles that are longer than 1
        counter = Counter(cycle_idxs)
        proper_cycles_idxs = [idx for idx in counter if counter[idx] > 1]
        
        num_cycle = len(proper_cycles_idxs)
        mean_length = round(np.mean([counter[idx] for idx in proper_cycles_idxs]))
        if debug:
            print(f'number of cycles: {num_cycle}')
            print(f'mean length of cycles: {mean_length}')
        
        cycles = {}
        for idx in proper_cycles_idxs:
            df_ = df_speed[df_speed['Cycle_Idx'] == idx]
            
            for key in columns:
                y = df_[key].to_numpy()
                if key not in cycles:
                    cycles[key] = []
                cycles[key].append(y)

        speed_data = {
            'speed': speed,
            'mean_length': mean_length,
            'cycles': cycles,
        }
        data.append(speed_data)

    return data