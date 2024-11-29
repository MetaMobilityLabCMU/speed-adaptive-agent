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

def plot_data(data, title, save=False, close=False, scatter=False, folder=None):
    mpl.rcParams.update({'font.size': 16})
    
    speeds = list(data.keys())
    idxs_list = []
    joints = ['Hip', 'Knee', 'Ankle']
    for speed in speeds:
        idx = np.linspace(0, 1, data[speed]['mean_length']) * 100
        idxs_list.append(idx)
        
    cmap = plt.get_cmap('plasma')
    norm = plt.Normalize(min(speeds), max(speeds))
    
    fig, axs = plt.subplots(3, 1, figsize=(9, 11))
    
    for i, joint in enumerate(joints):
        for j, speed in enumerate(speeds):
            avg_cycle = np.mean(data[speed]['cycles'][joint], axis=0)
            axs[i].plot(idxs_list[j], avg_cycle, color=cmap(norm(speed)))
            
            control_gait_phase = joint_peaks(avg_cycle, joint=joint, normalize=True)
            control_angle = np.interp(control_gait_phase, idxs_list[j], avg_cycle)

            if scatter:
                axs[i].scatter(control_gait_phase, control_angle, color='black', s=5, zorder=3)
            axs[i].set_xlim(-5, 105)
    
            if i == 2:
                axs[i].set_xticks([0, 20, 40, 60, 80, 100])
                axs[i].set_xlabel("Gait Phase (%)")
            else:
                axs[i].tick_params('x', labelbottom=False)

        if joint == 'Hip':
            axs[i].set_yticks([-20, 0, 20])
        if joint == 'Knee':
            axs[i].set_yticks([0, 20, 40, 60])
        if joint == 'Ankle':
            axs[i].set_yticks([-20, 0, 20])

        axs[i].yaxis.tick_right()
        axs[i].set_ylabel(joint)
                
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0)
    fig.subplots_adjust(right=0.7)
    
    cbar_ax = fig.add_axes([0.8, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='vertical')
    cbar.set_ticks([min(speeds), max(speeds)])
    
    fig.text(0.03, 0.5, 'Angle (deg)', va='center', rotation='vertical', fontdict={'size':22})
    fig.text(0.41, 0.9, f'{title}', horizontalalignment='center', fontdict={'size':22})
    fig.text(0.89, 0.5, 'Speed (m/s)', va='center', rotation='vertical', fontdict={'size':22})

    if save:
        if folder:
            import os
            os.makedirs(folder, exist_ok=True) 
            plt.savefig(f'{folder}/{title}.png')
        else:
            plt.savefig(f'{title}.png')
    if close:
        plt.close()

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

def load_all_data(root):
    subjects = os.listdir(root)
    all_data = dict()
    for subject in tqdm(subjects, desc='Loading data'):
        data = load_subject(root=root, subject=subject, plot=False, save=False)
        for _data in data:
            if _data['speed'] not in all_data:
                all_data[_data['speed']] = dict()
                
            all_data[_data['speed']][subject] = {
                'cycles': _data['cycles'],
                'mean_length': _data['mean_length'],
            }
    return all_data

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
    return data
    
def process_csv(ik, conditions, gcRight, debug=False):
    # helper function
    def remove_items(test_list, item): 
        res = [i for i in test_list if i != item] 
        return res 
    
    # Construct dataframe with hip angle, speed and gait
    df = pd.DataFrame()
    df['Time'] = ik['Header']
    df['Hip'] = ik['hip_flexion_r']
    df['Knee'] = ik['knee_angle_r'] * -1
    df['Ankle'] = ik['ankle_angle_r']
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
            
            for key in ['Hip', 'Knee', 'Ankle']:
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