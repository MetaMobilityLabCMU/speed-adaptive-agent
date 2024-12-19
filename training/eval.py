import pickle
import loco_mujoco
from loco_mujoco import LocoEnv
import numpy as np
from dm_control.mujoco import Physics
from utils import process_data, interpolate_data, eval_model, calculate_metrics
from speed_env_wrapper import SpeedWrapper
from scipy.signal import find_peaks
from sklearn.metrics import root_mean_squared_error, r2_score
import os

if __name__ == "__main__":
    library_path = os.path.dirname(loco_mujoco.__file__)
    file_path = os.path.join(library_path, 'environments', 'data', 'humanoid', 'humanoid_torque.xml')
    physics = Physics.from_xml_path(file_path)
    mass = np.sum(physics.named.model.body_mass._field)
    speed_range = np.round(np.linspace(0.65, 1.85, 13), 2)
    mdp = LocoEnv.make("HumanoidTorque.walk", headless=True)
    mdp = SpeedWrapper(mdp, (speed_range[0], speed_range[-1]))
    _ = mdp.reset()

    # Load synthetic data as ground truth
    with open('../data/locomujoco_13_speeds_dataset_unformatted.pkl', 'rb') as f:
        synthetic_data = pickle.load(f)
        
    min_peak_synthetic_data = {}

    for speed in synthetic_data:
        min_peak_synthetic_data[speed] = {}

        concat_data = {}
        for joint in ['q_hip_flexion_r', 'q_knee_angle_r', 'q_ankle_angle_r']:
            concat_data[joint] = np.concatenate(synthetic_data[speed][joint])
        
        hip_data = np.concatenate(synthetic_data[speed]['q_hip_flexion_r'])
        heel_strikes, _ = find_peaks(-1*hip_data, height=np.deg2rad(8), distance=40)
        cycle_lengths = np.diff(heel_strikes)
        
        for joint in ['q_hip_flexion_r', 'q_knee_angle_r', 'q_ankle_angle_r']:
            min_peak_synthetic_data[speed][joint] = []   
            for i, cycle_length in enumerate(cycle_lengths):
                if cycle_length > 40:
                    min_peak_synthetic_data[speed][joint].append(concat_data[joint][heel_strikes[i]:heel_strikes[i+1]])
                    
    interpolated_data = interpolate_data(min_peak_synthetic_data, 114)   

    # Specify model path
    model_path = 'path/to/your/model'

    # Model inference and calculate RMSE and R2
    results = eval_model(mdp, model_path, speed_range, mass, n_trials=5, n_episodes=3, cycle_length_cutoff=60, record=False)
    processed_results = {}
    metric = calculate_metrics(results, interpolated_data)
    speed_RMSEs, speed_R2s, bio_RMSEs, bio_R2s = [], [], [], []
    for trial in metric:
        actual_speeds = [metric[trial][speed]['actual_speed'] for speed in metric[trial]]
        target_speeds = list(metric[trial].keys())
        
        speed_RMSEs.append(root_mean_squared_error(target_speeds, actual_speeds))
        speed_R2s.append(r2_score(target_speeds, actual_speeds))
        bio_RMSEs.append(np.mean([metric[trial][speed]['RMSE'] for speed in metric[trial]]))
        bio_R2s.append(np.mean([metric[trial][speed]['R2'] for speed in metric[trial]]))
        
    processed_results['speed_RMSE'] = speed_RMSEs
    processed_results['speed_R2'] = speed_R2s
    processed_results['bio_RMSE'] = bio_RMSEs
    processed_results['bio_R2'] = bio_R2s

    bio_RMSE_avg = np.mean(processed_results['bio_RMSE'])
    bio_RMSE_std = np.std(processed_results['bio_RMSE'])
    bio_R2_avg = np.mean(processed_results['bio_R2'])
    bio_R2_std = np.std(processed_results['bio_R2'])

    speed_RMSE_avg = np.mean(processed_results['speed_RMSE'])
    speed_RMSE_std = np.std(processed_results['speed_RMSE'])
    speed_R2_avg = np.mean(processed_results['speed_R2'])
    speed_R2_std = np.std(processed_results['speed_R2'])

    print(f'bio RMSE avg {bio_RMSE_avg}')
    print(f'bio RMSE std {bio_RMSE_std}')
    print(f'bio R2 avg {bio_R2_avg}')
    print(f'bio R2 std {bio_R2_std}')

    print(f'speed RMSE avg {speed_RMSE_avg}')
    print(f'speed RMSE std {speed_RMSE_std}')
    print(f'speed R2 avg {speed_R2_avg}')
    print(f'speed R2 std {speed_R2_std}')