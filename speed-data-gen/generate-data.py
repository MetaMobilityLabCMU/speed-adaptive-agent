import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import CubicSpline, PchipInterpolator
import os
import copy
import pickle 

from utils import *

def linear_transformation(control_gait_phase, control_angle, linear_models, joint, speed, root_speed):
    assert len(control_gait_phase) == linear_models[joint]['num_control_point']
    gait_phase_intercepts = np.array(linear_models[joint]['intercepts']['Gait Phase'])
    gait_phase_slopes = np.array(linear_models[joint]['slopes']['Gait Phase'])
    angle_intercepts = np.array(linear_models[joint]['intercepts']['Angle'])
    angle_slopes = np.array(linear_models[joint]['slopes']['Angle'])
    transformed_gait_phase = np.add(control_gait_phase, np.add(gait_phase_slopes * (speed - root_speed), gait_phase_intercepts))
    transformed_angle = np.add(control_angle, np.add(angle_slopes * (speed - root_speed), angle_intercepts))
    return transformed_gait_phase, transformed_angle

def transform_data(data, transform, linear_models, root_speed=1.2):
    transformed_data = dict()
    
    for speed in tqdm(data, desc='Transforming Data'):
        transformed_data[speed] = {}
        
        for subject in data[root_speed]:
            transformed_data[speed][subject] = dict()
            transformed_data[speed][subject]['mean_length'] = data[root_speed][subject]['mean_length']
            new_cycles = {}
            
            for key in ['Hip', 'Knee', 'Ankle']:
                joint_cycles = data[root_speed][subject]['cycles'][key]
                if key not in new_cycles:
                    new_cycles[key] = []
    
                for cycle in joint_cycles:
                    cycle_length = len(cycle)
                    idx = np.arange(cycle_length)

                    control_gait_phase = joint_peaks(cycle, joint=key)
                    control_angle = np.interp(control_gait_phase, idx, cycle)
                    
                    # apply transformation
                    transformed_gait_phase, transformed_angle = transform(control_gait_phase, control_angle, linear_models, key, speed, root_speed)
                    Pchip = PchipInterpolator(transformed_gait_phase, transformed_angle, extrapolate='period')
                    
                    new_cycle = Pchip(idx)
                    new_cycles[key].append(new_cycle)

            transformed_data[speed][subject]['cycles'] = new_cycles
            
    return transformed_data

if __name__ == "__main__":

	data = load_all_data(root='../../gatech-dataset/dataset_csv')

	del data[1.9]
	del data[1.95]
	del data[2.0]
	del data[2.05]

	interpolated_data = interpolate_data(data)

	speeds = list(data.keys())
	joints = ['Hip', 'Knee', 'Ankle']
	control_points = {joint: {} for joint in joints}
	idxs_list = dict()
	for speed in speeds:
		idx = np.linspace(0, 1, interpolated_data[speed]['mean_length']) * 100
		idxs_list[speed] = idx

	for joint in joints:
		for speed in speeds:
			avg_cycle = np.mean(interpolated_data[speed]['cycles'][joint], axis=0)
			cycle_length = len(avg_cycle)
			
			control_xs = joint_peaks(avg_cycle, joint=joint, normalize=True)
			control_ys = np.interp(control_xs, idxs_list[speed], avg_cycle)
			
			control_points[joint][speed] = {
				'Gait Phase': control_xs,
				'Angle': control_ys
			}

	root_speed = 1.2
	linear_models = {
		'Root Speed': root_speed
	}

	for joint in joints:
		num_control_point = len(control_points[joint][root_speed]['Gait Phase'])
		diff_speed = [speed-root_speed for speed in control_points[joint]] 
		
		intercepts_gait_phase = []
		intercepts_angle = []
		slopes_gait_phase = []
		slopes_angle = []
		for control_point_num in range(num_control_point):
			diff_gait = [control_points[joint][speed]['Gait Phase'][control_point_num]-control_points[joint][root_speed]['Gait Phase'][control_point_num] for speed in control_points[joint]]
			diff_gait_result = stats.linregress(diff_speed, diff_gait)
		
			diff_angle = [control_points[joint][speed]['Angle'][control_point_num]-control_points[joint][root_speed]['Angle'][control_point_num] for speed in control_points[joint]]
			diff_angle_result = stats.linregress(diff_speed, diff_angle)
			intercepts_gait_phase.append(diff_gait_result.intercept)
			intercepts_angle.append(diff_angle_result.intercept)
			slopes_gait_phase.append(diff_gait_result.slope)
			slopes_angle.append(diff_angle_result.slope)

		linear_models[joint] = {
			'intercepts': {
				'Gait Phase': intercepts_gait_phase,
				'Angle': intercepts_angle
			},
			'slopes': {
				'Gait Phase': slopes_gait_phase,
				'Angle': slopes_angle
			},
			'num_control_point': num_control_point
		}

	transformed_data = transform_data(data, linear_transformation, linear_models)

	fake_data = interpolate_data(transformed_data)
	plot_data(fake_data, f'Artificial Data', save=True, scatter=False, close=True)

	real_data = interpolate_data(data)
	plot_data(real_data, 'Real Data', save=True, scatter=False, close=True)