import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import CubicSpline, PchipInterpolator
import os
os.environ['MUJOCO_GL'] = 'egl'
import copy
import pickle 

from utils import *

def get_control_points(cycle, normalize=False):
    cycle_length = len(cycle)
    points = [round(cycle_length * 0.05 * x) for x in range(20)]
    points.append(cycle_length -1)
    if normalize:
        points = [point*100/cycle_length for point in points]
    return points

def linear_transformation(control_gait_phase, control_angle, linear_models, joint, speed, root_speed):
    assert len(control_gait_phase) == linear_models[joint]['num_control_point']
    gait_phase_intercepts = np.array(linear_models[joint]['intercepts']['Gait Phase'])
    gait_phase_slopes = np.array(linear_models[joint]['slopes']['Gait Phase'])
    angle_intercepts = np.array(linear_models[joint]['intercepts']['Angle'])
    angle_slopes = np.array(linear_models[joint]['slopes']['Angle'])
    transformed_gait_phase = np.add(control_gait_phase, np.add(gait_phase_slopes * (speed - root_speed), gait_phase_intercepts))
    transformed_angle = np.add(control_angle, np.add(angle_slopes * (speed - root_speed), angle_intercepts))
    return transformed_gait_phase, transformed_angle

def generate_data(cycles, transform, linear_models, root_speed=1.25):
	speeds = np.round(np.linspace(0.65, 1.85, 13), 2)
	transformed_data = dict()
	for speed in tqdm(speeds, desc='Generating Synthetic Data'):
		transformed_data[speed] = {}
		for joint in cycles:
			transformed_data[speed][joint] = []
			for cycle in cycles[joint]:
				cycle_length = len(cycle)
				idx = np.arange(cycle_length)
				if 'knee' in joint:
					cycle = -1*cycle

				if joint == 'q_pelvis_tx':
					new_cycle = cycle * speed / root_speed
					transformed_data[speed][joint].append(new_cycle)
					continue
					
				control_gait_phase = get_control_points(cycle)
				control_angle = np.interp(control_gait_phase, idx, cycle)
				
				# apply transformation
				transformed_gait_phase, transformed_angle = linear_transformation(control_gait_phase, control_angle, linear_models, joint[2:], speed, root_speed)
				Pchip = PchipInterpolator(transformed_gait_phase, transformed_angle, extrapolate='period')
				
				new_cycle = Pchip(idx)
				transformed_data[speed][joint].append(new_cycle)
	return transformed_data

if __name__ == "__main__":
	data, joints = load_gatech_data(root='../../gatech-dataset/dataset_csv')

	del data[1.9]
	del data[1.95]
	del data[2.0]
	del data[2.05]

	interpolated_data = interpolate_data(data)

	speeds = list(data.keys())
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

	root_speed = 1.25
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


	loco_data = extract_locomujoco_data()
	generated_data = generate_data(loco_data, linear_transformation, linear_models)
	format_data = to_training_format(generated_data)


	with open('../data/locomujoco_13_speeds_dataset.pkl', 'wb') as f:
		pickle.dump(format_data, f)