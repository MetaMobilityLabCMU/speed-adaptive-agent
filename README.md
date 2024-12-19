# speed-adaptive-agent
code for ICORR 2025 Speed Adaptive Agent paper

## Install Dependencies
1. Follow the instructions to install LocoMuJoCo [link](https://github.com/robfiras/loco-mujoco)
2. Follow the instructions to install the imitation library [link](https://github.com/robfiras/ls-iq)
## Synthetic Data Generation
1. Download [Camargo et al. Dataset](https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/) from Epic Lab and put it into the `data` folder.
     * Since the entire dataset is quite large, you could only download the treadmill data from each subject, but remember to organize the data in the same folder structure.
2. Run `data_generation/mat2csv.m` in matlab to convert the treadmill data into csv format
3. Run `data_generation/generate_synthetic_data.py` to generate the synthetic dataset.
## Speed-Adaptive-Agent Training
1. Run `training/launcher.py` will train a speed-adaptive agent with the optimal training
     * Baseline settings: Change the `algorithm` flag in `training/confs.yaml` from `"SpeedVAIL"` to `"VAIL"`
     * `reward_ratio` changes the speed reward ratio (float from 0 - 1)
     * `curriculum` changes the training curriculum (available options are 'progression' and 'random') 
2. You can check the training status with tensorboard `tensorboard --logdir logs`
3. Model checkpoints will be saved in the `logs` folder
## Agent Evaluation
We provide a jupyter notebook example `training/eval.ipynb` for inferencing a model and calculating its speed and joint angle RMSE/R2 compared to synthetic data.


## TODO
1. matlab code for generating synthetic dataset