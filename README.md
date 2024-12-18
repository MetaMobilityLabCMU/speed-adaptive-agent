# speed-adaptive-agent
code for ICORR 2025 Speed Adaptive Agent paper

## Install Dependencies
## Synthetic Data Generation
1. Download [Camargo et al. Dataset](https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/) from Epic Lab and put it into the `data` folder.
     * Since the entire dataset is quite large, you could only download the treadmill data from each subject, but remember to organize the data in the same folder structure.
2. Run `data_generation/mat2csv.m` in matlab to convert the treadmill data into csv format
3. Run `data_generation/generate_synthetic_data.py` to generate the synthetic dataset.
## Speed-Adaptive-Agent Training
## Agent Evaluation


## TODO
1. matlab code for generating synthetic dataset
2. code for training
3. code for evaluation
4. write README.md