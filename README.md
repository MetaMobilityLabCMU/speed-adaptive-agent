# Learning Speed-Adaptive Walking Agent Using Imitation Learning with Physics-Informed Simulation
Code for our paper submission to ICORR 2025 | [Arxiv Version](https://arxiv.org/pdf/2412.03949)

## Install Dependencies
1. Follow the instructions to install [LocoMuJoCo](https://github.com/robfiras/loco-mujoco), and download the real dataset by running
```bash
loco-mujoco-download-real
```
2. Follow the instructions to install the [imitation learning library](https://github.com/robfiras/ls-iq)
3. Install [Matlab](https://www.mathworks.com/help/install/ug/install-products-with-internet-connection.html) and add it to path
## Synthetic Data Generation
First download and unzip the [Camargo et al. Dataset](https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/) from Epic Lab in `data` folder.

> [!NOTE]
> Since the entire dataset is quite large, you could just download the treadmill 
> data from each subject, but make sure to organize the data in the folder structure.

Next run the matlab script to convert all the .mat files to csv format in `data/dataset_csv`.
> [!NOTE]
> you could also run `mat2csv.m` in matlab UI if matlab is not in the system PATH
```bash
cd data
matlab -nodisplay -nosplash -nodesktop  -r "mat2csv; exit"
```
Lastly, we could generate the synthetic dataset by running
```bash
python generate_synthetic_data.py
```
This script first fit a linear model from Camargo's dataset and generate synthetic varying speed (13 speeds from 0.65 to 1.85 m/s in 0.1 increment) data from locomujoco's data into `data/locomujoco_13_speeds_dataset_unformatted.pkl` for evaluation and `data/locomujoco_13_speeds_dataset.pkl` for training.
## Speed-Adaptive-Agent Training
We use the experiment launcher package to organize the training, `confs.yaml` provides the imitation learning algorithm specification and related hyper parameters, `experiment.py` defines the main training loop and `launcher.py` specifies the training settings including environment, speed reward ratio, curriculum type and training epoch.
To train a speed-adaptive agent with the optimal settings run
```bash
cd training
python launcher.py
```
The training results and model checkpoints are saved in `./logs` folder, you can use `tensorboard` to visualize the results by 
```bash
tensorboard --logdir logs
```
"Eval_R-stochastic", "Eval_J-stochastic", and "Eval_L-stochastic" are the three main metrics to monitor, which are the mean undiscounted return, mean discounted return, and the mean length of an episode the agent, respectively.
## Agent Evaluation
To evaluate a speed-adaptive-agent, run the following with one of the model checkpoint
```bash
cd training
python eval.py --model_path PATH/TO/YOUR/MODEL
```
This script will perform inference using the model at each of the 13 speeds from the synthetic dataset. It will compute and print the R² score and RMSE for joint kinematics by comparing the agent's predicted motions against the synthetic dataset. Additionally, it compute and print the R² score and RMSE for how well the agent matches the target speeds.