import yaml
import numpy as np

import torch.nn.functional as F
import torch.optim as optim

from mushroom_rl.core.serialization import *
from mushroom_rl.policy import GaussianTorchPolicy

from imitation_lib.imitation import VAIL_TRPO
from imitation_lib.utils import FullyConnectedNetwork, NormcInitializer, Standardizer, VariationalNet, VDBLoss

import pickle
from speed_vail import SpeedVAIL

def get_agent(env_id, mdp, use_cuda, sw, conf_path=None):

    if conf_path is None:
        conf_path = 'confs.yaml'    # use default one

    with open(conf_path, 'r') as f:
        confs = yaml.safe_load(f)

    # get conf for environment
    try:
        # get the default conf (task agnostic)
        env_id_short = env_id.split('.')[0]
        conf = confs[env_id_short]
    except KeyError:
        # get the conf for the specific environment and task
        env_id_short = ".".join(env_id.split('.')[:2])
        conf = confs[env_id_short]

    if conf["algorithm"] == "VAIL":
        agent = create_vail_agent(mdp, sw, use_cuda, **conf["algorithm_config"])
    elif conf["algorithm"] == "SpeedVAIL":
        agent = create_speed_vail_agent(mdp, sw, use_cuda, **conf["algorithm_config"])
    else:
        raise ValueError(f"Invalid algorithm: {conf['algorithm']}")

    return agent

def create_vail_agent(mdp, sw, use_cuda, std_0, info_constraint, lr_beta, z_dim, disc_only_states,
                      disc_use_next_states, train_disc_n_th_epoch, disc_batch_size, learning_rate_critic,
                      learning_rate_disc, policy_entr_coef, max_kl, n_epochs_cg, use_noisy_targets,
                      last_policy_activation):

    mdp_info = deepcopy(mdp.info)
    expert_data = mdp.create_dataset()

    trpo_standardizer = Standardizer(use_cuda=use_cuda)
    policy_params = dict(network=FullyConnectedNetwork,
                         input_shape=mdp_info.observation_space.shape,
                         output_shape=mdp_info.action_space.shape,
                         std_0=std_0,
                         n_features=[512, 256],
                         initializers=[NormcInitializer(1.0), NormcInitializer(1.0), NormcInitializer(0.001)],
                         activations=['relu', 'relu', last_policy_activation],
                         standardizer=trpo_standardizer,
                         use_cuda=use_cuda)

    critic_params = dict(network=FullyConnectedNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': learning_rate_critic,
                                               'weight_decay': 0.0}},
                         loss=F.mse_loss,
                         batch_size=256,
                         input_shape=mdp_info.observation_space.shape,
                         activations=['relu', 'relu', 'identity'],
                         standardizer=trpo_standardizer,
                         squeeze_out=False,
                         output_shape=(1,),
                         initializers=[NormcInitializer(1.0), NormcInitializer(1.0), NormcInitializer(0.001)],
                         n_features=[512, 256],
                         use_cuda=use_cuda)

    discrim_obs_mask = mdp.get_kinematic_obs_mask()
    discrim_act_mask = [] if disc_only_states else np.arange(mdp_info.action_space.shape[0])
    discrim_input_shape = (len(discrim_obs_mask) + len(discrim_act_mask),) if not disc_use_next_states else \
        (2 * len(discrim_obs_mask) + len(discrim_act_mask),)
    discrim_standardizer = Standardizer()
    z_size = z_dim
    encoder_net = FullyConnectedNetwork(input_shape=discrim_input_shape, output_shape=(128,), n_features=[256],
                                        activations=['relu', 'relu'], standardizer=None,
                                        squeeze_out=False, use_cuda=use_cuda)
    decoder_net = FullyConnectedNetwork(input_shape=(z_size,), output_shape=(1,), n_features=[],
                                        # no features mean no hidden layer -> one layer
                                        activations=['identity'], standardizer=None,
                                        initializers=[NormcInitializer(std=0.1)],
                                        squeeze_out=False, use_cuda=use_cuda)

    discriminator_params = dict(optimizer={'class': optim.Adam,
                                           'params': {'lr': learning_rate_disc,
                                                      'weight_decay': 0.0}},
                                batch_size=disc_batch_size,
                                network=VariationalNet,
                                input_shape=discrim_input_shape,
                                output_shape=(1,),
                                z_size=z_size,
                                encoder_net=encoder_net,
                                decoder_net=decoder_net,
                                use_next_states=disc_use_next_states,
                                use_actions=not disc_only_states,
                                standardizer=discrim_standardizer,
                                use_cuda=use_cuda)

    alg_params = dict(train_D_n_th_epoch=train_disc_n_th_epoch,
                      state_mask=discrim_obs_mask,
                      act_mask=discrim_act_mask,
                      n_epochs_cg=n_epochs_cg,
                      trpo_standardizer=trpo_standardizer,
                      D_standardizer=discrim_standardizer,
                      loss=VDBLoss(info_constraint=info_constraint, lr_beta=lr_beta),
                      ent_coeff=policy_entr_coef,
                      use_noisy_targets=use_noisy_targets,
                      max_kl=max_kl,
                      use_next_states=disc_use_next_states)

    agent = VAIL_TRPO(mdp_info=mdp_info, policy_class=GaussianTorchPolicy, policy_params=policy_params, sw=sw,
                      discriminator_params=discriminator_params, critic_params=critic_params,
                      demonstrations=expert_data, **alg_params)
    return agent
def create_speed_vail_agent(mdp, sw, use_cuda, std_0, info_constraint, lr_beta, z_dim, disc_only_states,
                      disc_use_next_states, train_disc_n_th_epoch, disc_batch_size, learning_rate_critic,
                      learning_rate_disc, policy_entr_coef, max_kl, n_epochs_cg, use_noisy_targets,
                      last_policy_activation):
    mdp_info = deepcopy(mdp.info)

    with open('locomujoco_13_speeds_dataset.pkl', 'rb') as f:
        expert_data = pickle.load(f)

    trpo_standardizer = Standardizer(use_cuda=use_cuda)
    policy_params = dict(network=FullyConnectedNetwork,
                         input_shape=mdp_info.observation_space.shape,
                         output_shape=mdp_info.action_space.shape,
                         std_0=std_0,
                         n_features=[512, 256],
                         initializers=[NormcInitializer(1.0), NormcInitializer(1.0), NormcInitializer(0.001)],
                         activations=['relu', 'relu', last_policy_activation],
                         standardizer=trpo_standardizer,
                         use_cuda=use_cuda)

    critic_params = dict(network=FullyConnectedNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': learning_rate_critic,
                                               'weight_decay': 0.0}},
                         loss=F.mse_loss,
                         batch_size=256,
                         input_shape=mdp_info.observation_space.shape,
                         activations=['relu', 'relu', 'identity'],
                         standardizer=trpo_standardizer,
                         squeeze_out=False,
                         output_shape=(1,),
                         initializers=[NormcInitializer(1.0), NormcInitializer(1.0), NormcInitializer(0.001)],
                         n_features=[512, 256],
                         use_cuda=use_cuda)

    discrim_obs_mask = mdp.get_kinematic_obs_mask()
    discrim_act_mask = [] if disc_only_states else np.arange(mdp_info.action_space.shape[0])
    discrim_input_shape = (len(discrim_obs_mask) + len(discrim_act_mask),) if not disc_use_next_states else \
        (2 * len(discrim_obs_mask) + len(discrim_act_mask),)
    discrim_standardizer = Standardizer()
    z_size = z_dim
    encoder_net = FullyConnectedNetwork(input_shape=discrim_input_shape, output_shape=(128,), n_features=[256],
                                        activations=['relu', 'relu'], standardizer=None,
                                        squeeze_out=False, use_cuda=use_cuda)
    decoder_net = FullyConnectedNetwork(input_shape=(z_size,), output_shape=(1,), n_features=[],
                                        # no features mean no hidden layer -> one layer
                                        activations=['identity'], standardizer=None,
                                        initializers=[NormcInitializer(std=0.1)],
                                        squeeze_out=False, use_cuda=use_cuda)

    discriminator_params = dict(optimizer={'class': optim.Adam,
                                           'params': {'lr': learning_rate_disc,
                                                      'weight_decay': 0.0}},
                                batch_size=disc_batch_size,
                                network=VariationalNet,
                                input_shape=discrim_input_shape,
                                output_shape=(1,),
                                z_size=z_size,
                                encoder_net=encoder_net,
                                decoder_net=decoder_net,
                                use_next_states=disc_use_next_states,
                                use_actions=not disc_only_states,
                                standardizer=discrim_standardizer,
                                use_cuda=use_cuda)

    alg_params = dict(train_D_n_th_epoch=train_disc_n_th_epoch,
                      state_mask=discrim_obs_mask,
                      act_mask=discrim_act_mask,
                      n_epochs_cg=n_epochs_cg,
                      trpo_standardizer=trpo_standardizer,
                      D_standardizer=discrim_standardizer,
                      loss=VDBLoss(info_constraint=info_constraint, lr_beta=lr_beta),
                      ent_coeff=policy_entr_coef,
                      use_noisy_targets=use_noisy_targets,
                      max_kl=max_kl,
                      use_next_states=disc_use_next_states)

    agent = SpeedVAIL(mdp_info=mdp_info, policy_class=GaussianTorchPolicy, policy_params=policy_params, sw=sw,
                      discriminator_params=discriminator_params, critic_params=critic_params,
                      demonstration_collections=expert_data, **alg_params)
    return agent

class ProgressionCurriculum():
    def __init__(self, speed_range):
        self.speed_range = speed_range
        self.idx = round(len(self.speed_range)/2)
        self.direction = 1
    def get_target_speed(self):
        target_speed = self.speed_range[self.idx]
        if self.idx == len(self.speed_range)-1 and self.direction == 1:
            self.direction = -1
        if self.idx == 0 and self.direction == -1:
            self.direction = 1
        self.idx = self.idx + self.direction
        return target_speed

class RandomCurriculum():
    def __init__(self, speed_range):
        self.speed_range = speed_range
        self.rng = np.random.default_rng()
    def get_target_speed(self):
        target_speed = self.rng.choice(self.speed_range, 1)[0]
        return target_speed

def compute_mean_speed(env, dataset):
    x_vel_idx = env.get_obs_idx("dq_pelvis_tx")
    speeds = []
    for i in range(len(dataset)):
        speeds.append(dataset[i][0][x_vel_idx])
    return np.mean(speeds)