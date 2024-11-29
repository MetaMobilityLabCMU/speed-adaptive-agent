import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SpeedWrapper(gym.Wrapper):
    def __init__(self, env, speed_range):
        """
        Args:
            env: The environment to wrap.
            speed_range: A tuple of the form (min, max) specifying the range of speeds to be used.
        """
        super(SpeedWrapper, self).__init__(env)
        self.speed_range = speed_range
        self.set_operate_speed()

        # Extend the observation space to include target_speed
        base_obs_space = self._convert_space(self.env.info.observation_space)
        high = np.concatenate([base_obs_space.high, [self.speed_range[1]]])
        low = np.concatenate([base_obs_space.low, [self.speed_range[0]]])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = self._convert_space(self.env.info.action_space)
        
        # fix mdp info so network initiated correctly
        self.env.info.observation_space = spaces.flatten_space(self.observation_space)

    def render(self, mode='human', **kwargs):
        return self.env.render(mode)
    
    def set_operate_speed(self, speed=1.2):
        self.operate_speed = speed
        reward_params = dict(target_velocity=speed)
        self.env._reward_function = self.env._get_reward_function(reward_type="target_velocity", reward_params=reward_params)


    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, base_obs):
        return np.concatenate([base_obs, [self.operate_speed]])
    
    def reset(self, obs=None):
        obs = self.env.reset(obs=obs)

        return self._get_obs(obs)

    @staticmethod
    def _convert_space(space):
        shape = space.shape
        return spaces.Box(space.low, space.high, shape, np.float32)
