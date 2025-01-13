"""
This module defines the SpeedWrapper class, which extends the gym.Wrapper class
to add speed adaptation functionality to a given environment.

Classes:
- SpeedWrapper: A wrapper for gym environments that adds speed adaptation.

Usage:
Import the SpeedWrapper class and use it to wrap an existing environment to enable
speed adaptation during training and evaluation.

Example:
    import gymnasium as gym
    from speed_env_wrapper import SpeedWrapper

    env = gym.make('YourEnv-v0')
    speed_range = (0.65, 1.85)
    wrapped_env = SpeedWrapper(env, speed_range)
    obs = wrapped_env.reset()
    action = wrapped_env.action_space.sample()
    obs, reward, done, info = wrapped_env.step(action)
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SpeedWrapper(gym.Wrapper):
    """
    A wrapper for gym environments that adds speed adaptation functionality.

    This wrapper extends the observation space to include a target speed and modifies
    the reward function to adapt to different speeds during training and evaluation.

    Attributes:
        speed_range (tuple): A tuple specifying the range of speeds to be used.
        operate_speed (float): The current operating speed of the environment.
    """
    def __init__(self, env, speed_range):
        """
        Initialize the SpeedWrapper.

        Args:
            env: The environment to wrap.
            speed_range: A tuple of the form (min, max) specifying the range of speeds to be used.
        """
        super().__init__(env)
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

    def render(self, mode='human'):
        return self.env.render(mode)

    def set_operate_speed(self, speed=1.25):
        """
        Set the operating speed of the environment.

        Args:
            speed (float): The target speed to set. Default is 1.2.
        """
        self.operate_speed = speed
        reward_params = {"target_velocity": speed}
        self.env._reward_function = self.env._get_reward_function(
            reward_type="target_velocity",
            reward_params=reward_params)

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
