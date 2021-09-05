#!/bin/env/python3

from supersuit import observation_lambda_v0
from pettingzoo import AECEnv
import numpy as np
import gym

def one_hot_obs_wrapper(env: AECEnv) -> AECEnv:
    """
    :param env: env with observation space of Discrete(n)
    :return: wrapper env with observation as one-hot encoding
    """
    def one_hot(x, n):
        v = np.zeros(n)
        v[x] = 1.0
        return v
    max_obs_n = max([obs_space.n for obs_space in env.observation_spaces.values()])
    env = observation_lambda_v0(env,
                           lambda obs: one_hot(obs, max_obs_n),
                           lambda obs_space: gym.spaces.Box(low=np.full(obs_space.n, -np.inf),
                                                            high=np.full(obs_space.n, np.inf)
                                                            ),
                                )
    return env
