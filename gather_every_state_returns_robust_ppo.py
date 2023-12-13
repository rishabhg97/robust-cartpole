# This file loads a PPO agent trained using adversarial environment and
# tests its rollout performance on the entire state-space of the adversary.


import time

from tqdm import tqdm

import openai_cartpole # Required for using the modified cartpole environment
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.preprocessing import check_for_nested_spaces, is_image_space, is_image_space_channels_first
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.base_class import maybe_make_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecTransposeImage,
    is_vecenv_wrapped,
    unwrap_vec_normalize,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env


import gymnasium as gym
from gymnasium import spaces
import sys
import time

import torch
import torch.nn.functional as F


# We use PPO from stable-baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback

from stable_baselines3.common.vec_env import DummyVecEnv

import argparse
import numpy as np

from conjuagate_env_adversary_rl_agent import AugmentedPPO

import wandb




def get_observation_from_index(index):
    """
    Convert a single observation index back to the observation state.
    """
    observation = []
    base = 3
    for i in range(5):
        observation.append(index % base)
        index //= base
    # Reverse it because the last element corresponds to the highest place value
    return observation[::-1]


# Load the robustly trained PPO agent
rl_agent = AugmentedPPO.load('./models/env_robust_ppo_agent.zip')

# Environment
env = gym.make('openai_cartpole/ModifiedCartPole-v1')
env.reset()

orig_gravity, orig_masscart, orig_masspole, orig_length, orig_force_mag = env.get_params()

# State space definition for the adversary
GRAVITY_MIN, GRAVITY_MAX = (orig_gravity * 0.8, orig_gravity * 1.2)
MASSCART_MIN, MASSCART_MAX = (orig_masscart * 0.8, orig_masscart * 1.2)
MASSPOLE_MIN, MASSPOLE_MAX = (orig_masspole * 0.8, orig_masspole * 1.2)
LENGTH_MIN, LENGTH_MAX = (orig_length * 0.8, orig_length * 1.2)
FORCE_MAG_MIN, FORCE_MAG_MAX = (orig_force_mag * 0.8, orig_force_mag * 1.2)

# Set the ranges for the environment parameters
env.set_param_ranges(
    gravity=(GRAVITY_MIN, GRAVITY_MAX),
    masscart=(MASSCART_MIN, MASSCART_MAX),
    masspole=(MASSPOLE_MIN, MASSPOLE_MAX),
    length=(LENGTH_MIN, LENGTH_MAX),
    force_mag=(FORCE_MAG_MIN, FORCE_MAG_MAX),
)

# The observation space
observation_space = {
    "gravity": np.linspace(GRAVITY_MIN, GRAVITY_MAX, 3),
    "masscart": np.linspace(MASSCART_MIN, MASSCART_MAX, 3),
    "masspole": np.linspace(MASSPOLE_MIN, MASSPOLE_MAX, 3),
    "length": np.linspace(LENGTH_MIN, LENGTH_MAX, 3),
    "force_mag": np.linspace(FORCE_MAG_MIN, FORCE_MAG_MAX, 3),
}

# Action space for the adversary which is the set of all possible changes to the individual environment parameters
action_space = {
    "delta_gravity": np.linspace(GRAVITY_MIN-orig_gravity, GRAVITY_MAX-orig_gravity, 3),
    "delta_masscart": np.linspace(MASSCART_MIN-orig_masscart, MASSCART_MAX-orig_masscart, 3),
    "delta_masspole": np.linspace(MASSPOLE_MIN-orig_masspole, MASSPOLE_MAX-orig_masspole, 3),
    "delta_length": np.linspace(LENGTH_MIN-orig_length, LENGTH_MAX-orig_length, 3),
    "delta_force_mag": np.linspace(FORCE_MAG_MIN-orig_force_mag, FORCE_MAG_MAX-orig_force_mag, 3),
}

# Note that we change only one parameter at a time with a chosen delta value. This makes the action space smaller.
# From 3^5 to 3*5.

# Q-learning algorithm
action_space_variables = action_space.keys()
num_delta_per_variable = 3

get_action_from_index_map = {
    variable_index * num_delta_per_variable + change_index: (variable_index, change_index, variable)
    for variable_index, variable in enumerate(action_space_variables)
    for change_index in range(num_delta_per_variable)
}

# Iterate over the entire state-space of the adversary and evaluate the performance of the PPO agent
# on each state.
state_returns = []

for state_index in tqdm(range(243)):
    observation = get_observation_from_index(state_index)

    # Set the environment parameters to the current observation
    env.modify_params(
        delta_gravity=observation[0] - orig_gravity,
        delta_masscart=observation[1] - orig_masscart,
        delta_masspole=observation[2] - orig_masspole,
        delta_length=observation[3] - orig_length,
        delta_force_mag=observation[4] - orig_force_mag,
    )

    env.reset()
    # Evaluate the performance of the PPO agent
    mean_reward, _ = evaluate_policy(rl_agent, env, n_eval_episodes=10)
    state_returns.append(mean_reward)
    
# Save the returns
np.save("./logs/robust_every_state_returns.npy", state_returns)
