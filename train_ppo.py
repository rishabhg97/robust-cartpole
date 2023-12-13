# This script trains a PPO agent on the CartPole env.

import openai_cartpole  # This registers the environment
import gymnasium as gym


# We use PPO from stable-baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import os
import argparse
import numpy as np

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sn

from tqdm import tqdm


import wandb



parser  = argparse.ArgumentParser(description='Train PPO on CartPole')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--num_timesteps', type=int, default=10000, help='number of timesteps to train for')
parser.add_argument('--save_path', type=str, default='./ppo_agent.zip', help='path to save model')

args = parser.parse_args()

# Set random seed
seed = args.seed
np.random.seed(seed)

# Create the CartPole environment
env = gym.make('CartPole-v1')

PPO('MlpPolicy', env, verbose=1).learn(total_timesteps=args.num_timesteps).save(args.save_path)
