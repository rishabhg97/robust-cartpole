# This file tests the openai gym environment modified for the purpose of this project.


import gymnasium as gym
import numpy as np
import openai_cartpole
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


ppo_filename = "ppo_agent.zip"
agent = PPO.load(ppo_filename)

env = gym.make('openai_cartpole/ModifiedCartPole-v1')


# for _ in range(100):
#     action = env.action_space.sample()
#     env.step(action)
#     env.render()


# Change the parameters of the environment through the 
# delta changes in 'delta_gravity', 'delta_masscart', 'delta_masspole', 'delta_length', and 'delta_force_mag'.

print(evaluate_policy(agent, env, n_eval_episodes=100))

# Original physical parameters of the environment. These constitute the first observation for the adversary.
orig_gravity, orig_masscart, orig_masspole, orig_length, orig_force_mag = env.get_params()

# State space definition for the adversary
GRAVITY_MIN, GRAVITY_MAX = (orig_gravity * 0.7, orig_gravity * 1.3)
MASSCART_MIN, MASSCART_MAX = (orig_masscart * 0.7, orig_masscart * 1.3)
MASSPOLE_MIN, MASSPOLE_MAX = (orig_masspole * 0.7, orig_masspole * 1.3)
LENGTH_MIN, LENGTH_MAX = (orig_length * 0.7, orig_length * 1.3)
FORCE_MAG_MIN, FORCE_MAG_MAX = (orig_force_mag * 0.7, orig_force_mag * 1.3)

# Set the ranges for the environment parameters
env.set_param_ranges(
    gravity=(GRAVITY_MIN, GRAVITY_MAX),
    masscart=(MASSCART_MIN, MASSCART_MAX),
    masspole=(MASSPOLE_MIN, MASSPOLE_MAX),
    length=(LENGTH_MIN, LENGTH_MAX),
    force_mag=(FORCE_MAG_MIN, FORCE_MAG_MAX),
)

env.modify_params(
    delta_gravity=0,
    delta_masscart=0,
    delta_masspole=0,
    delta_length=0,
    delta_force_mag=0,
)

env.reset()


print('Reset the environment...')
print(evaluate_policy(agent, env, n_eval_episodes=100))


# for _ in range(100):
#     action = env.action_space.sample()
#     env.step(action)
#     env.render()
