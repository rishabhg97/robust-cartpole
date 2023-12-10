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


# TODO: change the 3 base to any user-specified base
def get_action_index(variable_index, change_index):
    """
    Convert a variable index and a change index to a single action index.
    There are 5 variables and 3 changes, so the total number of actions is 15.
    """
    return variable_index * 3 + change_index


def get_action_from_index(action_space, get_action_from_index_map, action_index):
    """
    Convert a single action index back to the variable index and change index.
    """
    variable_index, index, variable_name = get_action_from_index_map[action_index]
    delta_value = action_space[variable_name][index]

    return variable_name, delta_value


def get_observation_index(observation):
    """
    Convert an observation (a list of indices representing the state of each variable)
    to a single index using base-3 arithmetic.
    """
    base = 3
    observation_index = 0
    for i, obs in enumerate(observation):
        observation_index += obs * (base ** i)
    return observation_index


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


def main(args):

    # Create the modified CartPole environment
    env = gym.make('openai_cartpole/ModifiedCartPole-v1')
    env.reset()

    # Load the PPO agent
    ppo_filename = args.ppo_agent_path
    ppo_agent = PPO.load(ppo_filename)

    # Evaluate the PPO agent on the original environment
    mean_reward, std_reward = evaluate_policy(
        ppo_agent, env, n_eval_episodes=100)

    print(
        f"Trained PPO agent's returns on original environment: {mean_reward} (+/-{std_reward})")

    # Load Q-learning adversary agent
    Q = np.load(args.q_table_path)

    # Keep the trained ppo agent on cartpole task frozen throughout the adversary training

    # Original physical parameters of the environment. These constitute the first observation for the adversary.
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

    # Q-learning algorithm
    action_space_variables = action_space.keys()
    num_delta_per_variable = 3

    get_action_from_index_map = {
        variable_index * num_delta_per_variable + change_index: (variable_index, change_index, variable)
        for variable_index, variable in enumerate(action_space_variables)
        for change_index in range(num_delta_per_variable)
    }

    # Evaluation of the Q-learning adversary

    num_eval_episodes = args.num_trajectories
    trajectories = []

    for eval_episode in tqdm(range(num_eval_episodes)):

        trajectory = []

        # Randomly choose the initial set of environment parameters
        env.modify_params(
            delta_gravity=np.random.choice(action_space['delta_gravity']),
            delta_masscart=np.random.choice(action_space['delta_masscart']),
            delta_masspole=np.random.choice(action_space['delta_masspole']),
            delta_length=np.random.choice(action_space['delta_length']),
            delta_force_mag=np.random.choice(action_space['delta_force_mag']),
        )

        for _ in range(args.adversary_episode_length):

            # Initialize all actions to 'no change'
            action = {param: 0 for param in action_space.keys()}
            action_index = -1

            # Get the current environment parameters
            gravity, masscart, masspole, length, force_mag = env.get_params()

            state_index = get_observation_index([
                np.digitize([gravity], observation_space['gravity'])[0] - 1,
                np.digitize([masscart], observation_space['masscart'])[0] - 1,
                np.digitize([masspole], observation_space['masspole'])[0] - 1,
                np.digitize([length], observation_space['length'])[0] - 1,
                np.digitize([force_mag], observation_space['force_mag'])[
                    0] - 1,
            ])

            curr_s = state_index

            # Choose an action greedily
            action_index = np.argmax(Q[state_index])

            parameter_to_modify, delta = get_action_from_index(
                action_space, get_action_from_index_map, action_index)
            action[parameter_to_modify] = delta

            # Record the current action
            curr_a = action_index
            
            # Execute the action
            env.modify_params(**action)

            # Get the next state
            gravity, masscart, masspole, length, force_mag = env.get_params()

            next_state_index = get_observation_index([
                np.digitize([gravity], observation_space['gravity'])[0] - 1,
                np.digitize([masscart], observation_space['masscart'])[0] - 1,
                np.digitize([masspole], observation_space['masspole'])[0] - 1,
                np.digitize([length], observation_space['length'])[0] - 1,
                np.digitize([force_mag], observation_space['force_mag'])[
                    0] - 1,
            ])

            # Record the next state
            next_s = next_state_index


            # Record the transition
            trajectory.append([curr_s, curr_a, next_s])
        
        trajectories.append(trajectory)
    
    # Dump the trajectories
    np.save(os.path.join(args.out_dir, 'trajectories.npy'), trajectories)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Train a PPO agent with adversarial environment modifications.')
    parser.add_argument('--q-table-path', type=str, default='./logs/qtable_999.npy',
                        help='Path to the Q-table')
    parser.add_argument('--ppo-agent-path', type=str, default='./ppo_agent.zip',
                        help='Path to the PPO agent')
    parser.add_argument('--out-dir', type=str, default='./logs',
                        help='Path to the output directory')
    parser.add_argument('--num_trajectories', type=int, default=5,
                        help='Number of trajectories to plot')
    parser.add_argument('--adversary_episode_length', type=int, default=10,
                        help='Number of steps per episode')

    args = parser.parse_args()
    main(args)
