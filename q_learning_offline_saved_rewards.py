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

    # Initialize wandb
    if args.log:
        wandb.init(project="ppo-qlearn-cartpole", config=args)

    # Create the modified CartPole environment
    env = gym.make('openai_cartpole/ModifiedCartPole-v1')
    env.reset()

    # Q-learning adversary agent

    # Collect the stored rewards
    stored_rewards = np.load(args.stored_rewards_file)

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

    # Parameters for Q-learning
    learning_rate = args.learning_rate
    discount_factor = args.discount_factor
    epsilon = args.epsilon  # exploration rate
    num_episodes = args.num_q_learning_episodes
    init_scale = args.init_scale

    # Initialize Q-table with random noise
    statespacesize = np.prod([len(observation_space[key])
                              for key in observation_space])
    actionspacesize = 15
    Q = np.random.randn(statespacesize, actionspacesize) * init_scale
    print(f"Q-table shape: {Q.shape}")

    adv_rewards = []

    for adversary_episode_id in tqdm(range(num_episodes)):

        # Take the initial observation

        # Reset the environment parameters to randomly chosen values
        env.modify_params(
            delta_gravity=np.random.choice(action_space['delta_gravity']),
            delta_masscart=np.random.choice(action_space['delta_masscart']),
            delta_masspole=np.random.choice(action_space['delta_masspole']),
            delta_length=np.random.choice(action_space['delta_length']),
            delta_force_mag=np.random.choice(action_space['delta_force_mag']),
        )

        gravity, masscart, masspole, length, force_mag = env.get_params()

        # Find the index of the initial observation in the Q-table
        state_index = get_observation_index([
            np.digitize([gravity], observation_space['gravity'])[0] - 1,
            np.digitize([masscart], observation_space['masscart'])[0] - 1,
            np.digitize([masspole], observation_space['masspole'])[0] - 1,
            np.digitize([length], observation_space['length'])[0] - 1,
            np.digitize([force_mag], observation_space['force_mag'])[0] - 1,
        ])

        # Start the adversary episode
        for _ in range(args.adversary_episode_length):

            # Initialize all actions to 'no change'
            action = {param: 0 for param in action_space.keys()}
            action_index = -1

            # Choose an action using epsilon-greedy policy
            if np.random.rand() < epsilon:
                # Choose an action randomly
                action_index = np.random.randint(actionspacesize)
                parameter_to_modify, action_value = get_action_from_index(action_space,
                                                                          get_action_from_index_map,
                                                                          action_index)

                # Create an action dictionary with the chosen modification
                action[parameter_to_modify] = action_value

            else:
                # Choose an action greedily
                action_index = np.argmax(Q[state_index])
                parameter_to_modify, delta = get_action_from_index(
                    action_space, get_action_from_index_map, action_index)
                action[parameter_to_modify] = delta

            # Execute the action

            # Modify the environment parameters
            env.modify_params(**action)

            # Reset the internal cartpole environment using newly modified physical parameters
            env.reset()

            # The reward for the adversary is negative of the PPO agent's return
            # internal_agent_return, _ = evaluate_policy(
            #     ppo_agent, env, n_eval_episodes=args.num_policy_eval_episodes)
            # adversary_reward = -internal_agent_return

            # Take the reward from the stored rewards
            adversary_reward = -stored_rewards[state_index]

            if args.log:
                wandb.log({"adversary_reward": adversary_reward})

            # Next state
            next_g, next_m, next_mp, next_l, next_fmg = env.get_params()
            next_state_index = get_observation_index([
                np.digitize([next_g], observation_space['gravity'])[0] - 1,
                np.digitize([next_m], observation_space['masscart'])[0] - 1,
                np.digitize([next_mp], observation_space['masspole'])[0] - 1,
                np.digitize([next_l], observation_space['length'])[0] - 1,
                np.digitize([next_fmg], observation_space['force_mag'])[0] - 1,
            ])

            # Update the Q-table using Q-learning TD update
            Q[state_index, action_index] = Q[state_index, action_index] + learning_rate * \
                (adversary_reward + discount_factor *
                 np.max(Q[next_state_index]) - Q[state_index, action_index])
            adv_rewards.append(adversary_reward)

        # Save the Q-table
        if (adversary_episode_id + 1) % args.save_q_table_freq == 0:
            np.save(args.log_dir + "/qtable_" +
                    str(adversary_episode_id) + ".npy", Q)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Here we train a Q-learning adversary on the rewards stored by running PPO agent on modified versions of the cartpole.')
    parser.add_argument('--log', action='store_true',
                        help='Enable logging to files')
    parser.add_argument('--log_dir', type=str, default='./stored_rewards_logs/',
                        help='Directory to save Q-tables')
    parser.add_argument('--save_q_table_freq', type=int, default=1000,
                        help='Frequency with which to save Q-tables')
    parser.add_argument('--adversary_episode_length', type=int, default=10,
                        help='Maximum number of steps in an adversary episode')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for Q-learning')
    parser.add_argument('--discount_factor', type=float, default=0.99,
                        help='Discount factor for Q-learning')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='Exploration rate for Q-learning')
    parser.add_argument('--num_q_learning_episodes', type=int, default=100000,
                        help='Number of episodes for Q-learning')
    parser.add_argument('--init_scale', type=float, default=0.01,
                        help='Initial scale for Q-table')
    parser.add_argument('--stored_rewards_file', type=str,
                        default='every_state_returns_ppo.npy',)
    args = parser.parse_args()
    
    main(args)
