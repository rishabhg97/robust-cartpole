import openai_cartpole  # This registers the environment
import gymnasium as gym


# We use PPO from stable-baselines3
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import os
import argparse
import numpy as np

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sn

from tqdm import tqdm


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


def rollout_adversary(Q, rl_agent, env, num_episodes=100, len_episode=10, internal_agent_rollouts=2):
    """
    Rollout the Q-learning adversary and evaluate the internal agent.
    """

    # Start the agent in the original set of environment parameters
    env = gym.make('openai_cartpole/ModifiedCartPole-v1')
    env.reset()

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

    num_delta_per_variable = 3
    action_space_variables = action_space.keys()

    get_action_from_index_map = {
        variable_index * num_delta_per_variable + change_index: (variable_index, change_index, variable)
        for variable_index, variable in enumerate(action_space_variables)
        for change_index in range(num_delta_per_variable)
    }

    episodic_returns = []

    for _ in tqdm(range(num_episodes)):
        per_step_rewards = []
        for _ in range(len_episode):
            # Get action from Q-table
            state = env.get_params()
            state_index = get_observation_index([
                np.digitize([state[0]], observation_space['gravity'])[0] - 1,
                np.digitize([state[1]], observation_space['masscart'])[0] - 1,
                np.digitize([state[2]], observation_space['masspole'])[0] - 1,
                np.digitize([state[3]], observation_space['length'])[0] - 1,
                np.digitize([state[4]], observation_space['force_mag'])[
                    0] - 1,
            ])

            action_index = np.argmax(Q[state_index])

            parameter_to_modify, delta = get_action_from_index(
                action_space, get_action_from_index_map, action_index)

            action = dict(delta_gravity=0.0, delta_masscart=0.0,
                          delta_masspole=0.0, delta_length=0.0, delta_force_mag=0.0)
            action[parameter_to_modify] = delta
            # import ipdb; ipdb.set_trace()

            # Execute the action
            env.modify_params(**action)

            # Reset the internal cartpole environment using newly modified physical parameters
            env.reset()

            # Get reward
            internal_agent_return, _ = evaluate_policy(
                rl_agent, env, n_eval_episodes=internal_agent_rollouts)

            per_step_rewards.append(-internal_agent_return)

        episodic_returns.append(per_step_rewards)

    return episodic_returns


def main(args):

    # Create the modified CartPole environment
    env = gym.make('openai_cartpole/ModifiedCartPole-v1')
    env.reset()

    # PPO agent
    # Train a zoo of RL agents on the cartpole task and save them
    rl_agents_zoo_dir = args.rl_agents_zoo
    os.makedirs(rl_agents_zoo_dir, exist_ok=True)

    if not os.path.exists(rl_agents_zoo_dir + "/dqn_cartpole.zip"):
        print("Training DQN Model and saving")
        # Instantiate the agent
        dqn_agent = DQN("MlpPolicy", env, verbose=1)
        # Train the agent and display a progress bar
        dqn_agent.learn(total_timesteps=int(
            args.rl_agent_train_steps))
        # Save the agent
        dqn_agent.save("rl_agents_zoo/dqn_cartpole")
    else:
        print("Loading pretrained DQN Model")
        dqn_agent = DQN.load(rl_agents_zoo_dir + "/dqn_cartpole.zip")

    if not os.path.exists(rl_agents_zoo_dir + "/a2c_cartpole.zip"):
        print("Training A2C Model and saving")
        # Instantiate the agent
        a2c_agent = A2C("MlpPolicy", env, verbose=1)
        # Train the agent and display a progress bar
        a2c_agent.learn(total_timesteps=int(
            args.rl_agent_train_steps))
        # Save the agent
        a2c_agent.save("rl_agents_zoo/a2c_cartpole")
    else:
        print("Loading pretrained A2C Model")
        a2c_agent = A2C.load(rl_agents_zoo_dir + "/a2c_cartpole.zip")

    if not os.path.exists(rl_agents_zoo_dir + "/ppo_cartpole.zip"):
        print("Training PPO Model and saving")
        # Instantiate the agent
        ppo_agent = PPO("MlpPolicy", env, verbose=1)
        # Train the agent and display a progress bar
        ppo_agent.learn(total_timesteps=int(
            args.rl_agent_train_steps))
        # Save the agent
        ppo_agent.save("rl_agents_zoo/ppo_cartpole")
    else:
        print("Loading pretrained PPO Model")
        ppo_agent = PPO.load(rl_agents_zoo_dir + "/ppo_cartpole.zip")

    # Evaluate the RL agents
    # Evaluate the DQN agent
    dqn_mean_reward, dqn_std_reward = evaluate_policy(
        dqn_agent, env, n_eval_episodes=10)
    print(f"DQN mean reward: {dqn_mean_reward:.2f} +/- {dqn_std_reward:.2f}")

    # Evaluate the A2C agent
    a2c_mean_reward, a2c_std_reward = evaluate_policy(
        a2c_agent, env, n_eval_episodes=10)
    print(f"A2C mean reward: {a2c_mean_reward:.2f} +/- {a2c_std_reward:.2f}")

    # Evaluate the PPO agent
    ppo_mean_reward, ppo_std_reward = evaluate_policy(
        ppo_agent, env, n_eval_episodes=10)
    print(f"PPO mean reward: {ppo_mean_reward:.2f} +/- {ppo_std_reward:.2f}")

    # Load a trained Q-learning adversary agent
    q_table_path = args.adversary_agent_path
    Q = np.load(q_table_path)

    # Rollout the Q-learning adversary and evaluate the internal agent
    print('Testing adversary vs DQN...')
    dqn_episodic_returns = rollout_adversary(
        Q, dqn_agent, env, num_episodes=args.num_eval_episodes, len_episode=args.adversary_episode_length)
    
    # Store the results for further analysis
    np.save(args.log_dir + "/test_dqn_episodic_returns.npy", dqn_episodic_returns)
    
    print('Testing adversary vs A2C...')
    a2c_episodic_returns = rollout_adversary(
        Q, a2c_agent, env, num_episodes=args.num_eval_episodes, len_episode=args.adversary_episode_length)
    np.save(args.log_dir + "/test_a2c_episodic_returns.npy", a2c_episodic_returns)

    print('Testing adversary vs PPO...')
    ppo_episodic_returns = rollout_adversary(
        Q, ppo_agent, env, num_episodes=args.num_eval_episodes, len_episode=args.adversary_episode_length)
    np.save(args.log_dir + "/test_ppo_episodic_returns.npy", ppo_episodic_returns)



    # Load the adversary returns against different RL agents
    dqn_agent_returns = np.load(args.log_dir + '/test_dqn_episodic_returns.npy')
    a2c_agent_returns = np.load(args.log_dir + '/test_a2c_episodic_returns.npy')
    ppo_agent_returns = np.load(args.log_dir + '/test_ppo_episodic_returns.npy')

    # Convert the adversary rewards to internal agent returns and normalise
    dqn_agent_returns = -dqn_agent_returns
    a2c_agent_returns = -a2c_agent_returns
    ppo_agent_returns = -ppo_agent_returns


    # Take mean and std of the returns
    dqn_agent_returns_mean = np.mean(dqn_agent_returns)
    dqn_agent_returns_std = np.std(dqn_agent_returns)

    a2c_agent_returns_mean = np.mean(a2c_agent_returns)
    a2c_agent_returns_std = np.std(a2c_agent_returns)

    ppo_agent_returns_mean = np.mean(ppo_agent_returns)
    ppo_agent_returns_std = np.std(ppo_agent_returns)

    # Original reward level
    rl_agent_zoo_dir = './rl_agents_zoo/'
    dqn_agent = DQN.load(rl_agent_zoo_dir + 'dqn_cartpole')
    a2c_agent = A2C.load(rl_agent_zoo_dir + 'a2c_cartpole')
    ppo_agent = PPO.load(rl_agent_zoo_dir + 'ppo_cartpole')


    env = gym.make('CartPole-v1')
    dqn_orig_return, dqn_orig_return_std = evaluate_policy(dqn_agent, env, n_eval_episodes=100)
    a2c_orig_return, a2c_orig_return_std = evaluate_policy(a2c_agent, env, n_eval_episodes=100)
    ppo_orig_return, ppo_orig_return_std = evaluate_policy(ppo_agent, env, n_eval_episodes=100)

    # Normalise
    dqn_orig_return = dqn_orig_return
    a2c_orig_return = a2c_orig_return
    ppo_orig_return = ppo_orig_return


    print('Original Returns:')
    print('DQN:', dqn_orig_return, '±', dqn_orig_return_std)
    print('A2C:', a2c_orig_return, '±', a2c_orig_return_std)
    print('PPO:', ppo_orig_return, '±', ppo_orig_return_std)

    print('Returns when adversary is present:')
    print('DQN:', dqn_agent_returns_mean, '±', dqn_agent_returns_std)
    print('A2C:', a2c_agent_returns_mean, '±', a2c_agent_returns_std)
    print('PPO:', ppo_agent_returns_mean, '±', ppo_agent_returns_std)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Train a PPO agent with adversarial environment modifications.')
    parser.add_argument('--log', action='store_true',
                        help='Enable logging to files')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory to save logs')
    parser.add_argument('--save_dir', type=str, default='./models',
                        help='Directory to save models')
    parser.add_argument('--adversary_episode_length', type=int, default=10,
                        help='Maximum number of steps in an adversary episode')
    parser.add_argument('--num_eval_episodes', type=int, default=25,
                        help='Number of episodes to evaluate the adversary')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for Q-learning')
    parser.add_argument('--discount_factor', type=float, default=0.99,
                        help='Discount factor for Q-learning')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='Exploration rate for Q-learning')
    parser.add_argument('--num_q_learning_episodes', type=int, default=100,
                        help='Number of episodes for Q-learning')
    parser.add_argument('--init_scale', type=float, default=0.01,
                        help='Initial scale for Q-table')
    parser.add_argument('--rl_agent_train_steps', type=int, default=10000,
                        help='Number of training steps for RL agents')
    parser.add_argument('--num_policy_eval_episodes', type=int, default=10,
                        help='Number of episodes to evaluate the PPO agent')
    parser.add_argument('--rl_agents_zoo', type=str, default='./rl_agents_zoo',
                        help='Directory to save RL agents')
    parser.add_argument('--adversary_agent_path', type=str, default='./stored_rewards_logs/qtable_99999.npy',
                        help='Path to the trained Q-learning adversary agent')
    
    args = parser.parse_args()
    main(args) 