# This script contains code for training an RL agent with environment adversary in the loop


import time

import openai_cartpole  # Required for using the modified cartpole environment
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
from wandb.integration.sb3 import WandbCallback


from stable_baselines3.common.vec_env import DummyVecEnv

import argparse
import numpy as np

# Plotting libraries


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


class AugmentedPPO(PPO):
    """
    Augmented PPO class to allow for modification of the environment parameters
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def augment_adversary(self, Q, env, obs_space, action_space, get_action_from_index_map):

        self.Q, self.adversary_obs_space, self.adversary_action_space, self.get_action_from_index_map = Q, obs_space, action_space, get_action_from_index_map

        # Also extend the environment to allow for modification of the environment parameters
        if env is not None:
            env = maybe_make_env(env, self.verbose)
            env = self._wrap_env(env, self.verbose)

            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.n_envs = env.num_envs
            self.env = env

            # get VecNormalize object if needed
            self._vec_normalize_env = unwrap_vec_normalize(env)

    @staticmethod
    def _wrap_env(env: GymEnv, verbose: int = 0, monitor_wrapper: bool = True) -> VecEnv:
        """ "
        Wrap environment with the appropriate wrappers if needed.
        For instance, to have a vectorized environment
        or to re-order the image channels.

        :param env:
        :param verbose: Verbosity level: 0 for no output, 1 for indicating wrappers used
        :param monitor_wrapper: Whether to wrap the env in a ``Monitor`` when possible.
        :return: The wrapped environment.
        """
        if not isinstance(env, VecEnv):
            # Patch to support gym 0.21/0.26 and gymnasium
            env = _patch_env(env)
            if not is_wrapped(env, Monitor) and monitor_wrapper:
                if verbose >= 1:
                    print("Wrapping the env with a `Monitor` wrapper")
                env = Monitor(env)
            if verbose >= 1:
                print("Wrapping the env in a DummyVecEnv.")
            # type: ignore[list-item, return-value]
            env = ExtendDummyVecEnv([lambda: env])

        # Make sure that dict-spaces are not nested (not supported)
        check_for_nested_spaces(env.observation_space)

        if not is_vecenv_wrapped(env, VecTransposeImage):
            wrap_with_vectranspose = False
            if isinstance(env.observation_space, spaces.Dict):
                # If even one of the keys is a image-space in need of transpose, apply transpose
                # If the image spaces are not consistent (for instance one is channel first,
                # the other channel last), VecTransposeImage will throw an error
                for space in env.observation_space.spaces.values():
                    wrap_with_vectranspose = wrap_with_vectranspose or (
                        is_image_space(space) and not is_image_space_channels_first(
                            space)  # type: ignore[arg-type]
                    )
            else:
                wrap_with_vectranspose = is_image_space(env.observation_space) and not is_image_space_channels_first(
                    env.observation_space  # type: ignore[arg-type]
                )

            if wrap_with_vectranspose:
                if verbose >= 1:
                    print("Wrapping the env in a VecTransposeImage.")
                env = VecTransposeImage(env)

        return env

    def learn(
        self,
        total_timesteps,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        iteration = 0
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        # NOTE: the environment here has to be a modified version of the original cartpole environment

        while self.num_timesteps < total_timesteps:

            # The adversary changes start from here
            # Modify the environment according to the adversary's policy

            # Get the current environment parameters
            gravity, masscart, masspole, length, force_mag = self.env.get_params()

            state_index = get_observation_index([
                np.digitize([gravity], self.adversary_obs_space['gravity'])[
                    0] - 1,
                np.digitize([masscart], self.adversary_obs_space['masscart'])[
                    0] - 1,
                np.digitize([masspole], self.adversary_obs_space['masspole'])[
                    0] - 1,
                np.digitize([length], self.adversary_obs_space['length'])[
                    0] - 1,
                np.digitize([force_mag], self.adversary_obs_space['force_mag'])[
                    0] - 1,
            ])

            # Choose an action greedily
            action_index = np.argmax(self.Q[state_index])

            parameter_to_modify, delta = get_action_from_index(
                self.adversary_action_space, self.get_action_from_index_map, action_index)

            action = dict(delta_gravity=0, delta_masscart=0,
                          delta_masspole=0, delta_length=0, delta_force_mag=0)
            action[parameter_to_modify] = delta

            # Execute the action
            self.env.modify_params(**action)

            # Reset the internal cartpole environment using newly modified physical parameters
            self.env.reset()

            # The adversary changes end here

            continue_training = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(
                self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max(
                    (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int(
                    (self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations",
                                   iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record(
                        "rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record(
                        "rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed",
                                   int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps",
                                   self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self


class ExtendDummyVecEnv(DummyVecEnv):

    # def __init__(self, instance):
    #     self._instance = instance

    def get_params(self):
        return self.envs[0].get_params()

    def set_param_ranges(self, **kwargs):
        for env in self.envs:
            env.set_param_ranges(**kwargs)

    def modify_params(self, **kwargs):
        for env in self.envs:
            env.modify_params(**kwargs)


def main(args):

    # Initialize wandb
    # if args.log:
    wandb_run_id = wandb.init(project="env-robust-ppo",
                config=args,
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                save_code=True,  # optional
                )

    # Create the modified CartPole environment
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

    # Load adversary
    Q = np.load(args.q_table_path)

    # Regardless of whether the model already exists, train the ppo agent once again
    print('Training the PPO agent along with the adversary...')
    ppo_agent = AugmentedPPO("MlpPolicy", env, seed=args.seed, verbose=1, tensorboard_log=f"{args.log_dir}/{wandb_run_id}")
    ppo_agent.augment_adversary(Q=Q, env=env, obs_space=observation_space, action_space=action_space,
                                get_action_from_index_map=get_action_from_index_map)
    ppo_agent.learn(total_timesteps=args.robust_ppo_num_training_steps, callback=WandbCallback())
    ppo_agent.save(args.save_dir + "/env_robust_ppo_agent")
    env = gym.make('openai_cartpole/ModifiedCartPole-v1')
    env.reset()

    # Evaluate the trained PPO agent
    print('Evaluating the trained PPO agent on normal environment')
    mean_reward, std_reward = evaluate_policy(
        ppo_agent, env, n_eval_episodes=args.num_eval_episodes)

    print(
        f"Trained robust PPO agent's returns: {mean_reward} (+/-{std_reward})")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Train a PPO agent with adversarial environment modifications.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--log_dir', type=str, default='./conjugate_logs',
                        help='Directory to save logs')
    parser.add_argument('--q_table_path', type=str, default='./stored_rewards_logs/qtable_99999.npy',
                        help='Path to the Q-table')
    parser.add_argument('--save_dir', type=str, default='./models',
                        help='Directory to save the trained robust PPO agent')
    parser.add_argument('--robust_ppo_num_training_steps', type=int, default=20000,
                        help='Number of training steps for PPO')
    parser.add_argument('--num_eval_episodes', type=int, default=10,
                        help='Number of episodes to evaluate the agent at the end of the training')

    args = parser.parse_args()
    main(args)
