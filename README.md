# robustness-cartpole
Robustness in CartPole- 2D


## Installation Instructions

This code is tested on Python 3.9+. To install the dependencies, run the following command in the terminal:

```
# Create conda environment
conda create -n robustness-cartpole python=3.9 -y
conda activate robustness-cartpole

# Install dependencies
pip install gymnasium pygame stable-baselines3 matplotlib seaborn tqdm wandb

# Install the modified environment
pip install -e openai_cartpole/
```

To check if the installation has completed successfully, please run the following command:

```
python test_env.py
```

If the installation is successful, you should see a window pop up with a cartpole environment. Intially, the cartpole will run for 100 steps at normal length of the pole. After that you should 
be able to see cartpole with a longer pole for another 100 steps.

## Training Instructions


### Training Q-learning adversary

To train the environment adversary using Q-learning, you need to run the following commands:
    
```
# First train a PPO agent to provide rewards to the adversary
# This command will train the PPO agent using stable-baselines3 and save the agent in the specified path
python train_ppo.py --num_timesteps 20000  --save_path ./ppo_agent.zip --seed 0

# Gather rewards from the PPO agent
# This command collects the returns of the PPO agent for given number of evaluation episodes and saves it in the specified path as a numpy array
python gather_every_state_returns.py --ppo-agent-path ./ppo_agent.zip --out-dir ./models --num_eval_episodes 100

# Train the adversary using Q-learning over offline saved rewards. The final Q-table gets stored in the log directory specified
python q_learning_offline_saved_rewards.py --stored_rewards_file every_state_returns_ppo.npy --num_q_learning_episodes 100000 --adversary_episode_length 10 --learning_rate 0.1 --discount_factor 0.99 --epsilon 0.05  --init_scale 0.01  --log --log_dir ./stored_rewards_logs/ --save_q_table_freq 1000
```

Alternatively, if you want to train an adversary using PPO agent's online rollouts, you can run the following command  (warning: takes longer time to train):

```
python main.py --log --seed 42 --log_dir ./logs --save_dir ./models --adversary_episode_length 20 --num_eval_episodes 25 --learning_rate 0.1 --discount_factor 0.99 --epsilon 0.05 --num_q_learning_episodes 1000 --init_scale 0.01 --ppo_train_steps 10000 --num_policy_eval_episodes 2
```

### Training Robust PPO agent with environment adversary

```
python conjuagate_env_adversary_rl_agent.py --log_dir ./conjugate_logs --q_table_path ./stored_rewards_logs/qtable_99999.npy --save_dir ./models --robust_ppo_num_training_steps 20000 --num_eval_episodes 10
```

## Evaluation Instructions

To evaluate the trained adversary against unseen DQN, A2C and PPO agents, run the following command:
```
python evaluate_adversary_against_rl_agents.py --log --seed 0 --log_dir ./logs --save_dir ./models --adversary_episode_length 10 --num_eval_episodes 25 --learning_rate 0.1 --discount_factor 0.99 --epsilon 0.05 --num_q_learning_episodes 100 --init_scale 0.01 --rl_agent_train_steps 10000 --num_policy_eval_episodes 10 --rl_agents_zoo ./rl_agents_zoo --adversary_agent_path ./stored_rewards_logs/qtable_99999.npy
```