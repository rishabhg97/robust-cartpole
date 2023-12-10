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


To train an environment adversary, run the following command:

```
python main.py --log --seed 42 --log_dir ./logs --save_dir ./models --adversary_episode_length 20 --num_eval_episodes 25 --learning_rate 0.1 --discount_factor 0.99 --epsilon 0.05 --num_q_learning_episodes 1000 --init_scale 0.01 --ppo_train_steps 10000 --num_policy_eval_episodes 2
```
