from gymnasium.envs.registration import register

register(
     id="openai_cartpole/ModifiedCartPole-v1",
     entry_point="openai_cartpole.envs:ModifiedCartPoleEnv",
     max_episode_steps=500,
)