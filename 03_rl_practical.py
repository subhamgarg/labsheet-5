import gymnasium as gym
env=gym.make("FrozenLake-v1", is_slippery=False)
print("Observation space:",env.observation_space)
print("Action space:",env.action_space)
env.close()
