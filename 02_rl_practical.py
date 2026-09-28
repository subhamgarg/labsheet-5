import gymnasium as gym
env=gym.make("FrozenLake-v1", is_slippery=False)
obs,info=env.reset(seed=42)
print("Initial observation:",obs)
for _ in range(5):
    action=env.action_space.sample()
    obs,reward,terminated,truncated,info=env.step(action)
    print("action:",action,"state:",obs,"reward:",reward,"done:",terminated or truncated)
    if terminated or truncated: break
env.close()
