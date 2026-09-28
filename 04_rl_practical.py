import gymnasium as gym
env=gym.make("FrozenLake-v1", is_slippery=False)
obs,_=env.reset(seed=1)
for i in range(10):
    action=env.action_space.sample(); next_obs,reward,terminated,truncated,_=env.step(action)
    print(f"state={obs}, action={action}, reward={reward}, terminated={terminated}, truncated={truncated}")
    obs=next_obs
    if terminated or truncated: break
env.close()
