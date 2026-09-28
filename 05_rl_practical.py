import gymnasium as gym
env=gym.make("FrozenLake-v1", is_slippery=False)
for ep in range(5):
    s,_=env.reset(seed=ep); total=0
    done=False
    while not done:
        a=env.action_space.sample(); s,r,t,tr,_=env.step(a); total+=r; done=t or tr
    print("Episode",ep+1,"reward",total)
env.close()
