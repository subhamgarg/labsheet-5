import numpy as np, gymnasium as gym
env=gym.make("FrozenLake-v1", is_slippery=False)
q=np.zeros((env.observation_space.n,env.action_space.n)); alpha=.8; gamma=.95; eps=.1
for ep in range(2000):
    s,_=env.reset(); done=False
    while not done:
        a=env.action_space.sample() if np.random.rand()<eps else np.argmax(q[s])
        ns,r,t,tr,_=env.step(a); done=t or tr
        q[s,a]+=alpha*(r+gamma*np.max(q[ns])*(not done)-q[s,a]); s=ns
print(q); env.close()
