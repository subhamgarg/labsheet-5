import numpy as np, gymnasium as gym
env=gym.make("FrozenLake-v1",is_slippery=False); q=np.zeros((16,4)); alpha=.8; gamma=.95; eps=.1
for _ in range(3000):
 s,_=env.reset(); done=False
 while not done:
  a=np.random.randint(4) if np.random.rand()<eps else np.argmax(q[s]); ns,r,t,tr,_=env.step(a); done=t or tr; q[s,a]+=alpha*(r+gamma*np.max(q[ns])*(not done)-q[s,a]); s=ns
wins=0
for _ in range(100):
 s,_=env.reset(); done=False
 while not done:
  a=np.argmax(q[s]); s,r,t,tr,_=env.step(a); done=t or tr; wins+=r
print("Successes out of 100:",int(wins)); env.close()
