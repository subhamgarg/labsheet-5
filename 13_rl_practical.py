import numpy as np, gymnasium as gym
for gamma in [.5,.8,.95,.99]:
 env=gym.make("FrozenLake-v1",is_slippery=False); q=np.zeros((16,4)); wins=0
 for _ in range(1000):
  s,_=env.reset(); done=False
  while not done:
   a=np.random.randint(4) if np.random.rand()<.1 else np.argmax(q[s]); ns,r,t,tr,_=env.step(a); done=t or tr; q[s,a]+=.8*(r+gamma*np.max(q[ns])*(not done)-q[s,a]); s=ns; wins+=r
 print("gamma",gamma,"reward",wins); env.close()
