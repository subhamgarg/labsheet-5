# Compare learning rates
import numpy as np, gymnasium as gym
for alpha in [.1,.5,.8,1.0]:
 env=gym.make("FrozenLake-v1",is_slippery=False); q=np.zeros((16,4)); total=0
 for _ in range(1000):
  s,_=env.reset(); done=False
  while not done:
   a=np.random.randint(4) if np.random.rand()<.1 else np.argmax(q[s]); ns,r,t,tr,_=env.step(a); done=t or tr; q[s,a]+=alpha*(r+.95*np.max(q[ns])*(not done)-q[s,a]); s=ns; total+=r
 print("alpha",alpha,"rewards",total); env.close()
