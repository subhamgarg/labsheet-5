import numpy as np, gymnasium as gym
for eps in [.01,.1,.3,.7]:
 env=gym.make("FrozenLake-v1",is_slippery=False); q=np.zeros((16,4)); total=0
 for _ in range(1000):
  s,_=env.reset(); done=False
  while not done:
   a=np.random.randint(4) if np.random.rand()<eps else np.argmax(q[s]); ns,r,t,tr,_=env.step(a); done=t or tr; q[s,a]+=.8*(r+.95*np.max(q[ns])*(not done)-q[s,a]); s=ns; total+=r
 print("epsilon",eps,"reward",total); env.close()
