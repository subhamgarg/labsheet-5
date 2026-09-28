import numpy as np, matplotlib.pyplot as plt, gymnasium as gym
env=gym.make("FrozenLake-v1",is_slippery=False); q=np.zeros((16,4)); rewards=[]
for ep in range(1000):
 s,_=env.reset(); total=0; done=False
 while not done:
  a=env.action_space.sample() if np.random.rand()<.1 else np.argmax(q[s]); ns,r,t,tr,_=env.step(a); done=t or tr; q[s,a]+=.8*(r+.95*np.max(q[ns])*(not done)-q[s,a]); s=ns; total+=r
 rewards.append(total)
plt.plot(np.cumsum(rewards)); plt.xlabel('Episode'); plt.ylabel('Cumulative Reward'); plt.title('Q-Learning Cumulative Rewards'); plt.show()
