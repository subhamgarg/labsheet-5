import numpy as np
exec(open("16_rl_practical.py").read())
env=GridWorld(); q=np.zeros((16,4));
for _ in range(3000):
 s=env.reset(); s=s[0]*4+s[1]; done=False
 while not done:
  a=np.random.randint(4) if np.random.rand()<.1 else np.argmax(q[s]); (r,c),reward,done=env.step(a); ns=r*4+c; q[s,a]+=.8*(reward+.95*np.max(q[ns])*(not done)-q[s,a]); s=ns
print(q)
