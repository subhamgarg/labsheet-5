import numpy as np
class GridWorld:
 def __init__(self,size=4): self.size=size; self.start=(0,0); self.goal=(size-1,size-1)
 def reset(self): self.state=self.start; return self.state
 def step(self,a):
  r,c=self.state; r=max(0,min(self.size-1,r+[-1,1,0,0][a])); c=max(0,min(self.size-1,c+[0,0,-1,1][a])); self.state=(r,c); done=self.state==self.goal; return self.state,1 if done else -0.01,done
env=GridWorld(); print("Start:",env.reset()); print("Step:",env.step(3))
