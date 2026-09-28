import numpy as np, matplotlib.pyplot as plt
x=np.arange(1,101); y=20+180*(1-np.exp(-x/25))+np.random.normal(0,5,100)
plt.plot(x,y); plt.xlabel('Episode'); plt.ylabel('Reward'); plt.title('RL Learning Curve'); plt.show()
