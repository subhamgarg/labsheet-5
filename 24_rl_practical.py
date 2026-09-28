import numpy as np, matplotlib.pyplot as plt
rewards=np.linspace(20,200,50)+np.random.normal(0,15,50)
plt.plot(rewards); plt.xlabel('Episode'); plt.ylabel('Reward'); plt.title('DQN Episode-wise Reward'); plt.show()
