import matplotlib.pyplot as plt
q=[10,35,70,110,145]; d=[8,28,60,105,160]
plt.plot(q,label='Q-Learning'); plt.plot(d,label='DQN'); plt.legend(); plt.xlabel('Training stage'); plt.ylabel('Cumulative reward'); plt.title('RL Algorithm Reward Comparison'); plt.show()
