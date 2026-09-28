import numpy as np, matplotlib.pyplot as plt
convergence=[.05,.08,.12,.18,.30,.45,.62,.75,.82,.86]
plt.plot(convergence,marker='o'); plt.xlabel('Training block'); plt.ylabel('Average reward'); plt.title('Q-Learning Convergence'); plt.show()
