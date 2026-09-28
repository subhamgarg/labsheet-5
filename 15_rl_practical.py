import numpy as np
def epsilon_greedy(q_row,epsilon):
 return np.random.randint(len(q_row)) if np.random.rand()<epsilon else int(np.argmax(q_row))
q=np.array([.2,.8,.4,.1]); print("Selected action:",epsilon_greedy(q,.1))
