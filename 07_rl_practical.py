import numpy as np
n_states,n_actions=16,4
q=np.zeros((n_states,n_actions)); print("Initial Q-table:
",q)
state,action,next_state,reward=0,1,4,0
alpha,gamma=.8,.95
q[state,action]+=alpha*(reward+gamma*np.max(q[next_state])-q[state,action])
print("Updated Q-table:
",q)
