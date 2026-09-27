import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
ALPHA = 0.8         # Learning rate
GAMMA = 0.95        # Discount factor
EPSILON_START = 1.0 # Initial exploration rate
EPSILON_MIN = 0.01  # Minimum exploration rate
EPSILON_DECAY = 0.995
NUM_EPISODES = 2000

env = gym.make('FrozenLake-v1', is_slippery=False)

# Task 7: Initialize Q-table
num_states = env.observation_space.n
num_actions = env.action_space.n
q_table = np.zeros((num_states, num_actions))

# Task 15: Epsilon-Greedy Action Selection Function
def choose_action(state, q_table, epsilon, action_space):
    if np.random.uniform(0, 1) < epsilon:
        return action_space.sample() # Exploration
    else:
        return np.argmax(q_table[state, :]) # Exploitation

# Task 6, 8, 11: Train Agent and track cumulative rewards
rewards_per_episode = []
epsilon = EPSILON_START

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = choose_action(state, q_table, epsilon, env.action_space)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Bellman Equation Update (Q-Learning Rule)
        best_next_action = np.argmax(q_table[next_state, :])
        q_table[state, action] += ALPHA * (reward + GAMMA * q_table[next_state, best_next_action] - q_table[state, action])
        
        state = next_state
        total_reward += reward
        
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    rewards_per_episode.append(total_reward)

# Task 9: Display learned Q-table
print("Learned Q-Table:")
print(np.round(q_table, 2))

# Task 10: Evaluate Trained Agent (Testing with Epsilon = 0)
test_episodes = 100
successful_runs = 0
for _ in range(test_episodes):
    state, _ = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state, :]) # Pure exploitation
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if reward == 1.0:
            successful_runs += 1

print(f"\nEvaluation Accuracy over {test_episodes} test runs: {successful_runs}% success rate.")

# Task 11: Plot cumulative rewards (Moving Average)
window_size = 50
moving_avg = np.convolve(rewards_per_episode, np.ones(window_size)/window_size, mode='valid')

plt.figure(figsize=(10, 5))
plt.plot(moving_avg)
plt.title('Q-Learning Rewards Moving Average (FrozenLake)')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.grid(True)
plt.show()

env.close()