import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)

q_table = np.zeros((env.observation_space.n,env.action_space.n))

learning_rate = 0.8
discount_factor = 0.95
epsilon = 0.1

for episode in range(5000):

    state, info = env.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated):

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, info = env.step(action)

        q_table[state, action] += learning_rate * (reward + discount_factor * np.max(q_table[next_state])- q_table[state, action])

        state = next_state

print("Learned Q-Table:")
print(np.round(q_table, 3))

env.close()