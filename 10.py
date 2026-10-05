import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)

q_table = np.zeros((env.observation_space.n,env.action_space.n))

learning_rate = 0.8
discount_factor = 0.95
epsilon = 0.1

# Training
for episode in range(5000):

    state, info = env.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated):

        action = (
            env.action_space.sample()
            if np.random.random() < epsilon
            else np.argmax(q_table[state]))

        next_state, reward, terminated, truncated, info = env.step(action)

        q_table[state, action] += learning_rate * ( reward + discount_factor * np.max(q_table[next_state])- q_table[state, action])

        state = next_state

# Evaluation
successes = 0
test_episodes = 100

for episode in range(test_episodes):

    state, info = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):

        action = np.argmax(q_table[state])

        state, reward, terminated, truncated, info = env.step(action)

        if reward == 1:
            successes += 1

print("Test Episodes:", test_episodes)
print("Successful Episodes:", successes)
print("Success Rate:", successes / test_episodes * 100, "%")

env.close()