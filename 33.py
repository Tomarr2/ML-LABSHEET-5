import time
import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)

q_table = np.zeros(
    (env.observation_space.n,
     env.action_space.n)
)

start_time = time.time()

for episode in range(3000):

    state, info = env.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated):

        action = (
            env.action_space.sample()
            if np.random.random() < 0.1
            else np.argmax(q_table[state])
        )

        next_state, reward, terminated, truncated, info = env.step(action)

        q_table[state, action] += 0.8 * (
            reward + 0.95 * np.max(q_table[next_state])
            - q_table[state, action]
        )

        state = next_state

q_learning_time = time.time() - start_time

print("Q-Learning Training Time:",
      round(q_learning_time, 4),
      "seconds")

env.close()

print("\nDQN training generally requires more computation because")
print("it trains a neural network instead of a simple Q-table.")