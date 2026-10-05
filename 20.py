import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)

q_table = np.zeros((env.observation_space.n,
                    env.action_space.n))

max_changes = []

for episode in range(3000):

    state, info = env.reset()
    max_change = 0

    terminated = False
    truncated = False

    while not (terminated or truncated):

        action = (
            env.action_space.sample()
            if np.random.random() < 0.1
            else np.argmax(q_table[state])
        )

        next_state, reward, terminated, truncated, info = env.step(action)

        old_value = q_table[state, action]

        new_value = old_value + 0.8 * (
            reward + 0.95 * np.max(q_table[next_state])
            - old_value
        )

        q_table[state, action] = new_value

        max_change = max(max_change,
                         abs(new_value - old_value))

        state = next_state

    max_changes.append(max_change)

plt.plot(max_changes)
plt.xlabel("Episode")
plt.ylabel("Maximum Q-Value Change")
plt.title("Q-Learning Convergence")
plt.show()

env.close()