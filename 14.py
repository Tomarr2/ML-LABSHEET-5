import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

epsilon_values = [0.01, 0.1, 0.3, 0.5]

for epsilon in epsilon_values:

    env = gym.make("FrozenLake-v1", is_slippery=False)

    q_table = np.zeros((env.observation_space.n,
                        env.action_space.n))

    rewards = []

    for episode in range(2000):

        state, info = env.reset()
        total_reward = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):

            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, info = env.step(action)

            q_table[state, action] += 0.8 * (
                reward + 0.95 * np.max(q_table[next_state])
                - q_table[state, action]
            )

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

    average = np.convolve(
        rewards, np.ones(100) / 100, mode="valid"
    )

    plt.plot(average, label=f"Epsilon={epsilon}")

    env.close()

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Exploration vs Exploitation")
plt.legend()
plt.show()