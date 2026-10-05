import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

learning_rates = [0.1, 0.5, 0.8, 1.0]

for lr in learning_rates:

    env = gym.make("FrozenLake-v1", is_slippery=False)

    q_table = np.zeros((env.observation_space.n,env.action_space.n))

    rewards = []

    for episode in range(2000):

        state, info = env.reset()
        total_reward = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):

            action = (
                env.action_space.sample()
                if np.random.random() < 0.1
                else np.argmax(q_table[state])
            )

            next_state, reward, terminated, truncated, info = env.step(action)

            q_table[state, action] += lr * (reward + 0.95 * np.max(q_table[next_state])- q_table[state, action])

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

    moving_average = np.convolve(rewards, np.ones(100) / 100, mode="valid")

    plt.plot(moving_average, label=f"LR={lr}")

    env.close()

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Effect of Learning Rate")
plt.legend()
plt.show()