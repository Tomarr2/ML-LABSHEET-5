import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)

q_table = np.zeros((env.observation_space.n,env.action_space.n))

learning_rate = 0.8
discount_factor = 0.95
epsilon = 0.1

rewards = []

for episode in range(3000):

    state, info = env.reset()
    total_reward = 0

    terminated = False
    truncated = False

    while not (terminated or truncated):

        action = (
            env.action_space.sample()
            if np.random.random() < epsilon
            else np.argmax(q_table[state])
        )

        next_state, reward, terminated, truncated, info = env.step(action)

        q_table[state, action] += learning_rate * (reward + discount_factor * np.max(q_table[next_state])- q_table[state, action])

        state = next_state
        total_reward += reward

    rewards.append(total_reward)

cumulative_rewards = np.cumsum(rewards)

plt.plot(cumulative_rewards)
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Rewards during Q-Learning")
plt.show()

env.close()