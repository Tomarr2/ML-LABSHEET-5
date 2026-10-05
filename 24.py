import matplotlib.pyplot as plt

# Example rewards collected during training
episode_rewards = [
    20, 25, 18, 35, 40, 45, 50,
    55, 60, 70, 80, 90, 100
]

plt.plot(episode_rewards)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Episode-wise Rewards")

plt.show()