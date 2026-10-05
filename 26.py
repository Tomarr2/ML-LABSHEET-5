import matplotlib.pyplot as plt

q_learning_rewards = [0.55, 0.60, 0.65, 0.72, 0.80]
dqn_rewards = [20, 30, 45, 70, 90]

plt.plot(q_learning_rewards, label="Q-Learning")
plt.plot(dqn_rewards, label="DQN")

plt.xlabel("Training Stage")
plt.ylabel("Reward")
plt.title("Q-Learning vs DQN")

plt.legend()
plt.show()