import numpy as np
import matplotlib.pyplot as plt

q_learning = np.array(
    [0, 0, 1, 1, 2, 3, 4, 5]
)

dqn = np.array(
    [10, 20, 35, 50, 70, 85, 95, 100]
)

q_cumulative = np.cumsum(q_learning)
dqn_cumulative = np.cumsum(dqn)

plt.plot(q_cumulative, label="Q-Learning")
plt.plot(dqn_cumulative, label="DQN")

plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward Comparison")

plt.legend()
plt.show()