import numpy as np
import matplotlib.pyplot as plt

rewards = np.random.randint(
    10, 100, 100
)

window_size = 10

moving_average = np.convolve(
    rewards,
    np.ones(window_size) / window_size,
    mode="valid"
)

plt.plot(rewards, alpha=0.4, label="Reward")
plt.plot(
    range(window_size - 1, len(rewards)),
    moving_average,
    label="Learning Curve"
)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("RL Agent Learning Curve")

plt.legend()
plt.show()