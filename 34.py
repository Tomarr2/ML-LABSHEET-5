import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

learning_rates = [0.1, 0.5, 0.8]
gamma_values = [0.5, 0.9, 0.99]

results = []

for lr in learning_rates:

    for gamma in gamma_values:

        env = gym.make(
            "FrozenLake-v1",
            is_slippery=False
        )

        q_table = np.zeros(
            (env.observation_space.n,
             env.action_space.n)
        )

        total_success = 0

        for episode in range(1000):

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

                q_table[state, action] += lr * (
                    reward +
                    gamma * np.max(q_table[next_state])
                    - q_table[state, action]
                )

                state = next_state

                if reward == 1:
                    total_success += 1

        results.append(
            (lr, gamma, total_success)
        )

        env.close()

for result in results:
    print(
        "Learning Rate:", result[0],
        "Gamma:", result[1],
        "Successes:", result[2]
    )

plt.bar(
    range(len(results)),
    [result[2] for result in results]
)

plt.xlabel("Hyperparameter Combination")
plt.ylabel("Successful Episodes")
plt.title("Impact of Hyperparameters")

plt.show()