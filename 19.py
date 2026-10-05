import gymnasium as gym
import numpy as np

# FrozenLake
env = gym.make("FrozenLake-v1", is_slippery=False)

q_table = np.zeros((env.observation_space.n,
                    env.action_space.n))

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

success = 0

for _ in range(100):
    state, info = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, info = env.step(action)

        if reward == 1:
            success += 1

env.close()

print("FrozenLake Success Rate:",
      success, "%")