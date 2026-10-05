import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)

q_table = np.zeros((env.observation_space.n,
                    env.action_space.n))

print("Initial Q-Table:")
print(q_table)

state, info = env.reset()

action = env.action_space.sample()

next_state, reward, terminated, truncated, info = env.step(action)

learning_rate = 0.8
discount_factor = 0.95

old_value = q_table[state, action]

new_value = old_value + learning_rate * (
    reward + discount_factor * np.max(q_table[next_state])
    - old_value
)

q_table[state, action] = new_value

print("\nUpdated Q-Table:")
print(q_table)

env.close()