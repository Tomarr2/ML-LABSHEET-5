import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1")

q_table = np.random.random(
    (env.observation_space.n,
     env.action_space.n)
)

epsilon = 0.2

state, info = env.reset()

if np.random.random() < epsilon:
    action = env.action_space.sample()
    print("Exploration selected.")
else:
    action = np.argmax(q_table[state])
    print("Exploitation selected.")

print("State:", state)
print("Selected Action:", action)

env.close()