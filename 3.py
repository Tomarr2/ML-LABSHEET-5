import gymnasium as gym

env = gym.make("FrozenLake-v1")

print("Observation Space:")
print(env.observation_space)

print("\nAction Space:")
print(env.action_space)

print("\nNumber of States:", env.observation_space.n)
print("Number of Actions:", env.action_space.n)

env.close()