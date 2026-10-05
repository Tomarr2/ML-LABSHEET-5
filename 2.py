import gymnasium as gym

env = gym.make("CartPole-v1")

observation, info = env.reset()

print("Initial Observation:")
print(observation)

for step in range(5):
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    print("\nStep:", step + 1)
    print("Action:", action)
    print("Reward:", reward)
    print("Observation:", observation)

    if terminated or truncated:
        observation, info = env.reset()

env.close()