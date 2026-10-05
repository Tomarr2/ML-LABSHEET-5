import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)

state, info = env.reset()

total_reward = 0

for step in range(20):
    action = env.action_space.sample()

    state, reward, terminated, truncated, info = env.step(action)

    print("Step:", step + 1,
          "Action:", action,
          "State:", state,
          "Reward:", reward)

    total_reward += reward

    if terminated or truncated:
        break

print("\nTotal Reward:", total_reward)

env.close()