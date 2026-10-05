import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)

state, info = env.reset()

print("Initial State:", state)

for step in range(10):
    action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action)

    print("\nStep:", step + 1)
    print("Current State:", state)
    print("Action:", action)
    print("Reward:", reward)
    print("Next State:", next_state)
    print("Terminated:", terminated)
    print("Truncated:", truncated)

    state = next_state

    if terminated or truncated:
        break

env.close()