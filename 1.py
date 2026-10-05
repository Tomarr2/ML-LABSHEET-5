import gymnasium as gym

print("Gymnasium version:", gym.__version__)

env = gym.make("FrozenLake-v1")

print("Gymnasium configured successfully!")
print("Environment:", env)