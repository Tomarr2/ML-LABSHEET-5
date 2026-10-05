import gymnasium as gym
import torch
import torch.nn as nn


# Define DQN model
class DQN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, state):
        return self.network(state)


# Create model
model = DQN()

# Check and load saved model
model_file = "dqn_cartpole.pth"

try:
    model.load_state_dict(
        torch.load(model_file, map_location="cpu")
    )

    print("Trained DQN model loaded successfully.")

except FileNotFoundError:
    print("ERROR: Model file not found!")
    print("First run Program 23 to train the DQN.")
    print("Then run Program 29 to save the model.")
    exit()


# Set model to evaluation mode
model.eval()

# Create environment
env = gym.make("CartPole-v1")

test_episodes = 10
total_rewards = []

for episode in range(test_episodes):

    state, info = env.reset()

    episode_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = model(state_tensor)

            action = torch.argmax(
                q_values,
                dim=1
            ).item()

        state, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward

    total_rewards.append(episode_reward)

    print(
        "Episode",
        episode + 1,
        "Reward:",
        episode_reward
    )


# Calculate average reward
average_reward = sum(total_rewards) / test_episodes

print("\nEvaluation Completed!")
print("Rewards:", total_rewards)
print("Average Reward:", average_reward)

env.close()