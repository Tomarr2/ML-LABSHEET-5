import gymnasium as gym
import torch
import torch.nn as nn
import os


# Define DQN
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


# Check model file
model_file = "dqn_cartpole.pth"

if not os.path.exists(model_file):

    print("Error: dqn_cartpole.pth not found!")
    print("Please run Program 29 first.")

else:

    # Create model
    model = DQN()

    # Load saved model
    model.load_state_dict(
        torch.load(
            model_file,
            map_location="cpu"
        )
    )

    model.eval()

    print("DQN model loaded successfully!")

    # Create environment
    env = gym.make("CartPole-v1")

    test_episodes = 10
    rewards = []

    # Testing
    for episode in range(test_episodes):

        state, info = env.reset()

        total_reward = 0
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

            total_reward += reward

        rewards.append(total_reward)

        print(
            "Test Episode:",
            episode + 1,
            "Reward:",
            total_reward
        )

    # Average reward
    average_reward = sum(rewards) / len(rewards)

    print("\nTesting completed!")
    print("Rewards:", rewards)
    print("Average Reward:", average_reward)

    env.close()