import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


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


# Create environment and model
env = gym.make("CartPole-v1")

model = DQN()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

memory = deque(maxlen=10000)

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

batch_size = 64
episodes = 50


# Train the DQN
for episode in range(episodes):

    state, info = env.reset()

    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):

        # Epsilon-greedy action
        if random.random() < epsilon:

            action = env.action_space.sample()

        else:

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                action = torch.argmax(
                    model(state_tensor)
                ).item()

        next_state, reward, terminated, truncated, info = env.step(action)

        memory.append(
            (
                state,
                action,
                reward,
                next_state,
                terminated
            )
        )

        state = next_state
        total_reward += reward

        # Train from replay memory
        if len(memory) >= batch_size:

            batch = random.sample(
                memory,
                batch_size
            )

            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(
                states,
                dtype=torch.float32
            )

            actions = torch.tensor(
                actions,
                dtype=torch.long
            )

            rewards = torch.tensor(
                rewards,
                dtype=torch.float32
            )

            next_states = torch.tensor(
                next_states,
                dtype=torch.float32
            )

            dones = torch.tensor(
                dones,
                dtype=torch.float32
            )

            current_q = model(states).gather(
                1,
                actions.unsqueeze(1)
            ).squeeze()

            next_q = model(next_states).max(1)[0]

            target_q = (
                rewards
                + gamma * next_q * (1 - dones)
            )

            loss = nn.MSELoss()(
                current_q,
                target_q.detach()
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    print(
        "Episode:",
        episode + 1,
        "Reward:",
        total_reward
    )


# Save trained model
torch.save(
    model.state_dict(),
    "dqn_cartpole.pth"
)

print("\nTraining completed!")
print("DQN model saved successfully!")
print("File name: dqn_cartpole.pth")

env.close()