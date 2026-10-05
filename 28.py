import torch
import torch.nn as nn

class DQN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)


online_network = DQN()
target_network = DQN()

# Copy online network parameters
target_network.load_state_dict(
    online_network.state_dict()
)

target_network.eval()

print("Online Network:")
print(online_network)

print("\nTarget Network:")
print(target_network)

print("\nTarget network initialized successfully.")