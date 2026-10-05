import torch
import torch.nn as nn

class DQN(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, state):
        return self.network(state)


input_size = 4
output_size = 2

model = DQN(input_size, output_size)

print(model)