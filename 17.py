import numpy as np

grid_size = 5
states = grid_size * grid_size
actions = 4

q_table = np.zeros((states, actions))

learning_rate = 0.8
gamma = 0.95
epsilon = 0.1

def state_number(row, column):
    return row * grid_size + column

for episode in range(2000):

    row, column = 0, 0

    for step in range(100):

        current_state = state_number(row, column)

        if np.random.random() < epsilon:
            action = np.random.randint(actions)
        else:
            action = np.argmax(q_table[current_state])

        new_row, new_column = row, column

        if action == 0:
            new_row -= 1
        elif action == 1:
            new_row += 1
        elif action == 2:
            new_column -= 1
        elif action == 3:
            new_column += 1

        if not (0 <= new_row < grid_size and
                0 <= new_column < grid_size):
            new_row, new_column = row, column
            reward = -1
        elif (new_row, new_column) == (4, 4):
            reward = 10
        else:
            reward = -0.1

        new_state = state_number(new_row, new_column)

        q_table[current_state, action] += learning_rate * (
            reward + gamma * np.max(q_table[new_state])
            - q_table[current_state, action]
        )

        row, column = new_row, new_column

        if (row, column) == (4, 4):
            break

print("Training completed.")
print("Learned Q-Table:")
print(np.round(q_table, 2))