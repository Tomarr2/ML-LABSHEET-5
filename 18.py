import numpy as np

# Example trained Q-table
q_table = np.random.random((25, 4))

grid_size = 5
row, column = 0, 0

path = [(row, column)]

for step in range(25):

    state = row * grid_size + column
    action = np.argmax(q_table[state])

    if action == 0:
        row -= 1
    elif action == 1:
        row += 1
    elif action == 2:
        column -= 1
    elif action == 3:
        column += 1

    row = max(0, min(row, 4))
    column = max(0, min(column, 4))

    path.append((row, column))

    if (row, column) == (4, 4):
        break

print("Optimal Path:")
print(path)

grid = np.full((5, 5), ".")

for r, c in path:
    grid[r, c] = "*"

grid[0, 0] = "S"
grid[4, 4] = "G"

print("\nGrid:")
for row in grid:
    print(" ".join(row))