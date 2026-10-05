import numpy as np

grid_size = 5

start = (0, 0)
goal = (4, 4)

position = start

print("Grid World:")
print("S = Start")
print("G = Goal")
print(". = Empty")

for row in range(grid_size):

    for column in range(grid_size):

        if (row, column) == start:
            print("S", end=" ")
        elif (row, column) == goal:
            print("G", end=" ")
        else:
            print(".", end=" ")

    print()