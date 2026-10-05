import random
from collections import deque

memory = deque(maxlen=100)

print("Initial Memory Size:", len(memory))

for i in range(150):

    experience = (
        i,
        random.randint(0, 1),
        random.random(),
        i + 1,
        False
    )

    memory.append(experience)

print("Final Memory Size:", len(memory))

print("\nSample Experiences:")
for experience in list(memory)[:5]:
    print(experience)