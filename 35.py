import pandas as pd

data = {
    "Algorithm": [
        "Q-Learning",
        "DQN"
    ],

    "Environment": [
        "FrozenLake",
        "CartPole"
    ],

    "Learning Method": [
        "Q-Table",
        "Neural Network"
    ],

    "Exploration": [
        "Epsilon-Greedy",
        "Epsilon-Greedy"
    ],

    "Memory": [
        "Not Required",
        "Replay Memory"
    ],

    "Target Network": [
        "No",
        "Yes"
    ]
}

report = pd.DataFrame(data)

print("Reinforcement Learning Comparative Report")
print("=" * 50)

print(report)

report.to_csv(
    "RL_comparative_report.csv",
    index=False
)

print("\nReport saved as RL_comparative_report.csv")