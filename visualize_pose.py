import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("throw_landmarks.csv")

# Mediapipe connections (simplified)
connections = [
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 12),            # shoulders
    (23, 24),            # hips
    (11, 23), (12, 24),  # torso
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28)   # right leg
]

plt.ion()

for i in range(len(df)):
    row = df.iloc[i].values

    xs = row[0::3]
    ys = row[1::3]

    plt.clf()

    # Draw points
    plt.scatter(xs, ys)

    # Draw skeleton
    for a, b in connections:
        plt.plot([xs[a], xs[b]], [ys[a], ys[b]])

    plt.gca().invert_yaxis()
    plt.title(f"Frame {i}")
    plt.pause(0.01)

plt.ioff()
plt.show()
