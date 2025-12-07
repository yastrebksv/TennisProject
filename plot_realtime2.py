import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load both CSVs
df1 = pd.read_csv(r"H:\TemprorayCodes\Dr. Klaus\Ball Tracking\TennisProject\output\test.csv")
df2 = pd.read_csv(r"H:\TemprorayCodes\Dr. Klaus\Ball Tracking\TennisProject\output\test - Copy.csv")

# Drop unwanted columns
df1 = df1.drop(columns=[col for col in df1.columns if "Unnamed" in col])
df2 = df2.drop(columns=[col for col in df2.columns if "Unnamed" in col])

# Extract coordinates
x1, y1 = df1["ball_x"].values, -df1["ball_y"].values
x2, y2 = df2["ball_x"].values, -df2["ball_y"].values

# Setup plot
fig, ax = plt.subplots()
ax.set_xlim(min(min(x1), min(x2)) - 20, max(max(x1), max(x2)) + 20)
ax.set_ylim(min(min(y1), min(y2)) - 20, max(max(y1), max(y2)) + 20)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Ball Movement Over Time (Two Animations)")

# Two points
point1, = ax.plot([], [], 'ro', label="Ball 1")  # red
point2, = ax.plot([], [], 'bo', label="Ball 2")  # blue
ax.legend()

# Update function
def update(frame):
    if frame < len(x1):
        point1.set_data([x1[frame]], [y1[frame]])
    if frame < len(x2):
        point2.set_data([x2[frame]], [y2[frame]])
    return point1, point2

# Sync animations → use max frames
max_frames = max(len(x1), len(x2))
ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=1000/17, blit=True)

plt.show()
