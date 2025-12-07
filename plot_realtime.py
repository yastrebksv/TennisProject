import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load CSV file
df = pd.read_csv(r"H:\TemprorayCodes\Dr. Klaus\Ball Tracking\TennisProject\output\test - Copy.csv")

# Clean unwanted columns
df = df.drop(columns=[col for col in df.columns if "Unnamed" in col])

# Extract coordinates
x = df["ball_x"].values
y = -df["ball_y"].values

# Setup plot
fig, ax = plt.subplots()
ax.set_xlim(min(x) - 20, max(x) + 20)
ax.set_ylim(min(y) - 20, max(y) + 20)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Ball Movement Over Time")

point, = ax.plot([], [], 'ro')  # red point

# Update function
def update(frame):
    point.set_data([x[frame]], [y[frame]])  # wrap in list
    return point,

# Animate
ani = animation.FuncAnimation(fig, update, frames=len(x), interval=1000/17, blit=True)

plt.show()
