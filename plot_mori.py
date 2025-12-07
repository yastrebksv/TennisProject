import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

# Load data
df = pd.read_csv(r"H:\TemprorayCodes\Dr. Klaus\Ball Tracking\TennisProject\output\output4.csv")

# Filter valid rows
# valid_data = df[df['ball_x'].notnull() & df['ball_y'].notnull()].reset_index(drop=True)
valid_data = df.reset_index(drop=True)

x_axis = valid_data.index

# Interpolate missing values (choose method)
ball_x_interp = valid_data['ball_x'].interpolate(method='pchip')
ball_y_interp = valid_data['ball_y'].interpolate(method='pchip')

# Derivatives
dx = ball_x_interp.diff()
dy = ball_y_interp.diff()

# Speed
speed = np.sqrt(dx**2 + dy**2)

# Detect bounces as local minima in speed
bounce_candidates = argrelextrema(speed.to_numpy(), np.less, order=3)[0]  # order=3 = sensitivity

# Detect hits using angle change in velocity vectors
vx = dx
vy = dy
vx1, vy1 = vx.shift(1), vy.shift(1)

dot_product = vx * vx1 + vy * vy1
norm1 = np.sqrt(vx**2 + vy**2)
norm2 = np.sqrt(vx1**2 + vy1**2)

# Avoid divide by zero
cos_angle = dot_product / (norm1 * norm2)
cos_angle = np.clip(cos_angle, -1, 1)

angle_change = np.arccos(cos_angle)  # radians
hit_candidates = angle_change[angle_change > 1.0].index  # >1 rad (~57 degrees)

# Bounce (manual flag)
bounce_points = valid_data[valid_data['bounce'] == True]

# dy/dt zero-crossings
sign_change = np.sign(dy).diff()
zero_crossings = sign_change[sign_change != 0].index

# Plotting
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(12, 14), sharex=True)

# ball_x
ax1.plot(x_axis, valid_data['ball_x'], label='ball_x')
ax1.set_ylabel('ball_x')
ax1.set_title('Ball Tracking with Derivatives and Event Detection')
ax1.grid(True)
ax1.legend()

# ball_y
ax2.plot(x_axis, valid_data['ball_y'], label='ball_y')
ax2.set_ylabel('ball_y')
ax2.grid(True)
ax2.legend()

# Bounce (manual)
ax3.scatter(bounce_points.index, [1]*len(bounce_points), color='red', label='bounce', marker='o')
ax3.set_ylabel('Bounce')
ax3.set_yticks([0, 1])
ax3.set_yticklabels(['', 'True'])
ax3.grid(True)
ax3.legend()

# dy/dt
ax4.plot(x_axis, dy, label="dy/dt (ball_y')", color='purple')
ax4.set_ylabel("ball_y'")
ax4.grid(True)
ax4.legend()

# Zero-crossings of dy
ax5.scatter(zero_crossings, [1]*len(zero_crossings), color='black', label='dy/dt crosses 0', marker='x')
ax5.set_ylabel("Zero-Cross")
ax5.set_yticks([0, 1])
ax5.set_yticklabels(['', 'Cross'])
ax5.grid(True)
ax5.legend()

# Detection plot: Speed with bounce & hit events
ax6.plot(x_axis, speed, label='Speed (sqrt(dx² + dy²))', color='blue')
ax6.scatter(bounce_candidates, speed.iloc[bounce_candidates], label='Detected Bounce', color='green', marker='o')
ax6.scatter(hit_candidates, speed.iloc[hit_candidates], label='Detected Hit', color='orange', marker='x')
ax6.set_ylabel('Speed')
ax6.set_xlabel('Frame Number')
ax6.grid(True)
ax6.legend()

plt.tight_layout()
plt.show()
