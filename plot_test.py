import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load your CSV file (adjust path if needed)
df = pd.read_csv(r"H:\TemprorayCodes\Dr. Klaus\Ball Tracking\TennisProject\ball_y_interp.csv")

# Extract values (first column as Series)
y = pd.Series(df.iloc[:, 0].astype(float)).reset_index(drop=True)
x = y.index.to_numpy()
dy = y.diff()  # derivative for classification

# --- parameters you can tune ---
Q_LOW, Q_HIGH = 0.20, 0.80  # use only bottom/top 15% of y as hit zones
MIN_GAP = 8                 # frames between events
BOUNCE_PROM = 30.0          # prominence threshold for bounces
HIT_PROM = 60.0             # prominence threshold for hits

# --- detect bounces as strong local minima in y ---
bounces, _ = find_peaks(-y.values, prominence=BOUNCE_PROM, distance=MIN_GAP)

# --- detect hits as strong maxima/minima with prominence ---
hits_max, _ = find_peaks(y.values, prominence=HIT_PROM, distance=MIN_GAP)
hits_min, _ = find_peaks((-y).values, prominence=HIT_PROM, distance=MIN_GAP)
hits = np.union1d(hits_max, hits_min)

# --- filter hits to only top/bottom court zones ---
y_low = y.quantile(Q_LOW)
y_high = y.quantile(Q_HIGH)
hits = [i for i in hits if (y.iloc[i] <= y_low or y.iloc[i] >= y_high)]

# --- refine classification for bottom-side events ---
hits_bottom = []
real_bounces = []

for idx in bounces:
    if idx + 1 < len(dy):  # avoid last index
        if dy.iloc[idx + 1] > 0:
            real_bounces.append(idx)   # ball goes up again → bounce
        else:
            hits_bottom.append(idx)    # ball keeps going down → hit at bottom

# update sets
hits = np.union1d(hits, hits_bottom)
bounces = np.array(real_bounces)

# ---------------- plotting ----------------
fig, ax = plt.subplots(figsize=(12, 5))

# ball_y with events
ax.plot(x, y, marker='o', markersize=3, linewidth=1, label='ball_y_interp')
ax.scatter(hits, y.iloc[hits], color='red', s=60, marker='x', label='hit (top/bottom)')
# ax.scatter(bounces, y.iloc[bounces], color='orange', s=60, marker='o', label='bounce')
ax.axhline(y_low, linestyle='--', linewidth=1, color='gray')
ax.axhline(y_high, linestyle='--', linewidth=1, color='gray')

ax.set_ylabel('ball_y')
ax.set_xlabel('frame')
ax.set_title('Refined Hits & Bounces (with bottom-hit separation)')
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()

# Optional: export indices
# pd.Series(hits, name='hit_frame').to_csv('hit_candidates.csv', index=False)
# pd.Series(bounces, name='bounce_frame').to_csv('bounce_candidates.csv', index=False)
