#animate_surf_temp.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import re

def parse_surf_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    timestep = int(lines[1].strip())
    for i, line in enumerate(lines):
        if line.startswith('ITEM: SURFS'):
            data_start = i + 1
            break
    data = []
    for line in lines[data_start:]:
        vals = line.strip().split()
        if vals:
            data.append([float(v) for v in vals])
    return timestep, np.array(data)

# Load all files in chronological order
files = sorted(glob.glob('/mnt/c/Users/sbtol/Downloads/sparta-24Sep2025/EgSA/Goce_Reentry/surface.*.dat'),
               key=lambda x: int(re.search(r'surface\.(\d+)\.dat', x).group(1)))

timesteps = []
all_temps = []
for f in files:
    ts, data = parse_surf_file(f)
    timesteps.append(ts)
    all_temps.append(data[:, 5])  # s_temp_surf column

triangle_ids = np.arange(1, 13)
all_temps = np.array(all_temps)
vmin = all_temps.min()
vmax = all_temps.max()

# Build animation
fig, ax = plt.subplots(figsize=(10, 5))

def update(frame):
    ax.clear()
    temps = all_temps[frame]
    bars = ax.bar(triangle_ids, temps, color=plt.cm.hot(
                  (temps - vmin) / (vmax - vmin)), edgecolor='black')
    ax.set_ylim(vmin * 0.9, vmax * 1.1)
    ax.set_xlabel('Triangle ID')
    ax.set_ylabel('Temperature (K)')
    ax.set_title(f'Surface Temperature per Triangle — Step {timesteps[frame]}')
    ax.set_xticks(triangle_ids)
    ax.axhline(y=500, color='gray', linestyle='--', alpha=0.5, label='Initial 500K')
    ax.legend()

ani = animation.FuncAnimation(fig, update, frames=len(timesteps), interval=400)
outpath = '/mnt/c/Users/sbtol/Downloads/sparta-24Sep2025/EgSA/Goce_Reentry/surf_temp_evolution.gif'
ani.save(outpath, writer='pillow', dpi=150)
print(f"Saved {outpath}")