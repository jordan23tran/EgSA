#plot_nrho_slice.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import re

def parse_flow_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    timestep = int(lines[1].strip())
    for i, line in enumerate(lines):
        if line.startswith('ITEM: CELLS'):
            data_start = i + 1
            break
    data = []
    for line in lines[data_start:]:
        vals = line.strip().split()
        if vals:
            data.append([float(v) for v in vals])
    return timestep, np.array(data)

files = sorted(glob.glob('/mnt/c/Users/sbtol/Downloads/sparta-24Sep2025/EgSA/Goce_Reentry/flow.*.dat'),
               key=lambda x: int(re.search(r'flow\.(\d+)\.dat', x).group(1)))

timesteps = []
all_data = []
for f in files:
    ts, data = parse_flow_file(f)
    timesteps.append(ts)
    all_data.append(data)

def get_slice(data, axis='y', val=0, tol=1.5):
    ax_idx = {'x': 1, 'y': 2, 'z': 3}[axis]
    mask = np.abs(data[:, ax_idx] - val) < tol
    return data[mask]

all_nrho = np.concatenate([get_slice(d)[:, 7] for d in all_data])
vmin, vmax = np.percentile(all_nrho, 2), np.percentile(all_nrho, 98)

fig, ax = plt.subplots(figsize=(8, 10))
cbar = None

def update(frame):
    global cbar
    ax.clear()
    data = all_data[frame]
    slc = get_slice(data, axis='y', val=0, tol=1.5)

    if len(slc) == 0:
        return

    xc = slc[:, 1]
    zc = slc[:, 3]
    nrho = slc[:, 7]

    sc = ax.scatter(xc, zc, c=nrho, cmap='plasma', vmin=vmin, vmax=vmax,
                    s=40, marker='s')

    from matplotlib.patches import Rectangle
    sat = Rectangle((-0.825, -0.25), 1.65, 2.25, linewidth=1.5,
                    edgecolor='black', facecolor='gray', alpha=0.7)
    ax.add_patch(sat)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title(f'Number Density (nrho) Slice at Y≈0 — Step {timesteps[frame]}')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 5)

    if cbar is None:
        cbar = plt.colorbar(sc, ax=ax, label='Number Density (m⁻³)')

ani = animation.FuncAnimation(fig, update, frames=len(timesteps), interval=400)
outpath = '/mnt/c/Users/sbtol/Downloads/sparta-24Sep2025/EgSA/Goce_Reentry/nrho_slice.gif'
ani.save(outpath, writer='pillow', dpi=150)
print(f"Saved {outpath}")