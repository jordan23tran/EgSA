#plot_summary.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import glob
import re

SIMDIR = '/mnt/c/Users/sbtol/Downloads/sparta-24Sep2025/EgSA/30kg_Reentry/'

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

def get_slice(data, axis='y', val=0, tol=1.5):
    ax_idx = {'x': 1, 'y': 2, 'z': 3}[axis]
    mask = np.abs(data[:, ax_idx] - val) < tol
    return data[mask]

def add_satellite(ax, edgecolor='white'):
    sat = Rectangle((-0.5, -0.5), 1, 1, linewidth=1.5,
                    edgecolor=edgecolor, facecolor='gray', alpha=0.7)
    ax.add_patch(sat)

# Load final flow snapshot
files = sorted(glob.glob(SIMDIR + 'flow.*.dat'),
               key=lambda x: int(re.search(r'flow\.(\d+)\.dat', x).group(1)))
ts, final_data = parse_flow_file(files[-1])
slc = get_slice(final_data, axis='y', val=0, tol=1.5)
xc   = slc[:, 1]
zc   = slc[:, 3]
w    = slc[:, 6]
nrho = slc[:, 7]
temp = slc[:, 8]

# Load forces
forces = np.loadtxt(SIMDIR + 'forces.dat', comments='#')
steps  = forces[:, 0]
Fz     = forces[:, 3]
half   = len(Fz) // 2
mean_drag = np.mean(Fz[half:])

# Build figure
fig = plt.figure(figsize=(14, 10))
fig.suptitle(f'30kg Satellite Reentry — 250km DSMC Summary (Step {ts})',
             fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, :])

plot_kwargs = dict(s=40, marker='s')

# Panel 1: Z-velocity
sc1 = ax1.scatter(xc, zc, c=w, cmap='coolwarm', **plot_kwargs)
add_satellite(ax1, edgecolor='black')
ax1.set_title('Z-Velocity (m/s)')
ax1.set_xlabel('X (m)'); ax1.set_ylabel('Z (m)')
ax1.set_xlim(-15,15); ax1.set_ylim(-5,35)
plt.colorbar(sc1, ax=ax1)

# Panel 2: Number density
sc2 = ax2.scatter(xc, zc, c=nrho, cmap='plasma', **plot_kwargs)
add_satellite(ax2)
ax2.set_title('Number Density (m⁻³)')
ax2.set_xlabel('X (m)'); ax2.set_ylabel('Z (m)')
ax2.set_xlim(-15,15); ax2.set_ylim(-5,35)
plt.colorbar(sc2, ax=ax2)

# Panel 3: Flow temperature
sc3 = ax3.scatter(xc, zc, c=temp, cmap='inferno', **plot_kwargs)
add_satellite(ax3)
ax3.set_title('Flow Temperature (K)')
ax3.set_xlabel('X (m)'); ax3.set_ylabel('Z (m)')
ax3.set_xlim(-15,15); ax3.set_ylim(-5,35)
plt.colorbar(sc3, ax=ax3)

# Panel 4: Drag convergence
ax4.plot(steps, Fz, color='tomato', linewidth=1.2, label='Fz (drag)')
ax4.axhline(mean_drag, color='darkred', linestyle='--', linewidth=1.5,
            label=f'Mean (2nd half): {mean_drag:.4f} N')
ax4.axhline(0, color='gray', linestyle=':', linewidth=0.8)
ax4.set_xlabel('Timestep')
ax4.set_ylabel('Force (N)')
ax4.set_title('Drag Force Convergence')
ax4.legend()

plt.savefig(SIMDIR + 'summary.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved summary.png")