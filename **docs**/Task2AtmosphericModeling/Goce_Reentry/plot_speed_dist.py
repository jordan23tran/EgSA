#plot_speed_dist.py
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

SIMDIR = '/mnt/c/Users/sbtol/Downloads/sparta-24Sep2025/EgSA/Goce_Reentry/'

def parse_particle_file(filename, max_particles=50000):
    with open(filename, 'r') as f:
        lines = f.readlines()
    timestep = int(lines[1].strip())
    for i, line in enumerate(lines):
        if line.startswith('ITEM: ATOMS'):
            data_start = i + 1
            break
    data = []
    for line in lines[data_start:]:
        vals = line.strip().split()
        if vals:
            data.append([float(v) for v in vals])
        if len(data) >= max_particles:
            break
    return timestep, np.array(data)

# Use only the final snapshot for the distribution
files = sorted(glob.glob(SIMDIR + 'particles.*.dat'),
               key=lambda x: int(re.search(r'particles\.(\d+)\.dat', x).group(1)))

ts, data = parse_particle_file(files[-1], max_particles=50000)

vx = data[:, 5]
vy = data[:, 6]
vz = data[:, 7]
speed = np.sqrt(vx**2 + vy**2 + vz**2)
vz_bulk = np.abs(vz)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'Particle Speed Distribution — Step {ts} (sample of 50,000)', fontsize=13)

# Panel 1: Total speed magnitude
axes[0].hist(speed, bins=100, color='steelblue', edgecolor='none', density=True)
axes[0].axvline(7750, color='red', linestyle='--', linewidth=1.5,
                label='Bulk flow speed (7750 m/s)')
axes[0].axvline(np.mean(speed), color='orange', linestyle='--', linewidth=1.5,
                label=f'Sample mean ({np.mean(speed):.0f} m/s)')
axes[0].set_xlabel('Speed Magnitude (m/s)')
axes[0].set_ylabel('Probability Density')
axes[0].set_title('Total Speed |v|')
axes[0].legend()

# Panel 2: Z-velocity component only
axes[1].hist(vz, bins=100, color='tomato', edgecolor='none', density=True)
axes[1].axvline(-7750, color='red', linestyle='--', linewidth=1.5,
                label='Bulk vz (-7750 m/s)')
axes[1].axvline(np.mean(vz), color='orange', linestyle='--', linewidth=1.5,
                label=f'Sample mean ({np.mean(vz):.0f} m/s)')
axes[1].set_xlabel('Vz (m/s)')
axes[1].set_ylabel('Probability Density')
axes[1].set_title('Z-Velocity Component (flow direction)')
axes[1].legend()

plt.tight_layout()
plt.savefig(SIMDIR + 'speed_distribution.png', dpi=150)
plt.show()
print(f"Mean speed:    {np.mean(speed):.1f} m/s")
print(f"Median speed:  {np.median(speed):.1f} m/s")
print(f"Mean Vz:       {np.mean(vz):.1f} m/s")
print(f"Speed std dev: {np.std(speed):.1f} m/s")