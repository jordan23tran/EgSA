import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle



input_file = "particles5000.dat"

xs, ys, zs = [], [], []
vxs, vys, vzs = [], [], []

with open(input_file, "r") as f:
    lines = f.readlines()

read_data = False

for line in lines:
    if "ITEM: ATOMS" in line:
        read_data = True
        continue

    if read_data:
        parts = line.split()
        if len(parts) < 8:
            continue

        # id = parts[0], type = parts[1]
        x, y, z = map(float, parts[2:5])
        vx, vy, vz = map(float, parts[5:8])

        xs.append(x); ys.append(y); zs.append(z)
        vxs.append(vx); vys.append(vy); vzs.append(vz)

xs = np.array(xs)
ys = np.array(ys)
zs = np.array(zs)

vxs = np.array(vxs)
vys = np.array(vys)
vzs = np.array(vzs)

speed = np.sqrt(vxs**2 + vys**2 + vzs**2)

print("\n--- DSMC PARTICLE SNAPSHOT ---")
print("Particles loaded:", len(xs))
print("Speed min:", speed.min())
print("Speed max:", speed.max())
print("Speed mean:", speed.mean())
print("--------------------------------\n")

#MAKES VISUALIZATION

plt.figure(figsize=(8,6))

plt.scatter(
    zs, xs,
    c=speed,
    s=0.5,
    cmap='plasma',
    alpha=0.7,
    linewidths=0
)

plt.colorbar(label="Speed (m/s)")
plt.xlabel("z (flow direction)")
plt.ylabel("x (cross-stream)")
plt.title("DSMC Particle Projection (Wake + Shock Structure)")
plt.tight_layout()
plt.show()


#DENSITY BINNING
plt.figure(figsize=(8,6))

xbins = 100
ybins = 100

H, xedges, yedges = np.histogram2d(
    zs, xs,
    bins=[xbins, ybins]
)

plt.imshow(
    H.T,
    origin='lower',
    aspect='auto',
    cmap='inferno',
    extent=[zs.min(), zs.max(), xs.min(), xs.max()]
)

plt.colorbar(label="Particle Count (Density Proxy)")
plt.xlabel("z (flow direction)")
plt.ylabel("x (cross-stream)")
plt.title("DSMC Density Field (Shock + Wake)")
plt.tight_layout()
plt.show()


#VELOCITY QUIVER
nx, ny = 30, 30  # keep coarse for stability

vx_grid = np.zeros((nx, ny))
vy_grid = np.zeros((nx, ny))
vz_grid = np.zeros((nx, ny))
count   = np.zeros((nx, ny))

z_edges = np.linspace(zs.min(), zs.max(), nx + 1)
x_edges = np.linspace(xs.min(), xs.max(), ny + 1)

for i in range(len(xs)):
    xi = np.searchsorted(z_edges, zs[i]) - 1
    yi = np.searchsorted(x_edges, xs[i]) - 1

    if 0 <= xi < nx and 0 <= yi < ny:
        vx_grid[xi, yi] += vxs[i]
        vy_grid[xi, yi] += vys[i]
        vz_grid[xi, yi] += vzs[i]
        count[xi, yi] += 1

mask = count > 0

vx_grid[mask] /= count[mask]
vy_grid[mask] /= count[mask]
vz_grid[mask] /= count[mask]

plt.figure(figsize=(8,6))

Z, X = np.meshgrid(
    0.5 * (z_edges[:-1] + z_edges[1:]),
    0.5 * (x_edges[:-1] + x_edges[1:])
)

plt.quiver(
    Z, X,
    vz_grid.T, vx_grid.T,
    scale=5000
)

plt.xlabel("z (flow direction)")
plt.ylabel("x")
plt.title("DSMC Mean Velocity Field (Quasi-continuum)")
plt.show()


###################################33
x_center = np.mean(xs)
mask = (np.abs(xs - x_center) < 0.5)
plt.plot(zs[mask], np.ones_like(zs[mask]), '.')

z_bins = np.linspace(zs.min(), zs.max(), 60)
density_profile = np.zeros(len(z_bins)-1)

for i in range(len(z_bins)-1):
    region = (zs >= z_bins[i]) & (zs < z_bins[i+1])
    density_profile[i] = np.sum(region)

plt.plot(z_bins[:-1], density_profile)
plt.xlabel("z (flow direction)")
plt.ylabel("Relative number density")
plt.title("Shock / Wake Density Profile")
plt.show()


#UPGRADES TO 3D


step = 5

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    xs[::step],
    ys[::step],
    zs[::step],
    c=speed[::step],
    s=0.5,
    cmap='plasma'
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.title("DSMC Particle Cloud (Subsampled)")
plt.show()

# ── 0. Initialize Geometry ──────────────────────────────────────────────────────────
ax.add_patch(
    Rectangle(
        (0, -10),    # (z,x)
        30,          # z length
        20,          # x length
        fill=False,
        edgecolor='red',
        linewidth=4
    )
)


z_min, z_max = zs.min(), zs.max()

z_stag = 0.0   # approximate cube center (adjust if needed)

free_stream = zs < (z_stag - 5)
interaction = (zs >= (z_stag - 5)) & (zs <= (z_stag + 5))
wake = zs > (z_stag + 5)

print("FREE STREAM:", np.mean(speed[free_stream]))
print("INTERACTION:", np.mean(speed[interaction]))
print("WAKE:", np.mean(speed[wake]))

print("Density FS:", np.sum(free_stream))
print("Density INT:", np.sum(interaction))
print("Density WAKE:", np.sum(wake))

labels = ["Free Stream", "Interaction", "Wake"]

means = [
    np.mean(speed[free_stream]),
    np.mean(speed[interaction]),
    np.mean(speed[wake])
]

counts = [
    np.sum(free_stream),
    np.sum(interaction),
    np.sum(wake)
]

import matplotlib.pyplot as plt

plt.figure()

plt.bar(labels, means)
plt.ylabel("Mean Speed (m/s)")
plt.title("Flow Deceleration Across Cube Interaction Zones")

plt.show()