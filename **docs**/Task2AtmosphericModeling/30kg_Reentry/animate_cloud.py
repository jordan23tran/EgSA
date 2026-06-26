import glob
import matplotlib
#matplotlib.use("Agg")  # <-- Must be set BEFORE importing pyplot
import matplotlib.pyplot as plt
import imageio
import numpy as np

# ── 1. Collect files ──────────────────────────────────────────────────────────
files = sorted(glob.glob("particles*.dat"))
print("Files found:", files)

# ── 2. Parser ─────────────────────────────────────────────────────────────────
def read_particles(filename):
    xs, ys, zs = [], [], []
    vxs, vys, vzs = [], [], []
    with open(filename, "r") as f:
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
            x, y, z = map(float, parts[2:5])
            vx, vy, vz = map(float, parts[5:8])
            xs.append(x);  ys.append(y);  zs.append(z)
            vxs.append(vx); vys.append(vy); vzs.append(vz)
    xs    = np.array(xs)
    ys    = np.array(ys)
    zs    = np.array(zs)
    speed = np.sqrt(np.array(vxs)**2 + np.array(vys)**2 + np.array(vzs)**2)
    return xs, ys, zs, speed

# ── 3. Build frames  (everything stays INSIDE the loop) ───────────────────────
step = 5
frames = []
for f in files:
    xs, ys, zs, speed = read_particles(f)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
    xs,
    ys,
    zs,
    c=speed,
    s=0.5,
    cmap="viridis",
    vmin=6000,
    vmax=9500
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.title("DSMC Particle Cloud (Subsampled)")
    ax.view_init(elev=20, azim=45)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_zlim(-5, 35)
    fig.colorbar(sc, ax=ax, label="Particle Speed (m/s)")

    fig.canvas.draw()                          # render to buffer
    image = np.array(fig.canvas.buffer_rgba()) # grab RGBA pixels
    image = image[:, :, :3]                    # drop alpha → RGB
    frames.append(image)
    plt.close(fig)                             # free memory


# ── 4. Save GIF ───────────────────────────────────────────────────────────────
imageio.mimsave("dsmc_T1_cloud_animation.gif", frames, fps=1, loop=0)
print(f"GIF saved with {len(frames)} frames!")