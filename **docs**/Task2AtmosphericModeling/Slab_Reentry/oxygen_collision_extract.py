import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# User settings
# --------------------------------------------------

OXYGEN_ID = 9451

# --------------------------------------------------
# Gather dump files in timestep order
# --------------------------------------------------

dump_files = sorted(
    glob.glob("impact.*.dump"),
    key=lambda f: int(re.search(r'impact\.(\d+)\.dump', f).group(1))
)

records = []

# --------------------------------------------------
# Parse each dump
# --------------------------------------------------

for filename in dump_files:

    with open(filename, 'r') as f:
        lines = f.readlines()

    timestep = None
    atom_start = None

    for i, line in enumerate(lines):

        if "ITEM: TIMESTEP" in line:
            timestep = int(lines[i + 1].strip())

        if "ITEM: ATOMS" in line:
            atom_start = i + 1
            break

    if atom_start is None:
        continue

    for line in lines[atom_start:]:

        vals = line.split()

        atom_id = int(vals[0])

        if atom_id == OXYGEN_ID:

            records.append({
                "timestep": timestep,
                "id": atom_id,
                "type": int(vals[1]),
                "charge": float(vals[2]),
                "x": float(vals[3]),
                "y": float(vals[4]),
                "z": float(vals[5]),
                "vx": float(vals[6]),
                "vy": float(vals[7]),
                "vz": float(vals[8]),
                "peatom": float(vals[9]),
                "coord": float(vals[10])
            })

            break

# --------------------------------------------------
# Create dataframe
# --------------------------------------------------

df = pd.DataFrame(records)

# Convert timestep to physical time (ps)
# timestep size = 0.0001 ps

df["time_ps"] = df["timestep"] * 0.0001

# --------------------------------------------------
# Save CSV
# --------------------------------------------------

csv_name = "oxygen_history.csv"

df.to_csv(csv_name, index=False)

print(f"Wrote {csv_name}")

# --------------------------------------------------
# Generate plots
# --------------------------------------------------

fig, axes = plt.subplots(4, 1, figsize=(8, 12))

axes[0].plot(df["time_ps"], df["z"])
axes[0].set_ylabel("z (Å)")
axes[0].set_title("Oxygen Depth")

axes[1].plot(df["time_ps"], df["charge"])
axes[1].set_ylabel("Charge")

axes[2].plot(df["time_ps"], df["coord"])
axes[2].set_ylabel("Coordination")

axes[3].plot(df["time_ps"], df["peatom"])
axes[3].set_ylabel("PE (eV)")
axes[3].set_xlabel("Time (ps)")

plt.tight_layout()

plot_name = "oxygen_history.png"

plt.savefig(plot_name, dpi=300)

print(f"Wrote {plot_name}")

plt.show()