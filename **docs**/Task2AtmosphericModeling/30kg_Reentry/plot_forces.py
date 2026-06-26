#plot_forces.py

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('/mnt/c/Users/sbtol/Downloads/sparta-24Sep2025/EgSA/30kg_Reentry/forces.dat',
                  comments='#')

steps = data[:, 0]
Fx    = data[:, 1]
Fy    = data[:, 2]
Fz    = data[:, 3]

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(steps, Fx, color='steelblue')
axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
axes[0].set_ylabel('Fx (N)')
axes[0].set_title('Surface Force Components Over Time')

axes[1].plot(steps, Fy, color='seagreen')
axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
axes[1].set_ylabel('Fy (N)')

axes[2].plot(steps, Fz, color='tomato')
axes[2].axhline(0, color='gray', linestyle='--', linewidth=0.8)
axes[2].set_ylabel('Fz — Drag (N)')
axes[2].set_xlabel('Timestep')

# Annotate mean drag over second half of run (after initial transient)
half = len(Fz) // 2
mean_drag = np.mean(Fz[half:])
axes[2].axhline(mean_drag, color='darkred', linestyle=':', linewidth=1.5,
                label=f'Mean drag (2nd half): {mean_drag:.4f} N')
axes[2].legend()

plt.tight_layout()
plt.savefig('/mnt/c/Users/sbtol/Downloads/sparta-24Sep2025/EgSA/30kg_Reentry/forces.png', dpi=150)
plt.show()
print(f"Mean Fz (drag): {mean_drag:.4f} N")
print(f"Mean Fx (side): {np.mean(Fx[half:]):.4f} N")
print(f"Mean Fy (lift): {np.mean(Fy[half:]):.4f} N")