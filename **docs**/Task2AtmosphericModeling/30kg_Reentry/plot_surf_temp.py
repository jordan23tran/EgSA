import numpy as np
import matplotlib.pyplot as plt
import glob
import re

def parse_surf_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Extract timestep
    timestep = int(lines[1].strip())
    
    # Find data start (line after ITEM: SURFS header)
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

# Load all files in order
files = sorted(glob.glob('surface.*.dat'), 
               key=lambda x: int(re.search(r'(\d+)', x).group()))

timesteps = []
all_data = []
for f in files:
    ts, data = parse_surf_file(f)
    timesteps.append(ts)
    all_data.append(data)

# Plot temperature (column 5, index 5) per triangle at final timestep
final = all_data[-1]
triangle_ids = final[:, 0].astype(int)
temperatures = final[:, 5]

plt.figure(figsize=(10, 5))
bars = plt.bar(triangle_ids, temperatures, color='tomato', edgecolor='black')
plt.xlabel('Triangle ID')
plt.ylabel('Temperature (K)')
plt.title(f'Surface Temperature per Triangle at Step {timesteps[-1]}')
plt.xticks(triangle_ids)
plt.axhline(y=500, color='gray', linestyle='--', label='Initial temp (500K)')
plt.legend()
plt.tight_layout()
plt.savefig('surf_temp_final.png', dpi=150)
plt.show()
print("Saved surf_temp_final.png")