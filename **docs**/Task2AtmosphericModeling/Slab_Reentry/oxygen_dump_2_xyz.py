import glob
import re

files = sorted(
    glob.glob("oxygen.*.dump"),
    key=lambda f: int(re.search(r'oxygen\.(\d+)\.dump', f).group(1))
)

with open("oxygen.xyz","w") as xyz:

    for fn in files:

        with open(fn) as f:
            lines = f.readlines()

        natoms = 0
        header = None
        start = None
        timestep = 0

        for i,line in enumerate(lines):

            if line.startswith("ITEM: TIMESTEP"):
                timestep = int(lines[i+1])

            elif line.startswith("ITEM: NUMBER OF ATOMS"):
                natoms = int(lines[i+1])

            elif line.startswith("ITEM: ATOMS"):
                header = line.split()[2:]
                start = i+1
                break

        xyz.write(f"{natoms}\n")
        xyz.write(f"Timestep {timestep}\n")

        xcol = header.index("x")
        ycol = header.index("y")
        zcol = header.index("z")

        for line in lines[start:start+natoms]:

            vals = line.split()

            xyz.write(
                f"O "
                f"{vals[xcol]} "
                f"{vals[ycol]} "
                f"{vals[zcol]}\n"
            )

print("Wrote oxygen.xyz")