import glob
import re

dump_files = sorted(
    glob.glob("impact.*.dump"),
    key=lambda f: int(re.search(r'impact\.(\d+)\.dump', f).group(1))
)

with open("impact.xyz", "w") as xyz:

    for dumpfile in dump_files:

        with open(dumpfile) as f:
            lines = f.readlines()

        natoms = 0
        header = None
        start = None
        timestep = 0

        for i, line in enumerate(lines):

            if line.startswith("ITEM: TIMESTEP"):
                timestep = int(lines[i+1])

            elif line.startswith("ITEM: NUMBER OF ATOMS"):
                natoms = int(lines[i+1])

            elif line.startswith("ITEM: ATOMS"):
                header = line.split()[2:]
                start = i + 1
                break

        xyz.write(f"{natoms}\n")
        xyz.write(f"Timestep {timestep}\n")

        xcol = header.index("x")
        ycol = header.index("y")
        zcol = header.index("z")
        tcol = header.index("type")

        for line in lines[start:start+natoms]:

            vals = line.split()

            atype = int(vals[tcol])

            species = "Al" if atype == 1 else "O"

            xyz.write(
                f"{species} "
                f"{vals[xcol]} "
                f"{vals[ycol]} "
                f"{vals[zcol]}\n"
            )

print("Wrote impact.xyz")