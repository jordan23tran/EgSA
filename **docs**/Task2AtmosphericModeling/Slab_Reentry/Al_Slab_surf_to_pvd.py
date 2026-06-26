import glob
import re
import os

# ==========================================================
# Read SPARTA surface geometry
# ==========================================================

def read_surf(filename):

    with open(filename) as f:
        lines = [l.strip() for l in f if l.strip()]

    npoints = int(lines[0].split()[0])
    ntris = int(lines[1].split()[0])

    points_idx = lines.index("Points")
    tris_idx = lines.index("Triangles")

    points = []

    for line in lines[points_idx+1:points_idx+1+npoints]:
        pid,x,y,z = line.split()
        points.append((float(x),float(y),float(z)))

    triangles = []

    for line in lines[tris_idx+1:tris_idx+1+ntris]:
        tid,p1,p2,p3 = line.split()

        triangles.append((
            int(p1)-1,
            int(p2)-1,
            int(p3)-1
        ))

    return points, triangles

# ==========================================================
# Read one SPARTA surface data file
# ==========================================================

def read_dat(filename):

    with open(filename) as f:
        lines = f.readlines()

    surf_start = None

    for i,line in enumerate(lines):

        if line.startswith("ITEM: SURFS"):
            surf_start = i + 1
            break

    if surf_start is None:
        raise RuntimeError(f"Could not find SURFS block in {filename}")

    data = {}

    for line in lines[surf_start:]:

        parts = line.split()

        if len(parts) < 6:
            continue

        sid = int(parts[0])

        data[sid] = [
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
            float(parts[5])
        ]

    return data

# ==========================================================
# Write VTP
# ==========================================================

def write_vtp(outfile, points, triangles, surfdata):

    with open(outfile, "w") as f:

        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write('<PolyData>\n')

        f.write(
            f'<Piece NumberOfPoints="{len(points)}" '
            f'NumberOfPolys="{len(triangles)}">\n'
        )

        # ----------------------------------------------
        # Points
        # ----------------------------------------------

        f.write('<Points>\n')
        f.write('<DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')

        for x,y,z in points:
            f.write(f'{x} {y} {z} ')

        f.write('\n</DataArray>\n')
        f.write('</Points>\n')

        # ----------------------------------------------
        # Polygons
        # ----------------------------------------------

        f.write('<Polys>\n')

        f.write('<DataArray type="Int32" Name="connectivity" format="ascii">\n')

        for p1,p2,p3 in triangles:
            f.write(f'{p1} {p2} {p3} ')

        f.write('\n</DataArray>\n')

        f.write('<DataArray type="Int32" Name="offsets" format="ascii">\n')

        offset = 0

        for _ in triangles:
            offset += 3
            f.write(f'{offset} ')

        f.write('\n</DataArray>\n')

        f.write('</Polys>\n')

        # ----------------------------------------------
        # Cell data
        # ----------------------------------------------

        names = [
            "nflux",
            "pressure",
            "shx",
            "shy",
            "shz"
        ]

        f.write('<CellData>\n')

        for idx,name in enumerate(names):

            f.write(
                f'<DataArray type="Float32" '
                f'Name="{name}" '
                f'format="ascii">\n'
            )

            for sid in range(1,len(triangles)+1):

                val = surfdata.get(sid,[0,0,0,0,0])[idx]

                f.write(f'{val} ')

            f.write('\n</DataArray>\n')

        f.write('</CellData>\n')

        f.write('</Piece>\n')
        f.write('</PolyData>\n')
        f.write('</VTKFile>\n')

# ==========================================================
# Main
# ==========================================================

points, triangles = read_surf("Al_Slab.surf")

for pattern in ["surf_AO.*.dat","flux_surf.*.dat"]:

    files = sorted(
        glob.glob(pattern),
        key=lambda f: int(re.search(r'\.(\d+)\.dat',f).group(1))
    )

    pvd_entries = []

    prefix = pattern.split("*")[0].rstrip(".")

    for file in files:

        step = int(re.search(r'\.(\d+)\.dat',file).group(1))

        data = read_dat(file)

        outname = file.replace(".dat",".vtp")

        write_vtp(outname, points, triangles, data)

        pvd_entries.append((step,outname))

        print("Wrote",outname)

    pvd_name = f"{prefix}.pvd"

    with open(pvd_name,"w") as f:

        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1">\n')
        f.write('<Collection>\n')

        for step,file in pvd_entries:

            f.write(
                f'<DataSet timestep="{step}" '
                f'part="0" '
                f'file="{os.path.basename(file)}"/>\n'
            )

        f.write('</Collection>\n')
        f.write('</VTKFile>\n')

    print("Wrote",pvd_name)