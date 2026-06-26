import numpy as np

def read_particles(filename):
    xs, ys, zs = [], [], []
    vxs, vys, vzs = [], [], []

    with open(filename, "r") as f:
        lines = f.readlines()

    read = False

    for line in lines:
        if "ITEM: ATOMS" in line:
            read = True
            continue

        if read:
            p = line.split()
            if len(p) < 8:
                continue

            xs.append(float(p[2]))
            ys.append(float(p[3]))
            zs.append(float(p[4]))

            vxs.append(float(p[5]))
            vys.append(float(p[6]))
            vzs.append(float(p[7]))

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    vx = np.array(vxs)
    vy = np.array(vys)
    vz = np.array(vzs)

    speed = np.sqrt(vx**2 + vy**2 + vz**2)

    return xs, ys, zs, speed


def write_vtu(xs, ys, zs, speed, outname):
    import vtk

    points = vtk.vtkPoints()

    for x, y, z in zip(xs, ys, zs):
        points.InsertNextPoint(x, y, z)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    speed_array = vtk.vtkFloatArray()
    speed_array.SetName("speed")

    for s in speed:
        speed_array.InsertNextValue(float(s))

    polydata.GetPointData().AddArray(speed_array)
    polydata.GetPointData().SetActiveScalars("speed")

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(outname)
    writer.SetInputData(polydata)
    writer.Write()

import glob
import os
import re

SIMDIR = '/mnt/c/Users/sbtol/Downloads/sparta-24Sep2025/EgSA/Goce_Reentry/'

files = sorted(glob.glob(SIMDIR + 'particles.*.dat'),
               key=lambda x: int(re.search(r'particles\.(\d+)\.dat', x).group(1)))

for file in files:
    xs, ys, zs, speed = read_particles(file)
    ts = re.search(r'particles\.(\d+)\.dat', file).group(1)
    outfile = SIMDIR + f'particles.{ts}.vtp'
    write_vtu(xs, ys, zs, speed, outfile)
    print(f"Created: {outfile}")

# Write a PVD index file so ParaView treats all timesteps as one animation
pvd_path = SIMDIR + 'particles_series.pvd'
with open(pvd_path, 'w') as pvd:
    pvd.write('<?xml version="1.0"?>\n')
    pvd.write('<VTKFile type="Collection" version="0.1">\n')
    pvd.write('  <Collection>\n')
    for file in files:
        ts = re.search(r'particles\.(\d+)\.dat', file).group(1)
        pvd.write(f'    <DataSet timestep="{ts}" file="particles.{ts}.vtp"/>\n')
    pvd.write('  </Collection>\n')
    pvd.write('</VTKFile>\n')
print(f"Created: {pvd_path}")