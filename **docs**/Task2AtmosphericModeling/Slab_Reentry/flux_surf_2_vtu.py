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


xs, ys, zs, speed = read_particles("flux_surf.5000.dat")
write_vtu(xs, ys, zs, speed, "flux_surf.5000.vtp")

print("Done: flux_surf.5000.vtp")

#import glob
#import os

#files = sorted(glob.glob("../../EgSA/Goce_Reentry/flux_surf.*.dat"))

#for file in files:

    #xs, ys, zs, speed = read_particles(file)

    #outfile = os.path.basename(file).replace(".dat", ".vtu")

    #write_vtu(xs, ys, zs, speed, outfile)

    #print("Created:", outfile)