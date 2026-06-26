#!/usr/bin/env python3
"""
sparta_grid_to_vtu.py — Convert SPARTA grid dump files to real VTK
Unstructured Grid (.vtu) format readable by ParaView.

SPARTA grid dumps have the format:
    ITEM: TIMESTEP
    <N>
    ITEM: NUMBER OF CELLS
    <M>
    ITEM: BOX BOUNDS ...
    xlo xhi
    ylo yhi
    zlo zhi
    ITEM: CELLS col1 col2 ...
    v1 v2 ...
    ...

Column mapping from goce_ao_dsmc.sparta:
    c_cgrid[1] = nrho      (number density, m^-3)
    c_cgrid[2] = massrho   (mass density, kg/m^3)
    c_cgrid[3] = u         (x-velocity, m/s)
    c_cgrid[4] = v         (y-velocity, m/s)
    c_cgrid[5] = w         (z-velocity, m/s)
    c_cgrid[6] = temp      (translational temperature, K)

Usage:
    python3 sparta_grid_to_vtu.py "grid_flow.*.vtp"
    python3 sparta_grid_to_vtu.py "grid_flow.*.vtp" --outdir vtu_output

Output:
    grid_flow_<timestep>.vtu  — one real VTK file per timestep
    grid_flow.pvd             — ParaView collection file linking all timesteps
                                (open this single file to get the full time series)
"""

import argparse
import glob
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Column names matching compute cgrid in the SPARTA script
# ---------------------------------------------------------------------------
COLUMN_NAMES = ["nrho", "massrho", "u", "v", "w", "temp"]


# ---------------------------------------------------------------------------
# 1. Parser for SPARTA text grid dumps
# ---------------------------------------------------------------------------

def parse_sparta_grid_dump(path):
    """
    Parse one SPARTA grid dump file.
    Returns:
        timestep  int
        n_cells   int
        box       np.array shape (3,2) — [[xlo,xhi],[ylo,yhi],[zlo,zhi]]
        col_names list of str
        data      np.array shape (n_cells, n_cols)
    """
    timestep  = None
    n_cells   = None
    box       = np.zeros((3, 2))
    col_names = []
    data_rows = []

    section = None
    box_row = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("ITEM: TIMESTEP"):
                section = "timestep"
                continue
            if line.startswith("ITEM: NUMBER OF CELLS"):
                section = "ncells"
                continue
            if line.startswith("ITEM: BOX BOUNDS"):
                section = "box"
                box_row = 0
                continue
            if line.startswith("ITEM: CELLS"):
                col_names = line.replace("ITEM: CELLS", "").split()
                section = "data"
                continue

            if section == "timestep":
                timestep = int(line)
            elif section == "ncells":
                n_cells = int(line)
            elif section == "box":
                parts = line.split()
                box[box_row] = [float(parts[0]), float(parts[1])]
                box_row += 1
            elif section == "data":
                try:
                    data_rows.append([float(v) for v in line.split()])
                except ValueError:
                    continue

    data = np.array(data_rows) if data_rows else np.zeros((0, len(col_names)))
    return timestep, n_cells, box, col_names, data


# ---------------------------------------------------------------------------
# 2. Build a regular Cartesian grid geometry
#
#    SPARTA's grid dump does not include per-cell coordinates — it lists
#    cell data in a flat order corresponding to the Cartesian grid created
#    by create_grid Nx Ny Nz. We reconstruct cell centres from the box
#    bounds and the known grid dimensions.
#
#    Grid dimensions are inferred from the number of cells:
#      n_cells = Nx * Ny * Nz
#    We try to recover Nx, Ny, Nz from your script (64 64 60 → 245760 base,
#    but you changed it; the actual n_cells in the dump tells us the total).
#    For a uniform Cartesian grid the cell ordering is x-fastest (SPARTA default).
# ---------------------------------------------------------------------------

def infer_grid_dims(n_cells, box):
    """
    Try to recover (Nx, Ny, Nz) from the total cell count and box aspect ratio.
    Uses the box x:y:z ratio to distribute cells proportionally.
    Falls back to a cube root if ratios don't factor cleanly.
    """
    lx = box[0, 1] - box[0, 0]
    ly = box[1, 1] - box[1, 0]
    lz = box[2, 1] - box[2, 0]

    # Try ratios — find integer Nx,Ny,Nz whose product = n_cells
    # and whose ratios approximately match lx:ly:lz
    best = None
    best_err = 1e9
    cbrt = int(round(n_cells ** (1/3)))
    for nx in range(max(1, cbrt - 20), cbrt + 40):
        if n_cells % nx != 0:
            continue
        rem = n_cells // nx
        sqrt_rem = int(round(rem ** 0.5))
        for ny in range(max(1, sqrt_rem - 20), sqrt_rem + 20):
            if rem % ny != 0:
                continue
            nz = rem // ny
            # Score by how well nx/ny/nz match the box aspect ratio
            expected_ny = nx * (ly / lx)
            expected_nz = nx * (lz / lx)
            err = abs(ny - expected_ny) + abs(nz - expected_nz)
            if err < best_err:
                best_err = err
                best = (nx, ny, nz)

    if best is None:
        # Fallback: assume cube
        cbrt = int(round(n_cells ** (1/3)))
        best = (cbrt, cbrt, n_cells // (cbrt * cbrt))

    return best


def build_cell_centres(n_cells, box, col_names, data):
    """
    Build (n_cells, 3) array of cell centre coordinates.
    SPARTA cell ordering: x varies fastest, then y, then z.
    """
    nx, ny, nz = infer_grid_dims(n_cells, box)
    print(f"  Inferred grid: {nx} x {ny} x {nz} = {nx*ny*nz} cells "
          f"(dump has {n_cells})")

    # Cell centres along each axis
    xs = np.linspace(box[0, 0], box[0, 1], nx, endpoint=False) + \
         (box[0, 1] - box[0, 0]) / (2 * nx)
    ys = np.linspace(box[1, 0], box[1, 1], ny, endpoint=False) + \
         (box[1, 1] - box[1, 0]) / (2 * ny)
    zs = np.linspace(box[2, 0], box[2, 1], nz, endpoint=False) + \
         (box[2, 1] - box[2, 0]) / (2 * nz)

    # SPARTA ordering: x fastest
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')
    centres = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

    # Trim/pad if inferred count doesn't match exactly
    if len(centres) > n_cells:
        centres = centres[:n_cells]
    elif len(centres) < n_cells:
        pad = np.zeros((n_cells - len(centres), 3))
        centres = np.vstack([centres, pad])

    return centres


# ---------------------------------------------------------------------------
# 3. Write VTK UnstructuredGrid (.vtu) — ASCII XML format
#    Each SPARTA cell is represented as a VTK_VERTEX (point) at its centre.
#    This is the correct representation for cell-centred DSMC data.
# ---------------------------------------------------------------------------

def write_vtu(path, centres, col_names, data):
    """
    Write a VTK UnstructuredGrid XML file (.vtu) with:
      - one point per SPARTA cell (at cell centre)
      - one VTK_VERTEX cell per point
      - one scalar field per SPARTA compute column
    """
    n = len(centres)

    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" '
                'byte_order="LittleEndian">\n')
        f.write('  <UnstructuredGrid>\n')
        f.write(f'  <Piece NumberOfPoints="{n}" NumberOfCells="{n}">\n')

        # --- Points ---
        f.write('    <Points>\n')
        f.write('      <DataArray type="Float64" NumberOfComponents="3" '
                'format="ascii">\n')
        for cx, cy, cz in centres:
            f.write(f'        {cx:.6e} {cy:.6e} {cz:.6e}\n')
        f.write('      </DataArray>\n')
        f.write('    </Points>\n')

        # --- Cells (one VTK_VERTEX per point) ---
        f.write('    <Cells>\n')
        # connectivity: each cell references one point
        f.write('      <DataArray type="Int32" Name="connectivity" '
                'format="ascii">\n')
        f.write('        ' + ' '.join(str(i) for i in range(n)) + '\n')
        f.write('      </DataArray>\n')
        # offsets: cumulative count of connectivity entries
        f.write('      <DataArray type="Int32" Name="offsets" '
                'format="ascii">\n')
        f.write('        ' + ' '.join(str(i+1) for i in range(n)) + '\n')
        f.write('      </DataArray>\n')
        # types: 1 = VTK_VERTEX
        f.write('      <DataArray type="UInt8" Name="types" '
                'format="ascii">\n')
        f.write('        ' + ' '.join(['1'] * n) + '\n')
        f.write('      </DataArray>\n')
        f.write('    </Cells>\n')

        # --- Point data (SPARTA compute columns) ---
        # Map raw column names to friendly names
        friendly = {
            "c_cgrid[1]": "nrho_m3",
            "c_cgrid[2]": "massrho_kg_m3",
            "c_cgrid[3]": "vel_x_m_s",
            "c_cgrid[4]": "vel_y_m_s",
            "c_cgrid[5]": "vel_z_m_s",
            "c_cgrid[6]": "temp_K",
        }

        f.write('    <PointData>\n')
        for i, col in enumerate(col_names):
            if i >= data.shape[1]:
                break
            name = friendly.get(col, col.replace("[", "_").replace("]", ""))
            f.write(f'      <DataArray type="Float64" Name="{name}" '
                    f'format="ascii">\n')
            vals = data[:, i]
            for v in vals:
                f.write(f'        {v:.6e}\n')
            f.write('      </DataArray>\n')

        # Also write velocity as a 3-component vector if all three are present
        u_idx = next((i for i, c in enumerate(col_names) if c == "c_cgrid[3]"), None)
        v_idx = next((i for i, c in enumerate(col_names) if c == "c_cgrid[4]"), None)
        w_idx = next((i for i, c in enumerate(col_names) if c == "c_cgrid[5]"), None)
        if all(x is not None for x in [u_idx, v_idx, w_idx]):
            f.write('      <DataArray type="Float64" Name="velocity_m_s" '
                    'NumberOfComponents="3" format="ascii">\n')
            for row in data:
                f.write(f'        {row[u_idx]:.6e} {row[v_idx]:.6e} '
                        f'{row[w_idx]:.6e}\n')
            f.write('      </DataArray>\n')

        f.write('    </PointData>\n')
        f.write('  </Piece>\n')
        f.write('  </UnstructuredGrid>\n')
        f.write('</VTKFile>\n')


# ---------------------------------------------------------------------------
# 4. Write ParaView Data (.pvd) collection file
#    Opening this single file in ParaView gives access to the full
#    time series with the animation controls.
# ---------------------------------------------------------------------------

def write_pvd(pvd_path, timestep_file_pairs):
    with open(pvd_path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1">\n')
        f.write('  <Collection>\n')
        for ts, vtu_file in sorted(timestep_file_pairs):
            f.write(f'    <DataSet timestep="{ts}" '
                    f'file="{os.path.basename(vtu_file)}"/>\n')
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Convert SPARTA grid dump files to VTK .vtu for ParaView")
    ap.add_argument("pattern",
                    help='Glob pattern for SPARTA grid dumps, '
                         'e.g. "grid_flow.*.vtp"')
    ap.add_argument("--outdir", default=".",
                    help="Output directory for .vtu files (default: current dir)")
    ap.add_argument("--pvd", default="grid_flow.pvd",
                    help="Name of the PVD collection file (default: grid_flow.pvd)")
    args = ap.parse_args()

    dump_files = sorted(
        glob.glob(args.pattern),
        key=lambda p: int(re.search(r"(\d+)", Path(p).stem).group(1))
    )

    if not dump_files:
        print(f"ERROR: no files matching '{args.pattern}'", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Found {len(dump_files)} dump file(s). Converting...")

    pvd_pairs = []

    for path in dump_files:
        print(f"\nProcessing {path} ...")
        timestep, n_cells, box, col_names, data = parse_sparta_grid_dump(path)
        print(f"  Timestep: {timestep}  |  Cells: {n_cells}  |  "
              f"Columns: {col_names}")

        centres = build_cell_centres(n_cells, box, col_names, data)

        stem     = Path(path).stem
        out_name = stem.replace("vtp", "").strip(".") or stem
        out_path = os.path.join(args.outdir, f"{out_name}.vtu")

        write_vtu(out_path, centres, col_names, data)
        print(f"  Written: {out_path}")

        pvd_pairs.append((timestep, out_path))

    pvd_path = os.path.join(args.outdir, args.pvd)
    write_pvd(pvd_path, pvd_pairs)
    print(f"\nPVD collection file: {pvd_path}")
    print(f"\nIn ParaView: File -> Open -> {args.pvd}")
    print("Then click Apply. Use the time controls to scrub through timesteps.")
    print("Color by 'nrho_m3' (AO density) or 'temp_K' (translational temperature).")


if __name__ == "__main__":
    main()
