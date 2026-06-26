#!/usr/bin/env python3
"""
fluence_extract.py — Parse SPARTA surf_AO dump files and compute per-element
AO fluence, pressure, and shear force for the LAMMPS bombardment stage.

Matches goce_ao_dsmc.sparta as of current revision:

  compute  csurf_AO  surf all AO  nflux press shx shy shz ke
  dump     dsurf_AO  surf all 1000  surf_AO.*.dat
           id c_csurf_AO[1] c_csurf_AO[2] c_csurf_AO[3] c_csurf_AO[4] c_csurf_AO[5]

Column mapping in surf_AO.*.dat (after id):
  col 1 → nflux   (AO particles / m² / s)
  col 2 → press   (Pa)
  col 3 → shx     (Pa, shear stress x-component)
  col 4 → shy     (Pa, shear stress y-component)
  col 5 → shz     (Pa, shear stress z-component)
  (ke = col 6 is not dumped in this script revision)

Usage:
    python fluence_extract.py \\
        --surf    Al_Slab.surf \\
        --dumps   "surf_AO.*.dat" \\
        --dt      1e-3 \\
        --dump-interval 1000 \\
        --out     fluence_map.csv

Output fluence_map.csv columns:
    surf_id, cx, cy, cz, area_m2, area_cm2,
    nx, ny, nz,
    fluence_AO_per_cm2,
    mean_press_Pa, mean_shx_Pa, mean_shy_Pa, mean_shz_Pa,
    mean_angle_deg, priority
"""

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Column indices in surf_AO.*.dat (0-based after the id column)
# id | nflux | press | shx | shy | shz
# ---------------------------------------------------------------------------
COL_NFLUX = 0
COL_PRESS = 1
COL_SHX   = 2
COL_SHY   = 3
COL_SHZ   = 4
N_DATA_COLS = 5   # columns after id


# ---------------------------------------------------------------------------
# 1.  Parse SPARTA surf dump files
# ---------------------------------------------------------------------------

def parse_surf_dump(path):
    """
    Parse one SPARTA surf dump file.
    Returns dict: surf_id (int) -> np.array of length N_DATA_COLS

    SPARTA surf dump format:
        ITEM: TIMESTEP
        <N>
        ITEM: NUMBER OF SURFS
        <M>
        ITEM: SURFS id col1 col2 ...
        id v1 v2 ...
        ...
    """
    data = {}
    in_data = False

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("ITEM: SURFS"):
                in_data = True
                continue
            if line.startswith("ITEM:"):
                in_data = False
                continue
            if not in_data:
                continue
            tok = line.split()
            if len(tok) < N_DATA_COLS + 1:
                continue
            try:
                sid  = int(tok[0])
                vals = np.array([float(v) for v in tok[1:N_DATA_COLS+1]])
                data[sid] = vals
            except ValueError:
                continue

    return data


def accumulate_dumps(dump_files, dt, dump_interval):
    """
    Integrate nflux over all dump snapshots to get fluence, and
    accumulate press/shear as running sums for later averaging.

    Returns per surf_id:
        fluence_m2   — AO particles / m²  (time-integrated nflux)
        press_sum    — sum of press values (Pa) across snapshots
        shx_sum      — sum of shx across snapshots
        shy_sum      — sum of shy across snapshots
        shz_sum      — sum of shz across snapshots
        n_snaps      — number of snapshots with nonzero nflux
    """
    fluence_m2 = {}
    press_sum  = {}
    shx_sum    = {}
    shy_sum    = {}
    shz_sum    = {}
    n_snaps    = {}

    sorted_files = sorted(
        dump_files,
        key=lambda p: int(re.search(r"(\d+)", Path(p).stem).group(1))
    )

    if not sorted_files:
        print("ERROR: no dump files matched the pattern.", file=sys.stderr)
        sys.exit(1)

    for path in sorted_files:
        snap = parse_surf_dump(path)
        for sid, vals in snap.items():
            nflux = vals[COL_NFLUX]
            press = vals[COL_PRESS]
            shx   = vals[COL_SHX]
            shy   = vals[COL_SHY]
            shz   = vals[COL_SHZ]

            # Fluence increment: nflux [particles/m²/s] × dt [s] × dump_interval [steps]
            dF = nflux * dt * dump_interval

            fluence_m2[sid] = fluence_m2.get(sid, 0.0) + dF
            press_sum[sid]  = press_sum.get(sid, 0.0)  + press
            shx_sum[sid]    = shx_sum.get(sid, 0.0)    + shx
            shy_sum[sid]    = shy_sum.get(sid, 0.0)    + shy
            shz_sum[sid]    = shz_sum.get(sid, 0.0)    + shz
            n_snaps[sid]    = n_snaps.get(sid, 0)      + (1 if nflux > 0 else 0)

    return fluence_m2, press_sum, shx_sum, shy_sum, shz_sum, n_snaps, len(sorted_files)


# ---------------------------------------------------------------------------
# 2.  Parse SPARTA surface file for geometry
# ---------------------------------------------------------------------------

def parse_sparta_surf(path):
    """
    Read satellite_goce.surf and compute per-triangle centroid, area, normal.
    Returns dict: tri_id (int) -> dict with cx,cy,cz,area_m2,nx,ny,nz
    """
    points    = {}
    triangles = {}
    section   = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("SPARTA"):
                continue
            if line == "Points":
                section = "points"
                continue
            if line == "Triangles":
                section = "tris"
                continue
            if re.match(r"^\d+ (points|triangles)$", line):
                continue

            tok = line.split()
            if not tok:
                continue

            if section == "points":
                pid = int(tok[0])
                points[pid] = np.array([float(tok[1]),
                                        float(tok[2]),
                                        float(tok[3])])
            elif section == "tris":
                tid      = int(tok[0])
                v0, v1, v2 = points[int(tok[1])], points[int(tok[2])], points[int(tok[3])]
                cross    = np.cross(v1 - v0, v2 - v0)
                area     = 0.5 * np.linalg.norm(cross)
                normal   = cross / (np.linalg.norm(cross) + 1e-300)
                centroid = (v0 + v1 + v2) / 3.0
                triangles[tid] = dict(
                    cx=centroid[0], cy=centroid[1], cz=centroid[2],
                    area_m2=area,
                    nx=normal[0], ny=normal[1], nz=normal[2]
                )

    return triangles


# ---------------------------------------------------------------------------
# 3.  Impact angle (flow in -z direction per the SPARTA script)
# ---------------------------------------------------------------------------

def impact_angle(nx, ny, nz):
    """
    Angle of incidence between the AO ram flow (-z) and the surface normal.
    0° = normal incidence (ram-facing panel), 90° = grazing.
    """
    flow   = np.array([0.0, 0.0, -1.0])      # vstream direction unit vector
    normal = np.array([nx, ny, nz])
    cos_t  = np.clip(np.dot(-flow, normal), -1.0, 1.0)
    return np.degrees(np.arccos(cos_t))


# ---------------------------------------------------------------------------
# 4.  Priority classification for LAMMPS patch selection
# ---------------------------------------------------------------------------

def priority_score(fluence_cm2, angle_deg):
    """Higher fluence + lower angle = higher score = higher MD priority."""
    if fluence_cm2 <= 0:
        return -1.0
    return fluence_cm2 * max(1e-3, np.cos(np.radians(angle_deg)))


# ---------------------------------------------------------------------------
# 5.  Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Extract per-surface AO fluence and stress from SPARTA surf_AO dumps")
    ap.add_argument("--surf",           required=True,
                    help="SPARTA surface file (Al_Slab.surf)")
    ap.add_argument("--dumps",          required=True,
                    help='Glob for AO dump files, e.g. "surf_AO.*.dat"')
    ap.add_argument("--dt",             type=float, default=1e-3,
                    help="SPARTA timestep in seconds (default: 1e-3)")
    ap.add_argument("--dump-interval",  type=int,   default=1000,
                    dest="dump_interval",
                    help="Dump frequency in steps (default: 1000)")
    ap.add_argument("--out",            default="fluence_map.csv",
                    help="Output CSV (default: fluence_map.csv)")
    ap.add_argument("--top-n",          type=int,   default=20, dest="top_n",
                    help="Number of patches to tag as priority tiers (default: 20)")
    args = ap.parse_args()

    # --- find dumps ---
    dump_files = sorted(glob.glob(args.dumps))
    if not dump_files:
        print(f"ERROR: no files matching '{args.dumps}'", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(dump_files)} dump file(s) matching '{args.dumps}'")

    # --- geometry ---
    print(f"Parsing geometry from {args.surf} ...")
    geo = parse_sparta_surf(args.surf)
    print(f"  {len(geo)} triangles loaded.")

    # --- accumulate flux ---
    print("Accumulating AO flux from dumps ...")
    fluence_m2, press_sum, shx_sum, shy_sum, shz_sum, n_snaps, nf = \
        accumulate_dumps(dump_files, args.dt, args.dump_interval)
    print(f"  Processed {nf} dump file(s).")

    # --- build rows ---
    rows = []
    for sid, g in geo.items():
        F_m2    = fluence_m2.get(sid, 0.0)
        F_cm2   = F_m2 * 1e-4                        # m⁻² → cm⁻²
        ns      = max(n_snaps.get(sid, 1), 1)
        p_mean  = press_sum.get(sid, 0.0) / ns
        shx_m   = shx_sum.get(sid, 0.0)  / ns
        shy_m   = shy_sum.get(sid, 0.0)  / ns
        shz_m   = shz_sum.get(sid, 0.0)  / ns
        angle   = impact_angle(g["nx"], g["ny"], g["nz"])
        score   = priority_score(F_cm2, angle)

        rows.append(dict(
            surf_id          = sid,
            cx               = g["cx"],
            cy               = g["cy"],
            cz               = g["cz"],
            area_m2          = g["area_m2"],
            area_cm2         = g["area_m2"] * 1e4,
            nx               = g["nx"],
            ny               = g["ny"],
            nz               = g["nz"],
            fluence_AO_per_cm2 = F_cm2,
            mean_press_Pa    = p_mean,
            mean_shx_Pa      = shx_m,
            mean_shy_Pa      = shy_m,
            mean_shz_Pa      = shz_m,
            mean_angle_deg   = angle,
            _score           = score,
        ))

    # --- sort and assign priority tiers ---
    rows.sort(key=lambda r: r["_score"], reverse=True)
    n1 = max(1, args.top_n // 3)
    for i, r in enumerate(rows):
        if r["_score"] <= 0:
            r["priority"] = "skip"
        elif i < n1:
            r["priority"] = "tier1_high"
        elif i < 2 * n1:
            r["priority"] = "tier2_mid"
        elif i < args.top_n:
            r["priority"] = "tier3_low"
        else:
            r["priority"] = "background"

    # --- write CSV ---
    header = ("surf_id,cx,cy,cz,area_m2,area_cm2,"
              "nx,ny,nz,"
              "fluence_AO_per_cm2,"
              "mean_press_Pa,mean_shx_Pa,mean_shy_Pa,mean_shz_Pa,"
              "mean_angle_deg,priority")

    with open(args.out, "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(
                f"{r['surf_id']},"
                f"{r['cx']:.6e},{r['cy']:.6e},{r['cz']:.6e},"
                f"{r['area_m2']:.6e},{r['area_cm2']:.6e},"
                f"{r['nx']:.6f},{r['ny']:.6f},{r['nz']:.6f},"
                f"{r['fluence_AO_per_cm2']:.6e},"
                f"{r['mean_press_Pa']:.6e},"
                f"{r['mean_shx_Pa']:.6e},{r['mean_shy_Pa']:.6e},{r['mean_shz_Pa']:.6e},"
                f"{r['mean_angle_deg']:.2f},"
                f"{r['priority']}\n"
            )

    print(f"\nWrote {len(rows)} rows to {args.out}")

    # --- console summary ---
    active = [r for r in rows if r["fluence_AO_per_cm2"] > 0]
    if active:
        F_vals = [r["fluence_AO_per_cm2"] for r in active]
        print(f"\n  AO fluence summary ({len(active)} patches with nonzero flux)")
        print(f"  Min  : {min(F_vals):.3e} AO/cm²")
        print(f"  Max  : {max(F_vals):.3e} AO/cm²")
        print(f"  Mean : {np.mean(F_vals):.3e} AO/cm²")

        t1 = [r for r in rows if r["priority"] == "tier1_high"]
        print(f"\n  Tier-1 patches (highest priority for LAMMPS MD): {len(t1)}")
        print(f"  {'ID':>6}  {'fluence/cm²':>13}  {'press(Pa)':>10}  "
              f"{'|shear|(Pa)':>11}  {'angle°':>7}")
        for r in t1:
            shear_mag = np.sqrt(r["mean_shx_Pa"]**2 +
                                r["mean_shy_Pa"]**2 +
                                r["mean_shz_Pa"]**2)
            print(f"  {r['surf_id']:>6}  {r['fluence_AO_per_cm2']:>13.3e}"
                  f"  {r['mean_press_Pa']:>10.3e}"
                  f"  {shear_mag:>11.3e}"
                  f"  {r['mean_angle_deg']:>7.1f}")
    else:
        print("\n  WARNING: No nonzero AO flux found.")
        print("  Check that surf_AO.*.dat files exist and contain data.")
        print("  Confirm the production run completed (not just equilibration).")

    print(f"\nNext step: feed fluence_map.csv into the LAMMPS bombardment script.")


if __name__ == "__main__":
    main()