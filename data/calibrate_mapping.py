"""
Interactive calibration: tune the coordinate mapping from keyboard to robot frame.

Usage:
    python data/calibrate_mapping.py

Edit the three angles below, run it, and see how each key maps to robot-space directions.
Once satisfied, copy the three angle values back into dataset.py.
"""

import numpy as np

# ====== TUNE THESE THREE VALUES ======
ROTATION_DEG = 100.0
TILT_DEG     = -36.0
XY_ROT_DEG   = 36.0

# ====== Key-to-mouse-axis mapping (which mouse axis each key drives) ======
# fmt: off
KEY_MAP = {
    # translation (mouse frame x,y,z)
    "w": ("trans",  0, +1),   "s": ("trans",  0, -1),   # y-axis (forward/back)
    "a": ("trans",  1, -1),   "d": ("trans",  1, +1),   # x-axis (left/right)
    "r": ("trans",  2, +1),   "f": ("trans",  2, -1),   # z-axis (up/down)
    # rotation (mouse frame rx,ry,rz)
    "o": ("rot",    2, +1),   "k": ("rot",    2, -1),   # rz (yaw)
    "p": ("rot",    1, +1),   "l": ("rot",    1, -1),   # ry (pitch)
    "m": ("rot",    0, +1),   "n": ("rot",    0, -1),   # rx (roll)
}
# fmt: on


def compute_rotation(theta_deg, tilt_deg, xy_rot_deg):
    """Compute 3x3 rotation matrix and basis vectors from three Euler-like angles."""
    theta, tilt, phi = np.radians([theta_deg, tilt_deg, xy_rot_deg])

    # Step 1 — base vector u in YZ plane
    u_base = np.array([0, -np.sin(theta), np.cos(theta)])
    zenith = np.array([1, 0, 0])

    # Step 2 — orthogonal v, then tilt v toward zenith
    v_base_h = np.cross(zenith, u_base)
    u = u_base
    v = v_base_h * np.cos(tilt) + zenith * np.sin(tilt)
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)

    # Step 3 — in-plane rotation by phi
    u_final = u * np.cos(phi) - v * np.sin(phi)
    v_final = u * np.sin(phi) + v * np.cos(phi)

    # Step 4 — third axis (right-handed: w = v x u)
    w_final = np.cross(v_final, u_final)

    # Rotation matrix: robot_vec = R @ mouse_vec
    # mouse x → v_final, mouse y → u_final, mouse z → w_final
    R = np.column_stack([v_final, u_final, w_final])
    return R, u_final, v_final, w_final


def main():
    R, u, v, w = compute_rotation(ROTATION_DEG, TILT_DEG, XY_ROT_DEG)

    print("=" * 62)
    print(f"  ROTATION = {ROTATION_DEG}   TILT = {TILT_DEG}   XY_ROT = {XY_ROT_DEG}")
    print("=" * 62)

    # ---- basis vectors ----
    print("\nBasis vectors (robot frame):")
    print(f"  u (forward)  = [{u[0]:+7.4f}  {u[1]:+7.4f}  {u[2]:+7.4f}]")
    print(f"  v (lateral)  = [{v[0]:+7.4f}  {v[1]:+7.4f}  {v[2]:+7.4f}]")
    print(f"  w (up)       = [{w[0]:+7.4f}  {w[1]:+7.4f}  {w[2]:+7.4f}]")

    # ---- orthogonality check ----
    dot_uv = np.dot(u, v)
    dot_uw = np.dot(u, w)
    dot_vw = np.dot(v, w)
    print(f"\nOrthogonality: u.v={dot_uv:+.6f}  u.w={dot_uw:+.6f}  v.w={dot_vw:+.6f}")
    print(f"  (all should be ~0)")

    # ---- rotation matrix ----
    print("\nRotation matrix R (robot_vec = R @ mouse_vec):")
    for row in R:
        print(f"  [{row[0]:+7.4f}  {row[1]:+7.4f}  {row[2]:+7.4f}]")

    # ---- per-key mapping ----
    print("\n" + "-" * 62)
    print("Key → robot-space direction (unit velocity):")
    print("-" * 62)

    trans_keys = []
    rot_keys = []
    for key, (kind, axis, sign) in sorted(KEY_MAP.items()):
        mouse_vec = np.zeros(3)
        mouse_vec[axis] = sign
        robot_vec = R @ mouse_vec
        label = "trans" if kind == "trans" else "rot "
        line = (
            f"  [{label}]  {key}  "
            f"→  [{robot_vec[0]:+7.4f}  {robot_vec[1]:+7.4f}  {robot_vec[2]:+7.4f}]"
        )
        if kind == "trans":
            trans_keys.append(line)
        else:
            rot_keys.append(line)

    for line in trans_keys:
        print(line)
    for line in rot_keys:
        print(line)

    # ---- combo examples ----
    print("\n" + "-" * 62)
    print("Common combos (translation only, unit magnitude each axis):")
    print("-" * 62)

    combos = [
        ("W+D (forward+right)", [0, +1, 0, +1, 0, 0]),
        ("W+A (forward+left) ", [0, +1, 0, -1, 0, 0]),
        ("W+R (forward+up)   ", [0, +1, +1,  0, 0, 0]),
        ("D+R (right+up)     ", [0,  0, +1, +1, 0, 0]),
    ]
    for name, (_, my, mz, mx, _, _) in combos:
        mouse_trans = np.array([mx, my, mz])
        robot = R @ mouse_trans
        norm = np.linalg.norm(robot)
        print(f"  {name} → [{robot[0]:+7.4f}  {robot[1]:+7.4f}  {robot[2]:+7.4f}]  "
              f"(norm={norm:.3f})")


if __name__ == "__main__":
    main()
