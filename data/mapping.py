"""
Coordinate mapping: keyboard intuition -> robot base frame.

Tune the three angles below by running dataset.py and observing which way
the robot moves.  Each angle has a simple, independent effect.

How to calibrate (do in order):

  1. Press W (forward).  If robot moves left/right instead, adjust RZ.
  2. Press W (forward).  If robot moves up/down instead, adjust RY.
  3. Press R (up).      If "up" goes sideways instead of up, adjust RX.
  4. Press D (right).   If right moves left, change sign of the angle.
"""

import numpy as np

# ============================================================
# TUNE THESE THREE ANGLES (degrees)
# ============================================================
RX_DEG = 0.0    # rotate around robot X — fixes "up" direction
RY_DEG = 0.0    # rotate around robot Y — fixes forward/up mix
RZ_DEG = 0.0    # rotate around robot Z — fixes forward/left mix
# ============================================================


def _rot_x(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _rot_y(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def _rot_z(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def build_rotation(rx_deg=RX_DEG, ry_deg=RY_DEG, rz_deg=RZ_DEG):
    """
    Rotation matrix that maps keyboard-intuition vectors to robot frame.

    keyboard frame:  x = left/right (A/D)
                     y = forward/back (W/S)
                     z = up/down (R/F)

    robot frame:  whatever your UR robot uses (X forward, Y left, Z up)

    Rotation order: RZ @ RY @ RX  (applied to keyboard vector)
    """
    return _rot_z(rz_deg) @ _rot_y(ry_deg) @ _rot_x(rx_deg)


# Pre-computed matrix used by dataset.py
R = build_rotation()


def apply(mouse_trans, mouse_rot):
    """Map keyboard velocity [vx,vy,vz, rx,ry,rz] -> robot velocity."""
    t = np.asarray(mouse_trans, dtype=float)
    r = np.asarray(mouse_rot, dtype=float)
    return [*(R @ t), *(R @ r)]


# ------- print current mapping (run this file directly) -------
if __name__ == "__main__":
    R = build_rotation()
    print(f"RX={RX_DEG}  RY={RY_DEG}  RZ={RZ_DEG}")
    print(f"Matrix:\n{R}")
    keys = {
        "W": [ 0, +1,  0], "S": [ 0, -1,  0],
        "A": [-1,  0,  0], "D": [+1,  0,  0],
        "R": [ 0,  0, +1], "F": [ 0,  0, -1],
    }
    for k, v in keys.items():
        r = R @ np.array(v)
        print(f"  {k} → [{r[0]:+7.4f}  {r[1]:+7.4f}  {r[2]:+7.4f}]")
