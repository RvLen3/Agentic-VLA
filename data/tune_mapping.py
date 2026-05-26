"""
Minimal keyboard teleop — tune the coordinate mapping against the real robot.

Usage:
    python data/tune_mapping.py

Keys:
    WASD / RF  = translation    OKPLMN = rotation    G = toggle gripper
    Q          = quit

Calibration steps (do in order):
  1. Press W. Adjust ROTATION_DEG until forward/backward is correct.
  2. Press A/D. Adjust XY_ROT_DEG until left/right is correct.
  3. Press W+A (diagonal). Adjust TILT_DEG until the WASD plane aligns
     with the work surface. RF will automatically be perpendicular.

Then copy the values back to dataset.py.
"""
import time
import threading
import numpy as np

# ── robot connection ──────────────────────────────────────────
ROBOT_IP = "192.168.1.88"

from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface

# ── terminal keyboard (non-blocking) ──────────────────────────
import sys
import select
import tty
import termios


def setup_terminal():
    global _orig
    _orig = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())


def restore_terminal():
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _orig)


def read_key():
    """Non-blocking single-char read. Returns lowercase char or None."""
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        ch = sys.stdin.read(1)
        return ch.lower() if ch else None
    return None


# ============================================================
#  TUNE THESE THREE ANGLES (degrees)
# ============================================================
ROTATION_DEG = 0.0   # step 1: W/S direction in horizontal plane
TILT_DEG     = 0.0   # step 3: tilt WASD plane to match work surface
XY_ROT_DEG   = 0.0    # step 2: A/D direction in horizontal plane
# ============================================================

# speed
TRANS_SPEED = 0.03   # m/s
ROT_SPEED   = 0.15   # rad/s
SPEEDL_ACC  = 0.5
SPEEDL_TIME = 0.05


def compute_rotation(theta_deg, tilt_deg, xy_rot_deg):
    """Three-step mapping: keyboard frame -> robot base frame.

    keyboard frame:  x = left/right (A/D)
                     y = forward/back (W/S)
                     z = up/down (R/F), auto-perpendicular to WASD plane
    """
    theta, tilt, phi = np.radians([theta_deg, tilt_deg, xy_rot_deg])

    # Step 1 — base vector u in YZ plane (controls W/S forward)
    u_base = np.array([0, -np.sin(theta), np.cos(theta)])
    zenith = np.array([1, 0, 0])

    # Step 2 — orthogonal v, then tilt v toward zenith
    v_base_h = np.cross(zenith, u_base)
    u = u_base
    v = v_base_h * np.cos(tilt) + zenith * np.sin(tilt)
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)

    # Step 3 — in-plane rotation by phi (controls A/D left/right)
    u_final = u * np.cos(phi) - v * np.sin(phi)
    v_final = u * np.sin(phi) + v * np.cos(phi)

    # Step 4 — third axis (R/F up/down), right-handed: w = v x u
    w_final = np.cross(v_final, u_final)

    # Rotation matrix: robot_vec = R @ mouse_vec
    # mouse x -> v_final, mouse y -> u_final, mouse z -> w_final
    R = np.column_stack([v_final, u_final, w_final])
    return R, u_final, v_final, w_final


class ArmCtrl:
    """50 Hz speedL thread."""
    def __init__(self, ip):
        self.vel = [0.0]*6
        self._lock = threading.Lock()
        self._run = False
        self._c = None

    def start(self):
        try:
            self._c = RTDEControlInterface(ROBOT_IP)
        except Exception as e:
            print(f"RTDE Control failed: {e}")
            return False
        try:
            self._c.speedL([0.0]*6, SPEEDL_ACC, 0.0)
        except Exception:
            try:
                self._c.unlockProtectiveStop()
                time.sleep(1.5)
                self._c.speedL([0.0]*6, SPEEDL_ACC, 0.0)
            except Exception as e:
                print(f"speedL failed: {e}")
                return False
        self._run = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        return True

    def _loop(self):
        while self._run:
            t0 = time.perf_counter()
            with self._lock:
                v = list(self.vel)
            try:
                self._c.speedL(v, SPEEDL_ACC, SPEEDL_TIME)
            except Exception:
                pass
            dt = 0.02 - (time.perf_counter() - t0)
            if dt > 0:
                time.sleep(dt)

    def set(self, v):
        with self._lock:
            self.vel = list(v)

    def stop(self):
        self._run = False
        if self._c:
            try:
                self._c.speedL([0.0]*6, SPEEDL_ACC, 0.0)
                self._c.speedStop()
                self._c.disconnect()
            except Exception:
                pass


def main():
    R, u, v, w = compute_rotation(ROTATION_DEG, TILT_DEG, XY_ROT_DEG)

    print(f"ROTATION={ROTATION_DEG}  TILT={TILT_DEG}  XY_ROT={XY_ROT_DEG}")
    print("Basis vectors (robot frame):")
    print(f"  u (forward) = [{u[0]:+7.4f}  {u[1]:+7.4f}  {u[2]:+7.4f}]")
    print(f"  v (lateral) = [{v[0]:+7.4f}  {v[1]:+7.4f}  {v[2]:+7.4f}]")
    print(f"  w (up)      = [{w[0]:+7.4f}  {w[1]:+7.4f}  {w[2]:+7.4f}]")
    print(f"Orthogonality: u.v={np.dot(u,v):+.6f}  u.w={np.dot(u,w):+.6f}  v.w={np.dot(v,w):+.6f}")
    print("Rotation matrix:")
    for row in R:
        print(f"  [{row[0]:+7.4f}  {row[1]:+7.4f}  {row[2]:+7.4f}]")

    ctrl = ArmCtrl(ROBOT_IP)
    if not ctrl.start():
        print("Failed to start control thread. Check robot mode (Remote Control).")
        return

    gripper = 0.0  # 0=closed, 1=open
    print("\nWASD/RF=move  OKPLMN=rot  G=grip-toggle  Q=quit\n")

    setup_terminal()
    try:
        while True:
            # drain key buffer (10ms)
            keys = set()
            stop = False
            t_dead = time.perf_counter() + 0.01
            while time.perf_counter() < t_dead:
                k = read_key()
                if k is None:
                    break
                if k == 'q':
                    stop = True
                    break
                keys.add(k)
            if stop:
                break

            # gripper toggle
            if 'g' in keys:
                gripper = 1.0 - gripper
                print(f"  gripper -> {'OPEN' if gripper > 0.5 else 'CLOSED'}")

            # build velocity
            vx = vy = vz = rx = ry = rz = 0.0
            if 'd' in keys: vx += TRANS_SPEED
            if 'a' in keys: vx -= TRANS_SPEED
            if 'w' in keys: vy += TRANS_SPEED
            if 's' in keys: vy -= TRANS_SPEED
            if 'r' in keys: vz += TRANS_SPEED
            if 'f' in keys: vz -= TRANS_SPEED
            if 'o' in keys: rz += ROT_SPEED
            if 'k' in keys: rz -= ROT_SPEED
            if 'p' in keys: ry += ROT_SPEED
            if 'l' in keys: ry -= ROT_SPEED
            if 'm' in keys: rx += ROT_SPEED
            if 'n' in keys: rx -= ROT_SPEED

            robot_t = R @ np.array([vx, vy, vz])
            robot_r = R @ np.array([rx, ry, rz])
            ctrl.set([*robot_t, *robot_r])

            time.sleep(0.02)
    finally:
        restore_terminal()
        ctrl.stop()
        print("done")


if __name__ == "__main__":
    main()
