"""
UR7e 左臂 键盘控制示教数据采集
双 RealSense RGB + D405 Depth
完全通过 XML-RPC 控制机械臂移动和夹爪

使用前：
1. 示教器上启动 xvla control program（包含 rpc.get_target() 循环）
2. 确认本机 IP 为 192.168.1.100（或修改 XMLRPC_SERVER_IP）
3. 确认左臂两台 RealSense 摄像头已连接

控制原理：
  UR 示教器程序每 10ms 调用 rpc.get_target() 获取 7 个值：
    cmd[0:6] = 目标 TCP 位姿 [x, y, z, rx, ry, rz]
    cmd[6]   = 夹爪位置（0=全闭，100=全开）
  本脚本启动 XML-RPC 服务器，键盘输入转换为位姿增量更新 target。

键盘映射：
  W/S      → 前进/后退       R/F      → 上升/下降
  A/D      → 左移/右移
  O/K      → 左旋/右旋 (yaw)
  P/L      → 前倾/后仰 (pitch)
  M/N      → 手爪正旋/反旋 (roll)
  G        → 夹爪开/闭（按一次切换）
  Q        → 结束当前轨迹并保存
  松手即停（无按键时不更新目标位姿）

保存内容:
- images, images_wrist, depths_d405, tcp_poses, joint_positions, gripper, instruction, fps
"""

import cv2
import time
import random
import threading
import numpy as np
import pyrealsense2 as rs
from pathlib import Path
from xmlrpc.server import SimpleXMLRPCServer
from rtde_receive import RTDEReceiveInterface

# ── 终端非阻塞按键检测（跨平台）─────────────────────────────────────
try:
    import msvcrt
    _USE_MSVCRT = True
except ImportError:
    _USE_MSVCRT = False
    import sys
    import select
    import tty
    import termios


SCRIPT_VERSION = "LEFT_RGB_D405_XMLRPC_CONTROL_V1"
print(f"当前运行脚本版本: {SCRIPT_VERSION}")


# ============ 配置 ============
ARM_NAME = "left"
ROBOT_IP = "192.168.1.88"

# XML-RPC 服务器（本机，UR 程序连接的地址）
XMLRPC_SERVER_IP   = "192.168.1.100"
XMLRPC_SERVER_PORT = 50000

SAVE_DIR = Path(f"./raw_demos_{ARM_NAME}_third")
FPS = 30

VEGETABLES = [
    "red pepper", "green pepper", "yellow pepper",
    "corn", "purple sweet potato", "pumpkin",
]

MAX_FRAMES = 900
COLOR_WIDTH, COLOR_HEIGHT = 640, 480
DEPTH_WIDTH, DEPTH_HEIGHT = 640, 480
SAVE_WIDTH, SAVE_HEIGHT = 256, 256

# 键盘控制速度（位姿增量/帧，约 30fps）
TRANS_STEP = 0.003    # 每帧平移增量 (m)，0.003m * 30fps ≈ 9cm/s
ROT_STEP   = 0.01     # 每帧旋转增量 (rad)，0.01rad * 30fps ≈ 0.3rad/s

# 坐标映射
MAP_ROTATION_DEG = 100.0
MAP_TILT_DEG     = 35
MAP_XY_ROT_DEG   = 30

def get_teleop_rotation(theta_deg, tilt_deg, xy_rot_deg):
    theta, tilt, phi = np.radians([theta_deg, tilt_deg, xy_rot_deg])
    u_base = np.array([0, -np.sin(theta), np.cos(theta)])
    zenith = np.array([1, 0, 0])
    v_base_h = np.cross(zenith, u_base)
    u_orig = u_base
    v_orig = v_base_h * np.cos(tilt) + zenith * np.sin(tilt)
    u_orig /= np.linalg.norm(u_orig)
    v_orig /= np.linalg.norm(v_orig)
    u_final = u_orig * np.cos(phi) - v_orig * np.sin(phi)
    v_final = u_orig * np.sin(phi) + v_orig * np.cos(phi)
    w_final = np.cross(v_final, u_final)
    return np.column_stack([v_final, u_final, w_final])

_MOUSE2ROBOT = get_teleop_rotation(MAP_ROTATION_DEG, MAP_TILT_DEG, MAP_XY_ROT_DEG)
# ==============================

SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ============ XML-RPC 控制器（移动 + 夹爪）============

class RobotXMLRPCController:
    """
    通过 XML-RPC 同时控制机械臂位姿和夹爪。
    UR 示教器程序每 10ms 调用 get_target() 获取目标。
    """

    def __init__(self, server_ip, server_port, robot_ip):
        self._server_ip = server_ip
        self._server_port = server_port
        self._robot_ip = robot_ip
        self._lock = threading.Lock()
        # [x, y, z, rx, ry, rz, gripper_pos]
        self._target = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]
        self._server = None

    def _get_target(self):
        """XML-RPC 方法，UR 程序每 10ms 调用。必须返回 Python 原生 float 列表。"""
        with self._lock:
            return [float(v) for v in self._target]

    def start(self) -> bool:
        """读取当前位姿并启动 XML-RPC 服务器。"""
        try:
            rtde_r = RTDEReceiveInterface(self._robot_ip)
            tcp = rtde_r.getActualTCPPose()
            rtde_r.disconnect()
            with self._lock:
                self._target[0:6] = [float(v) for v in tcp]
            print(f"  [XMLRPC] 当前位姿: [{tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f}]")
        except Exception as e:
            print(f"  [XMLRPC] 读取位姿失败: {e}")
            return False

        try:
            self._server = SimpleXMLRPCServer(
                (self._server_ip, self._server_port),
                allow_none=True, logRequests=False,
            )
            self._server.register_function(self._get_target, "get_target")
            t = threading.Thread(target=self._server.serve_forever, daemon=True)
            t.start()
            print(f"  [XMLRPC] 服务器已启动: {self._server_ip}:{self._server_port}")
            return True
        except Exception as e:
            print(f"  [XMLRPC] 启动失败: {e}")
            return False

    def update_pose_from_robot(self):
        """从机器人读取当前位姿同步到 target（防止漂移）。"""
        try:
            rtde_r = RTDEReceiveInterface(self._robot_ip)
            tcp = rtde_r.getActualTCPPose()
            rtde_r.disconnect()
            with self._lock:
                self._target[0:6] = [float(v) for v in tcp]
        except Exception:
            pass

    def add_delta(self, dx, dy, dz, drx, dry, drz):
        """在当前目标位姿上叠加增量（键盘控制）。"""
        with self._lock:
            self._target[0] += float(dx)
            self._target[1] += float(dy)
            self._target[2] += float(dz)
            self._target[3] += float(drx)
            self._target[4] += float(dry)
            self._target[5] += float(drz)

    def set_gripper(self, pos: float):
        """设置夹爪位置（0=闭合，100=打开）。"""
        pos = max(0.0, min(100.0, pos))
        with self._lock:
            self._target[6] = pos

    def gripper_open(self):
        self.set_gripper(100.0)
        print("  [夹爪] → 打开")

    def gripper_close(self):
        self.set_gripper(0.0)
        print("  [夹爪] → 闭合")

    def stop(self):
        if self._server:
            self._server.shutdown()
            print("  [XMLRPC] 服务器已停止")


# ============ 工具函数 ============

def get_next_episode_id(save_dir):
    existing = sorted(save_dir.glob("episode_*.npz"))
    if not existing:
        return 0
    return max(int(f.stem.split("_")[-1]) for f in existing) + 1


def reconnect_rtde(old_rtde=None):
    if old_rtde is not None:
        try: old_rtde.disconnect()
        except Exception: pass
    rtde_r = RTDEReceiveInterface(ROBOT_IP)
    tcp = rtde_r.getActualTCPPose()
    print(f"  RTDE 已连接，TCP: [{tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f}]")
    return rtde_r


def start_realsense_pipelines():
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) < 2:
        raise RuntimeError(f"需要 2 个 RealSense，只找到 {len(devices)} 个")
    print(f"找到 {len(devices)} 个 RealSense 设备:")
    pipelines = []
    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"  - {name} (S/N: {serial})")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
        enable_depth = "D405" in name.upper()
        if enable_depth:
            config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)
        try:
            profile = pipeline.start(config)
            depth_scale, align = None, None
            if enable_depth:
                depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                align = rs.align(rs.stream.color)
            pipelines.append({"name": name, "serial": serial, "pipe": pipeline,
                              "enable_depth": enable_depth, "depth_scale": depth_scale, "align": align})
            print(f"  ▶ {name} 启动成功: {'RGB+Depth' if enable_depth else 'RGB'}")
        except Exception as e:
            for p in pipelines:
                try: p["pipe"].stop()
                except: pass
            raise RuntimeError(f"{name} 启动失败: {e}")
    print("预热中...")
    for _ in range(15):
        for p in pipelines:
            try: p["pipe"].wait_for_frames(timeout_ms=1000)
            except: pass
    return pipelines


def get_rgb_frame(p):
    frames = p["pipe"].wait_for_frames(timeout_ms=2000)
    if p.get("enable_depth", False):
        frames = p["align"].process(frames)
    cf = frames.get_color_frame()
    return np.asanyarray(cf.get_data()) if cf else None


def capture_frames(pipelines):
    frames, depth = [], None
    for p in pipelines:
        try:
            f = p["pipe"].wait_for_frames(timeout_ms=2000)
            if p.get("enable_depth", False):
                f = p["align"].process(f)
                df = f.get_depth_frame()
                depth = np.asanyarray(df.get_data()) if df else None
            cf = f.get_color_frame()
            frames.append(np.asanyarray(cf.get_data()) if cf else None)
        except:
            frames.append(None)
    if any(f is None for f in frames) or depth is None:
        return None, None, None
    return frames[0], frames[1], depth


def check_terminal_key():
    if _USE_MSVCRT:
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch in (b'\xe0', b'\x00'):
                msvcrt.getch()
                return None
            try: return ch.decode('utf-8').lower()
            except: return None
        return None
    else:
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            ch = sys.stdin.read(1)
            return ch.lower() if ch else None
        return None


_original_termios = None
def _setup_terminal():
    global _original_termios
    if not _USE_MSVCRT:
        _original_termios = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

def _restore_terminal():
    global _original_termios
    if not _USE_MSVCRT and _original_termios is not None:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _original_termios)
        _original_termios = None


def generate_episode_sequence():
    shuffled = VEGETABLES[:]
    random.shuffle(shuffled)
    sequence = []
    for veg in shuffled:
        sequence.append((f"pick up the {veg}", "pick_up"))
        sequence.append(("place it in the basket", "place"))
    return sequence


# ============ 主函数 ============

def main():
    pipelines = []
    rtde_r = None
    controller = None
    saved_count = 0

    try:
        # ── RealSense ─────────────────────────────────────────────────
        print("\n正在启动 RealSense...")
        pipelines = start_realsense_pipelines()
        pipe_main, pipe_wrist = pipelines[0], pipelines[1]

        # ── 确认视角 ──────────────────────────────────────────────────
        print(f"\n主视角: {pipe_main['name']}, 腕部: {pipe_wrist['name']}")
        print("终端按 's' 交换，Enter 确认")
        _setup_terminal()
        while True:
            img1, img2 = get_rgb_frame(pipe_main), get_rgb_frame(pipe_wrist)
            if img1 is not None and img2 is not None:
                v1, v2 = img1.copy(), img2.copy()
                cv2.putText(v1, f"Main: {pipe_main['name']}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(v2, f"Wrist: {pipe_wrist['name']}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.imshow("Preview", np.hstack([v1, v2]))
            cv2.waitKey(50)
            key = check_terminal_key()
            if key in ('\r', '\n'): break
            elif key == 's':
                pipe_main, pipe_wrist = pipe_wrist, pipe_main
                print(f"  交换 → 主:{pipe_main['name']}, 腕:{pipe_wrist['name']}")
        cv2.destroyAllWindows()
        _restore_terminal()

        active_pipelines = [pipe_main, pipe_wrist]
        pipe_d405 = next((p for p in active_pipelines if "D405" in p["name"].upper()), None)
        if not pipe_d405 or not pipe_d405.get("enable_depth"):
            raise RuntimeError("未找到 D405")
        print(f"\nD405: {pipe_d405['name']}, scale={pipe_d405['depth_scale']}")

        # ── RTDE Receive ──────────────────────────────────────────────
        print("\n连接 RTDE Receive...")
        rtde_r = reconnect_rtde()

        # ── XML-RPC 控制器（移动 + 夹爪）─────────────────────────────
        print("\n启动 XML-RPC 控制器...")
        controller = RobotXMLRPCController(XMLRPC_SERVER_IP, XMLRPC_SERVER_PORT, ROBOT_IP)
        if not controller.start():
            raise RuntimeError("XML-RPC 控制器启动失败")

        gripper_state = 0.0
        controller.gripper_close()

        # ── 操作说明 ──────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("UR7e 键盘控制采集（XML-RPC 模式）")
        print("=" * 60)
        print("  W/S=前后  A/D=左右  R/F=上下")
        print("  O/K=yaw  P/L=pitch  M/N=roll")
        print("  G=夹爪切换  Q=结束录制")
        print("  确保示教器 xvla control program 正在运行！")
        print("=" * 60)

        episode = get_next_episode_id(SAVE_DIR)
        interval = 1.0 / FPS

        # ── 采集循环 ──────────────────────────────────────────────────
        while True:
            sequence = generate_episode_sequence()
            total = len(sequence)
            print("\n" + "=" * 60)
            print("本轮序列（12 条）：")
            for i, (instr, _) in enumerate(sequence, 1):
                print(f"  {i:2d}. {instr}")
            print("=" * 60)
            cont = input("开始？(y/n): ")
            if cont.lower() != 'y':
                continue

            for step_idx, (task_instruction, op_type) in enumerate(sequence):
                step_num = step_idx + 1
                print(f"\n{'─'*60}")
                print(f"  第 {step_num}/{total} 条 | Ep#{episode} | {task_instruction}")
                print(f"{'─'*60}")
                input("移到起始位置后按 Enter 开始录制...")

                # 同步当前位姿到 XML-RPC target
                rtde_r = reconnect_rtde(rtde_r)
                controller.update_pose_from_robot()

                print(">>> 录制中！WASD/RF=移动 OKPLMN=旋转 G=夹爪 Q=结束 <<<")
                _setup_terminal()

                images_main, images_wrist, depths_d405 = [], [], []
                tcp_poses, joint_positions, gripper_states = [], [], []
                frame_count = 0
                start_tcp = None

                while frame_count < MAX_FRAMES:
                    loop_start = time.time()

                    # 读取状态
                    tcp = rtde_r.getActualTCPPose()
                    joints = rtde_r.getActualQ()
                    if tcp is None or joints is None:
                        time.sleep(0.01); continue

                    if start_tcp is None:
                        start_tcp = np.array(tcp[:3])
                    move_dist = np.linalg.norm(np.array(tcp[:3]) - start_tcp)

                    # 读取图像
                    frame_m, frame_w, depth = capture_frames(active_pipelines)
                    if frame_m is None:
                        time.sleep(0.01); continue

                    # 保存数据
                    images_main.append(cv2.resize(cv2.cvtColor(frame_m, cv2.COLOR_BGR2RGB), (SAVE_WIDTH, SAVE_HEIGHT)))
                    images_wrist.append(cv2.resize(cv2.cvtColor(frame_w, cv2.COLOR_BGR2RGB), (SAVE_WIDTH, SAVE_HEIGHT)))
                    depths_d405.append(cv2.resize(depth, (SAVE_WIDTH, SAVE_HEIGHT), interpolation=cv2.INTER_NEAREST).astype(np.uint16))
                    tcp_poses.append(tcp)
                    joint_positions.append(joints)
                    gripper_states.append(gripper_state)

                    if frame_count % 30 == 0:
                        print(f"  F:{frame_count} TCP:[{tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f}] "
                              f"d:{move_dist:.3f}m grip:{'OPEN' if gripper_state>0.5 else 'CLOSED'}")
                    frame_count += 1

                    # 视频预览
                    disp_m = frame_m.copy()
                    grip_str = "OPEN" if gripper_state > 0.5 else "CLOSED"
                    grip_color = (0,255,0) if gripper_state > 0.5 else (0,0,255)
                    cv2.putText(disp_m, f"Ep#{episode} [{step_num}/{total}] F:{frame_count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.putText(disp_m, f"TCP:[{tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f}]", (10,58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
                    cv2.putText(disp_m, f"Gripper: {grip_str}", (10,85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, grip_color, 2)
                    cv2.putText(disp_m, f"Task: {task_instruction}", (10,115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,255), 1)
                    cv2.putText(disp_m, "WASD/RF OKPLMN G=grip Q=stop", (10,460), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
                    disp_w = frame_w.copy()
                    cv2.putText(disp_w, "Wrist", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                    cv2.imshow("UR7e Collection", np.hstack([disp_m, disp_w]))
                    cv2.waitKey(1)

                    # 键盘输入（drain all available keys）
                    keys = set()
                    stop_req = False
                    while True:
                        k = check_terminal_key()
                        if k is None:
                            break
                        if k == 'q':
                            stop_req = True
                            break
                        keys.add(k)
                    if stop_req: break

                    # 夹爪切换
                    if 'g' in keys:
                        gripper_state = 1.0 - gripper_state
                        gripper_states[-1] = gripper_state
                        if gripper_state > 0.5:
                            controller.gripper_open()
                        else:
                            controller.gripper_close()

                    # 位姿增量
                    dx = dy = dz = drx = dry = drz = 0.0
                    if 'd' in keys: dx += TRANS_STEP
                    if 'a' in keys: dx -= TRANS_STEP
                    if 'w' in keys: dy += TRANS_STEP
                    if 's' in keys: dy -= TRANS_STEP
                    if 'r' in keys: dz += TRANS_STEP
                    if 'f' in keys: dz -= TRANS_STEP
                    if 'o' in keys: drz += ROT_STEP
                    if 'k' in keys: drz -= ROT_STEP
                    if 'p' in keys: dry += ROT_STEP
                    if 'l' in keys: dry -= ROT_STEP
                    if 'm' in keys: drx += ROT_STEP
                    if 'n' in keys: drx -= ROT_STEP

                    if dx or dy or dz or drx or dry or drz:
                        # 坐标映射
                        trans = _MOUSE2ROBOT @ np.array([dx, dy, dz])
                        rot = _MOUSE2ROBOT @ np.array([drx, dry, drz])
                        controller.add_delta(*trans, *rot)

                    elapsed = time.time() - loop_start
                    if elapsed < interval:
                        time.sleep(interval - elapsed)

                _restore_terminal()

                # 保存
                if len(images_main) < 30:
                    print(f"  太短（{len(images_main)}帧），丢弃")
                    continue

                save_path = SAVE_DIR / f"episode_{episode:04d}.npz"
                np.savez_compressed(
                    save_path,
                    images=np.array(images_main, dtype=np.uint8),
                    images_wrist=np.array(images_wrist, dtype=np.uint8),
                    depths_d405=np.array(depths_d405, dtype=np.uint16),
                    d405_depth_scale=np.float64(pipe_d405["depth_scale"]),
                    tcp_poses=np.array(tcp_poses, dtype=np.float64),
                    joint_positions=np.array(joint_positions, dtype=np.float64),
                    gripper=np.array(gripper_states, dtype=np.float64),
                    instruction=task_instruction,
                    fps=FPS,
                )
                print(f"  ✓ 保存: {save_path} ({len(images_main)}帧) | {task_instruction}")
                episode += 1
                saved_count += 1

                if step_num < total:
                    remaining = sequence[step_idx+1:]
                    print(f"\n  剩余 {len(remaining)} 条")
                    input("  Enter 继续...")

            print(f"\n本轮完成，累计 {saved_count} 条")
            if input("继续下一轮？(y/n): ").lower() != 'y':
                break

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        _restore_terminal()
        if controller: controller.stop()
        if rtde_r:
            try: rtde_r.disconnect()
            except: pass
        for p in pipelines:
            try: p["pipe"].stop()
            except: pass
        cv2.destroyAllWindows()
        print(f"\n采集结束，保存 {saved_count} 条到 {SAVE_DIR}")


if __name__ == "__main__":
    main()
