"""
UR7e 左臂 3D Mouse 主动控制示教数据采集
双 RealSense RGB + D405 Depth

与 dataset.py 的区别：
  - 不再使用示教器 Freedrive 手拖，改为 3D Mouse 主动控制机械臂
  - 使用 RTDEControlInterface 发送末端速度指令（speedl）
  - 3D Mouse 左键 / 右键控制夹爪开合（优遨夹爪 Modbus TCP）
  - 其余采集逻辑（序列生成、保存格式、预览）与 dataset.py 完全一致

依赖安装：
    pip install pyspacemouse easyhid   # 3D Mouse HID 读取
    pip install ur-rtde                # RTDEControlInterface
    pip install pymodbus               # 夹爪 Modbus TCP 控制

3D Mouse 操作说明：
    平移轴  X/Y/Z  → TCP 末端 x/y/z 方向移动
    旋转轴 Rx/Ry/Rz → TCP 末端 roll/pitch/yaw 旋转
    左键（Button 0）→ 夹爪闭合
    右键（Button 1）→ 夹爪打开
    按 'q' 或 ESC（OpenCV 窗口）→ 结束当前条录制

夹爪说明（优遨 X760-0034-D2 E系列）：
    通过 UR Tool RS485 接口连接，UR 控制器将其代理为 Modbus TCP。
    Modbus TCP 地址：机器人 IP:502，从站 ID=9（Tool 端口默认）
    寄存器 0x0100（256）写入位置指令：0=全开，1000=全闭
    如寄存器地址与实际不符，请参考夹爪手册修改 GRIPPER_* 常量。
"""

import cv2
import time
import threading
import random
import numpy as np
import pyrealsense2 as rs
from pathlib import Path

# ── 机械臂控制（需要 ur-rtde）──────────────────────────────────────────
from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface

# ── Dashboard 接口（用于检查远程控制模式）─────────────────────────────
import socket

# ── 3D Mouse（需要 pyspacemouse + easyhid）────────────────────────────
try:
    import pyspacemouse
    _SPACEMOUSE_AVAILABLE = True
except ImportError:
    _SPACEMOUSE_AVAILABLE = False
    print("警告: pyspacemouse 未安装，请运行 pip install pyspacemouse easyhid")

# ── 夹爪 Modbus TCP（需要 pymodbus）──────────────────────────────────
try:
    from pymodbus.client import ModbusTcpClient as ModbusClient
    _MODBUS_AVAILABLE = True
except ImportError:
    try:
        from pymodbus.client.sync import ModbusTcpClient as ModbusClient
        _MODBUS_AVAILABLE = True
    except ImportError:
        _MODBUS_AVAILABLE = False
        print("警告: pymodbus 未安装，请运行 pip install pymodbus")


SCRIPT_VERSION = "3DMOUSE_CONTROL_V1"
print(f"当前运行脚本版本: {SCRIPT_VERSION}")

# ============ 配置（根据你的环境修改） ============
ARM_NAME = "left"
ROBOT_IP = "192.168.1.88"       # UR 控制柜 IP
SAVE_DIR = Path(f"./raw_demos_{ARM_NAME}_3dmouse")

FPS = 30
MAX_FRAMES = 900                 # 30fps × 30秒

COLOR_WIDTH  = 640
COLOR_HEIGHT = 480
DEPTH_WIDTH  = 640
DEPTH_HEIGHT = 480
SAVE_WIDTH   = 256
SAVE_HEIGHT  = 256

# ── 3D Mouse 速度缩放 ──────────────────────────────────────────────────
# 3D Mouse 输出范围约 [-1, 1]，乘以缩放系数得到 m/s 或 rad/s
TRANS_SCALE = 0.05   # 平移速度上限 m/s（每轴），可按手感调大/调小
ROT_SCALE   = 0.2   # 旋转速度上限 rad/s（每轴）
SPEEDL_ACC  = 1.0    # speedL 加速度 m/s²（调大让响应更快）
SPEEDL_TIME = 0.02    # speedL time：每条指令的有效时间（秒）
                     # 设为 0.1s 而非 0.0，给控制线程留出 GIL 竞争的余量
                     # 如果控制线程在 100ms 内未能发送下一条指令，机器人会减速停止
                     # 而不是立即触发保护停止

# ── 夹爪 Modbus 配置（优遨 E 系列）────────────────────────────────────
# UR 控制器将 Tool RS485 代理到 Modbus TCP 502 端口
GRIPPER_MODBUS_PORT   = 502
GRIPPER_SLAVE_ID      = 9       # Tool 端口默认从站 ID，如不对请查夹爪手册
GRIPPER_REG_POSITION  = 0x0100  # 位置指令寄存器（256）
GRIPPER_POS_OPEN      = 0       # 全开
GRIPPER_POS_CLOSE     = 1000    # 全闭（最大行程，可按需调整）
GRIPPER_REG_SPEED     = 0x0101  # 速度寄存器（可选，部分型号支持）
GRIPPER_SPEED_DEFAULT = 500     # 速度（0-1000）
# ===============================================

# 3D Mouse -> robot coordinate mapping (same math as demo_real_robot.py)
MAP_ROTATION_DEG = 100.0
MAP_TILT_DEG     = -36.0
MAP_XY_ROT_DEG   = 36.0

def get_teleop_rotation(theta_deg, tilt_deg, xy_rot_deg):
    """Compute 3x3 rotation matrix from 3D Mouse frame to robot frame."""
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
    # R @ [mouse_x, mouse_y, mouse_z] -> robot frame
    # mouse x -> v_final, mouse y -> u_final, mouse z -> w_final
    return np.column_stack([v_final, u_final, w_final])

_MOUSE2ROBOT = get_teleop_rotation(MAP_ROTATION_DEG, MAP_TILT_DEG, MAP_XY_ROT_DEG)

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 蔬菜种类（与 dataset.py 保持一致）
VEGETABLES = [
    "red pepper",
    "green pepper",
    "yellow pepper",
    "corn",
    "purple sweet potato",
    "pumpkin",
]


# ============ 夹爪控制 ============

class GripperController:
    """
    优遨 E 系列夹爪控制器。
    通过 UR 控制器的 Modbus TCP 代理（端口 502）发送位置指令。
    兼容 pymodbus 不同版本（自动探测 slave/unit 参数）。
    """

    def __init__(self, robot_ip: str):
        self._ip = robot_ip
        self._client: "ModbusClient | None" = None
        self._connected = False
        self._slave_key = "slave"  # 默认尝试 slave，失败后切换为 unit

    def _write_reg(self, register: int, value: int):
        """写入单个寄存器，自动处理 slave/unit 参数兼容性。"""
        kwargs = {self._slave_key: GRIPPER_SLAVE_ID}
        try:
            self._client.write_register(register, value, **kwargs)
        except TypeError:
            # 参数名不对，切换到另一个
            self._slave_key = "unit" if self._slave_key == "slave" else "slave"
            kwargs = {self._slave_key: GRIPPER_SLAVE_ID}
            self._client.write_register(register, value, **kwargs)

    def connect(self) -> bool:
        if not _MODBUS_AVAILABLE:
            print("  [夹爪] pymodbus 未安装，夹爪控制不可用")
            return False
        try:
            self._client = ModbusClient(host=self._ip, port=GRIPPER_MODBUS_PORT)
            self._client.connect()
            self._connected = True
            print(f"  [夹爪] Modbus TCP 已连接 {self._ip}:{GRIPPER_MODBUS_PORT}")
            # 初始化：设置速度（同时探测正确的参数名）
            self._write_reg(GRIPPER_REG_SPEED, GRIPPER_SPEED_DEFAULT)
            return True
        except Exception as e:
            print(f"  [夹爪] 连接失败: {e}")
            self._connected = False
            return False

    def open(self):
        """打开夹爪"""
        self._write_position(GRIPPER_POS_OPEN)

    def close(self):
        """闭合夹爪"""
        self._write_position(GRIPPER_POS_CLOSE)

    def _write_position(self, pos: int):
        if not self._connected or self._client is None:
            return
        try:
            self._write_reg(GRIPPER_REG_POSITION, pos)
        except Exception as e:
            print(f"  [夹爪] 写入失败: {e}")

    def disconnect(self):
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._connected = False


# ============ 3D Mouse 读取（新版 pyspacemouse API）============

class SpaceMouseReader:
    """
    封装新版 pyspacemouse API（v1.x）。
    open() 返回 SpaceMouseDevice 对象，read()/close() 是其方法。
    主循环调用 poll() 刷新状态，避免多线程竞争。
    """

    def __init__(self):
        self.state = None      # 最新的 SpaceMouseState
        self._device = None    # SpaceMouseDevice 实例
        self._running = False

    def start(self) -> bool:
        if not _SPACEMOUSE_AVAILABLE:
            print("  [3DMouse] pyspacemouse 未安装")
            return False
        try:
            self._device = pyspacemouse.open()
            if self._device is None:
                print("  [3DMouse] 未找到设备，请确认 3D Mouse 已连接")
                return False
            self._running = True
            print(f"  [3DMouse] 已连接: {self._device.product_name}")
            return True
        except Exception as e:
            print(f"  [3DMouse] 初始化失败: {e}")
            return False

    def poll(self):
        """
        在主循环里调用，从设备读取最新状态。
        如果没有新输入（t=-1.0），将 state 清零，确保松手后速度归零。
        """
        if not self._running or self._device is None:
            return False
        try:
            s = self._device.read()
            if s and s.t != -1.0:   # 有新数据
                self.state = s
                return True
            else:
                # 无新输入 → 用户已松手，清零速度
                self.state = None
        except Exception:
            pass
        return False

    def stop(self):
        self._running = False
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None

    def get_velocity(self):
        """返回 [vx, vy, vz, v_roll, v_pitch, v_yaw]，单位 m/s 和 rad/s。"""
        s = self.state
        if s is None:
            return [0.0] * 6
        mouse_trans = np.array([float(s.x), float(s.y), float(s.z)])
        mouse_rot   = np.array([float(s.roll), float(s.pitch), float(s.yaw)])
        robot_trans = _MOUSE2ROBOT @ mouse_trans * TRANS_SCALE
        robot_rot   = _MOUSE2ROBOT @ mouse_rot * ROT_SCALE
        return [
            robot_trans[0], robot_trans[1], robot_trans[2],
            robot_rot[0],   robot_rot[1],   robot_rot[2],
        ]

    def get_buttons(self):
        """返回 (btn0_pressed, btn1_pressed)"""
        s = self.state
        if s is None or not hasattr(s, "buttons"):
            return False, False
        btns = s.buttons
        b0 = bool(btns[0]) if len(btns) > 0 else False
        b1 = bool(btns[1]) if len(btns) > 1 else False
        return b0, b1


# ============ 机械臂速度控制线程 ============

class ArmControlThread:
    """
    在独立线程中以固定频率（默认 50 Hz）向机械臂发送 speedL 指令。
    主循环只需更新 .velocity，控制线程负责持续发送，避免主线程卡顿
    导致 RTDE 心跳超时触发保护停止。
    """

    CONTROL_HZ = 50           # 控制频率（Hz）
    CONTROL_DT = 1.0 / 50     # 20 ms 周期
    # 注意：SPEEDL_TIME=0.1s 意味着每条指令有效 100ms，
    # 所以 50Hz（20ms 周期）有足够的安全余量。
    # 即使偶尔因 GIL 延迟到 50-60ms，机器人仍在执行上一条指令。

    def __init__(self, robot_ip: str):
        self._ip = robot_ip
        self.velocity = [0.0] * 6   # 主循环写入，控制线程读取
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._rtde_c: RTDEControlInterface | None = None

    def start(self) -> bool:
        try:
            self._rtde_c = RTDEControlInterface(self._ip)
            print(f"  [ArmCtrl] 控制接口已连接 {self._ip}")
        except Exception as e:
            print(f"  [ArmCtrl] 连接失败: {e}")
            return False

        # 检查机器人是否处于可运行状态，如有保护停止则自动恢复
        if not self._ensure_robot_ready():
            return False

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ArmCtrl")
        self._thread.start()
        print("  [ArmCtrl] 控制线程已启动（50 Hz）")
        return True

    def _ensure_robot_ready(self) -> bool:
        """
        验证控制脚本是否正常运行。
        尝试发送零速指令，如果失败则尝试 unlockProtectiveStop 后重试。
        """
        try:
            # 先尝试直接发送零速指令
            try:
                self._rtde_c.speedL([0.0] * 6, SPEEDL_ACC, SPEEDL_TIME)
                print("  [ArmCtrl] 机器人状态正常，控制脚本已就绪")
                return True
            except Exception:
                pass

            # 可能有保护停止，尝试解除
            print("  [ArmCtrl] 控制脚本未就绪，尝试解除保护停止...")
            try:
                self._rtde_c.unlockProtectiveStop()
                time.sleep(1.5)  # 等待控制器恢复
            except Exception:
                pass

            # 再次尝试发送零速指令
            try:
                self._rtde_c.speedL([0.0] * 6, SPEEDL_ACC, SPEEDL_TIME)
                print("  [ArmCtrl] 保护停止已解除，控制脚本已就绪")
                return True
            except Exception:
                pass

            # 如果仍然不行，断开重连
            print("  [ArmCtrl] 控制脚本仍未就绪，尝试重新建立连接...")
            try:
                self._rtde_c.disconnect()
            except Exception:
                pass
            time.sleep(1.0)
            try:
                self._rtde_c = RTDEControlInterface(self._ip)
                time.sleep(0.5)
                self._rtde_c.speedL([0.0] * 6, SPEEDL_ACC, SPEEDL_TIME)
                print("  [ArmCtrl] 重连后控制脚本已就绪")
                return True
            except Exception as e:
                print(f"  [ArmCtrl] 重连后仍无法启动控制脚本: {e}")
                return False

        except Exception as e:
            print(f"  [ArmCtrl] 状态检查异常: {e}")
            return False

    def _loop(self):
        while self._running:
            t0 = time.perf_counter()
            with self._lock:
                vel = list(self.velocity)
            try:
                self._rtde_c.speedL(vel, SPEEDL_ACC, SPEEDL_TIME)
            except Exception as e:
                print(f"  [ArmCtrl] speedL 失败: {e}，尝试重连...")
                self._reconnect()
            elapsed = time.perf_counter() - t0
            sleep_t = self.CONTROL_DT - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _reconnect(self):
        try:
            self._rtde_c.disconnect()
        except Exception:
            pass
        time.sleep(1.0)  # 给控制器更多恢复时间
        try:
            self._rtde_c = RTDEControlInterface(self._ip)
            # 尝试解除保护停止（如果有的话）
            try:
                self._rtde_c.unlockProtectiveStop()
                time.sleep(0.5)
            except Exception:
                pass
            print("  [ArmCtrl] 重连成功")
        except Exception as e:
            print(f"  [ArmCtrl] 重连失败: {e}")
            time.sleep(1.0)  # 避免疯狂重试

    def set_velocity(self, vel: list):
        """主循环调用，线程安全地更新速度指令。"""
        with self._lock:
            self.velocity = list(vel)

    def stop(self):
        self._running = False
        # 先发零速停止机械臂
        if self._rtde_c is not None:
            try:
                self._rtde_c.speedL([0.0] * 6, SPEEDL_ACC, 0.0)
                self._rtde_c.speedStop()
                self._rtde_c.disconnect()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        print("  [ArmCtrl] 控制线程已停止")




def get_next_episode_id(save_dir):
    existing = sorted(save_dir.glob("episode_*.npz"))
    if not existing:
        return 0
    max_id = -1
    for f in existing:
        try:
            idx = int(f.stem.split("_")[-1])
            max_id = max(max_id, idx)
        except ValueError:
            pass
    return max_id + 1


def reconnect_rtde_receive(old_rtde=None):
    if old_rtde is not None:
        try:
            if hasattr(old_rtde, "disconnect"):
                old_rtde.disconnect()
        except Exception:
            pass
    rtde_r = RTDEReceiveInterface(ROBOT_IP)
    tcp = rtde_r.getActualTCPPose()
    q   = rtde_r.getActualQ()
    print(
        f"  [RTDE-R] 已连接，TCP: "
        f"[{tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f}], "
        f"q0={q[0]:.4f}"
    )
    return rtde_r


def check_robot_mode():
    """
    通过 UR Dashboard Server（端口 29999）检查机器人是否处于远程控制模式。
    返回 True 表示远程模式，False 表示本地模式或无法确认。
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3.0)
        s.connect((ROBOT_IP, 29999))
        # Dashboard 连接后会发送欢迎消息
        s.recv(1024)
        # 查询是否为远程控制模式
        s.sendall(b"is in remote control\n")
        time.sleep(0.2)
        response = s.recv(1024).decode("utf-8", errors="ignore").strip()
        s.close()
        print(f"  [Dashboard] 远程控制模式查询: {response}")
        if "true" in response.lower():
            return True
        elif "false" in response.lower():
            return False
        else:
            # 旧版 UR 可能不支持此命令，假设可用
            print("  [Dashboard] 无法确认远程模式（可能是旧版固件），继续尝试...")
            return True
    except Exception as e:
        print(f"  [Dashboard] 查询失败: {e}，继续尝试...")
        return True  # 无法确认时不阻塞


def connect_rtde_control():
    rtde_c = RTDEControlInterface(ROBOT_IP)
    print(f"  [RTDE-C] 控制接口已连接 {ROBOT_IP}")
    return rtde_c


def start_realsense_pipelines():
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) < 2:
        raise RuntimeError(f"需要 2 个 RealSense 摄像头，只找到 {len(devices)} 个")
    print(f"找到 {len(devices)} 个 RealSense 设备:")
    pipelines = []
    for dev in devices:
        name   = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"  - {name} (S/N: {serial})")
        pipeline = rs.pipeline()
        config   = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
        enable_depth = "D405" in name.upper()
        if enable_depth:
            config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)
        try:
            profile = pipeline.start(config)
            depth_scale = None
            align = None
            if enable_depth:
                depth_sensor = profile.get_device().first_depth_sensor()
                depth_scale  = depth_sensor.get_depth_scale()
                align        = rs.align(rs.stream.color)
            pipelines.append({
                "name": name, "serial": serial, "pipe": pipeline,
                "enable_depth": enable_depth, "depth_scale": depth_scale, "align": align,
            })
            print(f"  ▶ {name} 启动成功: {'RGB + Depth' if enable_depth else 'RGB only'}")
        except Exception as e:
            for p in pipelines:
                try: p["pipe"].stop()
                except Exception: pass
            raise RuntimeError(f"{name} 启动失败: {e}")
    print("正在预热 RealSense 摄像头...")
    for _ in range(15):
        for p in pipelines:
            try: p["pipe"].wait_for_frames(timeout_ms=1000)
            except RuntimeError: pass
    return pipelines


def get_rgb_and_depth(p):
    frames = p["pipe"].wait_for_frames(timeout_ms=2000)
    if p.get("enable_depth", False):
        aligned = p["align"].process(frames)
        cf = aligned.get_color_frame()
        df = aligned.get_depth_frame()
        if not cf or not df:
            return None, None
        return np.asanyarray(cf.get_data()), np.asanyarray(df.get_data())
    else:
        cf = frames.get_color_frame()
        if not cf:
            return None, None
        return np.asanyarray(cf.get_data()), None


def get_rgb_frame(p):
    img, _ = get_rgb_and_depth(p)
    return img


def capture_frames(pipelines):
    frames = []
    depth_d405 = None
    for p in pipelines:
        try:
            img, depth = get_rgb_and_depth(p)
            frames.append(img)
            if p.get("enable_depth", False):
                depth_d405 = depth
        except RuntimeError:
            frames.append(None)
            if p.get("enable_depth", False):
                depth_d405 = None
    if any(f is None for f in frames) or depth_d405 is None:
        return None, None, None
    return frames[0], frames[1], depth_d405


def generate_episode_sequence():
    """随机排列蔬菜，为每种蔬菜随机分配左/右篮子，生成 12 条原子指令。"""
    shuffled = VEGETABLES[:]
    random.shuffle(shuffled)
    sequence = []
    for veg in shuffled:
        side = random.choice(["left", "right"])
        sequence.append((f"pick up the {veg}", "pick_up"))
        sequence.append((f"place the {veg} in the {side} basket", "place"))
    return sequence


# ============ 主函数 ============

def main():
    pipelines = []
    rtde_r    = None
    arm_ctrl  = ArmControlThread(ROBOT_IP)
    gripper   = GripperController(ROBOT_IP)
    mouse     = SpaceMouseReader()
    saved_count = 0

    try:
        # ── 初始化 RealSense ──────────────────────────────────────────
        print("\n正在启动 RealSense 摄像头...")
        pipelines = start_realsense_pipelines()
        pipe_main  = pipelines[0]
        pipe_wrist = pipelines[1]

        # 交互式确认视角
        print("\n正在预览摄像头画面，按 's' 交换视角，按 Enter 确认...")
        while True:
            try:
                img1 = get_rgb_frame(pipe_main)
                img2 = get_rgb_frame(pipe_wrist)
                if img1 is not None and img2 is not None:
                    vis1 = img1.copy(); vis2 = img2.copy()
                    cv2.putText(vis1, f"Main: {pipe_main['name']}",  (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.putText(vis2, f"Wrist: {pipe_wrist['name']}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.imshow("Camera Preview - Enter=confirm, S=swap", np.hstack([vis1, vis2]))
            except RuntimeError:
                pass
            key = cv2.waitKey(100) & 0xFF
            if key in [13, 10]:
                break
            elif key == ord("s"):
                pipe_main, pipe_wrist = pipe_wrist, pipe_main
                print(f"已交换 -> 主视角: {pipe_main['name']}，腕部: {pipe_wrist['name']}")
        cv2.destroyAllWindows()

        active_pipelines = [pipe_main, pipe_wrist]
        pipe_d405 = next((p for p in active_pipelines if "D405" in p["name"].upper()), None)
        if pipe_d405 is None or not pipe_d405.get("enable_depth", False):
            raise RuntimeError("未找到 D405 或 D405 未启用 depth")
        print(f"\nD405: {pipe_d405['name']}, depth_scale={pipe_d405['depth_scale']}")

        # ── 初始化机械臂（仅 Receive 接口，Control 线程最后启动）────────
        print("\n正在连接机械臂...")
        rtde_r = reconnect_rtde_receive()

        # 检查远程控制模式
        if not check_robot_mode():
            print("\n  ⚠ 示教器当前处于「本地控制」模式！")
            print("  请在示教器右上角点击切换到「远程控制」（Remote Control）模式")
            input("  切换完成后按 Enter 继续...")
            if not check_robot_mode():
                raise RuntimeError("示教器仍未切换到远程控制模式，无法继续")

        # ── 初始化夹爪（必须在控制线程启动之前完成）─────────────────
        print("\n正在连接夹爪...")
        gripper_ok = gripper.connect()
        if gripper_ok:
            gripper.open()   # 初始状态：打开
            print("  [夹爪] 初始状态：打开")
        else:
            print("  [夹爪] 夹爪不可用，将仅记录状态（不实际控制）")

        # ── 初始化 3D Mouse（必须在控制线程启动之前完成）──────────────
        print("\n正在连接 3D Mouse...")
        mouse_ok = mouse.start()
        if not mouse_ok:
            raise RuntimeError("3D Mouse 初始化失败，请检查连接")

        # ── 启动机械臂控制线程（所有其他连接完成后最后启动）──────────
        # 重要：控制线程启动后，不能再做任何可能阻塞或干扰 RTDE 的操作
        print("\n正在启动机械臂控制线程...")
        arm_ctrl = ArmControlThread(ROBOT_IP)
        if not arm_ctrl.start():
            print("\n  ⚠ 机械臂控制线程启动失败！可能原因：")
            print("    1. 示教器未切换到「远程控制」模式（Remote Control）")
            print("    2. 存在未解除的保护性停止（Protective Stop）")
            print("    3. 机器人处于紧急停止状态")
            print("\n  请在示教器上：")
            print("    - 确认已切换到「远程控制」模式")
            print("    - 如有保护停止，点击「解除」按钮")
            print("    - 确认机器人电源已开启且无报警")
            input("\n  处理完毕后按 Enter 重试...")
            # 重试
            arm_ctrl = ArmControlThread(ROBOT_IP)
            if not arm_ctrl.start():
                raise RuntimeError(
                    "机械臂控制线程启动失败。请确认：\n"
                    "  1. 示教器处于远程控制模式\n"
                    "  2. 无保护停止/紧急停止\n"
                    "  3. 机器人程序未在运行"
                )

        # ── 操作说明 ──────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("UR7e 左臂 3D Mouse 控制采集（双 RealSense RGB + D405 Depth）")
        print("=" * 60)
        print("3D Mouse 操作：")
        print("  平移轴 X/Y/Z  → TCP 末端平移")
        print("  旋转轴 Rx/Ry/Rz → TCP 末端旋转")
        print("  左键（Button 0）→ 夹爪闭合")
        print("  右键（Button 1）→ 夹爪打开")
        print("  OpenCV 窗口按 'q' 或 ESC → 结束当前条录制")
        print("=" * 60)

        episode  = get_next_episode_id(SAVE_DIR)
        interval = 1.0 / FPS

        # ── 外层循环：每轮生成一个采集序列 ───────────────────────────
        # 在等待用户输入前，先停止控制线程，避免长时间阻塞导致 RTDE 超时
        arm_ctrl.stop()
        print("  [ArmCtrl] 等待用户确认期间已暂停控制线程")

        while True:
            sequence       = generate_episode_sequence()
            total_in_round = len(sequence)

            print("\n" + "=" * 60)
            print("本轮采集序列（共 12 条原子动作）：")
            for idx, (instr, _) in enumerate(sequence, 1):
                print(f"  {idx:2d}. {instr}")
            print("=" * 60)
            cont = input("开始本轮采集？(y/n，n 则重新生成): ")
            if cont.lower() != "y":
                continue

            # ── 内层循环：逐条录制 ────────────────────────────────────
            for step_idx, (task_instruction, op_type) in enumerate(sequence):
                step_num = step_idx + 1

                print(f"\n{'─' * 60}")
                print(f"  本轮第 {step_num}/{total_in_round} 条  |  Episode #{episode}")
                print(f"  指令: {task_instruction}")
                print(f"  操作: {'拾取' if op_type == 'pick_up' else '放置'}")
                print(f"{'─' * 60}")
                input("将机器人移到起始位置后，按 Enter 开始录制...")

                # 重新连接 RTDE Receive（此时控制线程未运行，不会冲突）
                rtde_r = reconnect_rtde_receive(rtde_r)

                # 启动控制线程（每条录制前重新启动，确保 RTDE 控制脚本新鲜）
                arm_ctrl = ArmControlThread(ROBOT_IP)
                if not arm_ctrl.start():
                    print("  [ArmCtrl] 控制线程启动失败，尝试清除保护停止后重试...")
                    # 等待用户在示教器上解除保护停止
                    input("  请在示教器上解除保护停止，然后按 Enter 重试...")
                    arm_ctrl = ArmControlThread(ROBOT_IP)
                    if not arm_ctrl.start():
                        raise RuntimeError("机械臂控制线程启动失败，请检查示教器状态")

                print(">>> 录制已开始！移动 3D Mouse 控制机械臂，按 q/ESC 结束 <<<")

                images_main    = []
                images_wrist   = []
                depths_d405    = []
                tcp_poses      = []
                joint_positions = []
                gripper_states = []

                gripper_state = 1.0   # 初始：打开（1.0=打开，0.0=闭合）
                frame_count   = 0
                start_tcp     = None

                # 按键防抖：避免单次按键触发多次
                btn0_prev = False
                btn1_prev = False

                while frame_count < MAX_FRAMES:
                    loop_start = time.time()

                    # ── 读取 3D Mouse 并更新速度指令 ─────────────────
                    mouse.poll()   # 刷新最新状态
                    vel = mouse.get_velocity()
                    btn0, btn1 = mouse.get_buttons()

                    # 把速度写入控制线程（线程安全），控制线程以 50 Hz 持续发送
                    arm_ctrl.set_velocity(vel)

                    # ── 夹爪按键处理（上升沿触发）────────────────────
                    if btn0 and not btn0_prev:
                        gripper_state = 0.0   # 闭合
                        if gripper_ok:
                            gripper.close()
                        print("  [夹爪] → 闭合")
                    if btn1 and not btn1_prev:
                        gripper_state = 1.0   # 打开
                        if gripper_ok:
                            gripper.open()
                        print("  [夹爪] → 打开")
                    btn0_prev = btn0
                    btn1_prev = btn1

                    # ── 读取机器人状态 ────────────────────────────────
                    tcp    = rtde_r.getActualTCPPose()
                    joints = rtde_r.getActualQ()
                    if tcp is None or joints is None:
                        print("警告: RTDE 读取为空，跳过该帧")
                        time.sleep(0.01)
                        continue

                    if start_tcp is None:
                        start_tcp = np.array(tcp[:3], dtype=np.float64)
                    delta_tcp  = np.array(tcp[:3], dtype=np.float64) - start_tcp
                    move_dist  = np.linalg.norm(delta_tcp)

                    # ── 读取图像 ──────────────────────────────────────
                    frame_m, frame_w, depth_d405 = capture_frames(active_pipelines)
                    if frame_m is None or frame_w is None or depth_d405 is None:
                        time.sleep(0.01)
                        continue

                    frame_m_rgb = cv2.cvtColor(frame_m, cv2.COLOR_BGR2RGB)
                    frame_w_rgb = cv2.cvtColor(frame_w, cv2.COLOR_BGR2RGB)
                    images_main.append(cv2.resize(frame_m_rgb, (SAVE_WIDTH, SAVE_HEIGHT)))
                    images_wrist.append(cv2.resize(frame_w_rgb, (SAVE_WIDTH, SAVE_HEIGHT)))
                    depth_resized = cv2.resize(depth_d405, (SAVE_WIDTH, SAVE_HEIGHT),
                                               interpolation=cv2.INTER_NEAREST)
                    depths_d405.append(depth_resized.astype(np.uint16))
                    tcp_poses.append(tcp)
                    joint_positions.append(joints)
                    gripper_states.append(gripper_state)

                    if frame_count % 30 == 0:
                        print(
                            f"  REC F:{frame_count} | "
                            f"TCP:[{tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f}] | "
                            f"dMove:{move_dist:.3f}m | "
                            f"vel:[{vel[0]:.3f},{vel[1]:.3f},{vel[2]:.3f},"
                            f"{vel[3]:.3f},{vel[4]:.3f},{vel[5]:.3f}] | "
                            f"grip:{'OPEN' if gripper_state>0.5 else 'CLOSED'}"
                        )
                    frame_count += 1

                    # ── 预览画面 ──────────────────────────────────────
                    display_m = frame_m.copy()
                    display_w = frame_w.copy()
                    grip_str   = "OPEN"   if gripper_state > 0.5 else "CLOSED"
                    grip_color = (0,255,0) if gripper_state > 0.5 else (0,0,255)

                    cv2.putText(display_m, f"Ep#{episode} [{step_num}/{total_in_round}] F:{frame_count}",
                                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.putText(display_m, f"TCP:[{tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f}]",
                                (10,58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
                    cv2.putText(display_m, f"vel:[{vel[0]:.2f},{vel[1]:.2f},{vel[2]:.2f}]",
                                (10,78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
                    cv2.putText(display_m, f"Move:{move_dist:.3f}m",
                                (10,98), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
                    cv2.putText(display_m, f"Gripper: {grip_str}",
                                (10,125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, grip_color, 2)
                    instr_disp = task_instruction if len(task_instruction) <= 38 else task_instruction[:35]+"..."
                    cv2.putText(display_m, instr_disp,
                                (10,155), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,255), 1)
                    cv2.putText(display_m, "Btn0=CLOSE  Btn1=OPEN  Q=stop",
                                (10,460), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
                    cv2.putText(display_w, "Wrist", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                    cv2.imshow("UR7e 3DMouse Collection", np.hstack([display_m, display_w]))

                    depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_d405, alpha=0.03), cv2.COLORMAP_JET)
                    cv2.putText(depth_vis, "D405 Depth", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    cv2.imshow("D405 Depth Preview", depth_vis)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break

                    elapsed = time.time() - loop_start
                    if elapsed < interval:
                        time.sleep(interval - elapsed)

                # 停止机械臂控制线程（发零速 + 断开 RTDE Control，避免超时）
                arm_ctrl.stop()

                # ── 保存本条轨迹 ──────────────────────────────────────
                if len(images_main) < 30:
                    print(f"  轨迹太短（{len(images_main)} 帧），丢弃")
                    print("  注意：跳过此条，继续下一条。如需重录请在本轮结束后重新开始。")
                    continue

                save_path = SAVE_DIR / f"episode_{episode:04d}.npz"
                np.savez_compressed(
                    save_path,
                    images=np.array(images_main,    dtype=np.uint8),
                    images_wrist=np.array(images_wrist, dtype=np.uint8),
                    depths_d405=np.array(depths_d405,   dtype=np.uint16),
                    d405_depth_scale=np.float64(pipe_d405["depth_scale"]),
                    tcp_poses=np.array(tcp_poses,        dtype=np.float64),
                    joint_positions=np.array(joint_positions, dtype=np.float64),
                    gripper=np.array(gripper_states,     dtype=np.float64),
                    instruction=task_instruction,
                    fps=FPS,
                )
                print(f"  ✓ 已保存: {save_path} ({len(images_main)} 帧) | {task_instruction}")
                episode     += 1
                saved_count += 1

                if step_num < total_in_round:
                    remaining = sequence[step_idx + 1:]
                    print(f"\n  剩余 {len(remaining)} 条：")
                    for r_idx, (r_instr, _) in enumerate(remaining, step_num + 1):
                        print(f"    {r_idx:2d}. {r_instr}")
                    input("  按 Enter 继续录制下一条...")

            print(f"\n{'=' * 60}")
            print(f"本轮采集完成！累计保存 {saved_count} 条轨迹。")
            print(f"{'=' * 60}")
            cont = input("继续采集下一轮？(y/n): ")
            if cont.lower() != "y":
                break

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        # 停止机械臂控制线程（内部会发零速并断开）
        try:
            arm_ctrl.stop()
        except Exception:
            pass
        if rtde_r is not None:
            try:
                if hasattr(rtde_r, "disconnect"):
                    rtde_r.disconnect()
            except Exception:
                pass
        # 停止 3D Mouse
        mouse.stop()
        # 断开夹爪
        gripper.disconnect()
        # 停止摄像头
        for p in pipelines:
            try: p["pipe"].stop()
            except Exception: pass
        cv2.destroyAllWindows()
        print(f"\n采集结束，本次保存 {saved_count} 条轨迹到 {SAVE_DIR}")


if __name__ == "__main__":
    main()
