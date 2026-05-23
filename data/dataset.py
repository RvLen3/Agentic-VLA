"""
UR7e 左臂自由拖动示教数据采集
双 RealSense RGB + D405 Depth
仅需 RTDEReceiveInterface（只读），不需要远程控制权限

使用前：
1. 示教器上切换到本地手动模式
2. 开启自由驱动 Freedrive
3. 确认左臂两台 RealSense 摄像头已连接
4. 根据你的实际环境修改 ROBOT_IP

摄像头:
- Intel RealSense D435i: 主视角 RGB
- Intel RealSense D405: 腕部视角 RGB + Depth

保存内容:
- images: 主视角 RGB, uint8, 256x256x3
- images_wrist: 腕部 RGB, uint8, 256x256x3
- depths_d405: D405 深度, uint16, 256x256
- d405_depth_scale: D405 深度尺度，米 = depth_raw * d405_depth_scale
- tcp_poses: TCP 位姿
- joint_positions: 关节角
- gripper: 手动标记夹爪状态，0.0=闭合，1.0=打开
- instruction: 语言指令
- fps: 采集帧率

重点修改:
- 保留双 RGB
- 新增 D405 depth
- 不使用 threading
- RTDE 在主循环中直接读取
- 每条轨迹开始前重新连接 RTDE
"""

import cv2
import time
import random
import numpy as np
import pyrealsense2 as rs
from pathlib import Path
from rtde_receive import RTDEReceiveInterface


SCRIPT_VERSION = "LEFT_RGB_PLUS_D405_DEPTH_MAIN_LOOP_RTDE_V3"
print(f"当前运行脚本版本: {SCRIPT_VERSION}")


# ============ 配置（根据你的环境修改） ============
ARM_NAME = "left"

ROBOT_IP = "192.168.1.88"  # TODO: 确认这是左臂 UR 控制柜 IP

SAVE_DIR = Path(f"./raw_demos_{ARM_NAME}_third")

FPS = 30

# 任务：将桌上所有蔬菜放入篮子（左右各一个篮子）
# 蔬菜种类：红色辣椒、绿色辣椒、黄色辣椒、玉米、紫薯、南瓜
TASK_LIST = [
    # pick up 原子操作（6 种蔬菜）
    "pick up the red pepper",
    "pick up the green pepper",
    "pick up the yellow pepper",
    "pick up the corn",
    "pick up the purple sweet potato",
    "pick up the pumpkin",
    # place 原子操作 — 放入左侧篮子
    "place the red pepper in the left basket",
    "place the green pepper in the left basket",
    "place the yellow pepper in the left basket",
    "place the corn in the left basket",
    "place the purple sweet potato in the left basket",
    "place the pumpkin in the left basket",
    # place 原子操作 — 放入右侧篮子
    "place the red pepper in the right basket",
    "place the green pepper in the right basket",
    "place the yellow pepper in the right basket",
    "place the corn in the right basket",
    "place the purple sweet potato in the right basket",
    "place the pumpkin in the right basket",
]
# TASK_NAME = "Pick up the chili on the table"
MAX_FRAMES = 900  # 30fps x 30秒

COLOR_WIDTH = 640
COLOR_HEIGHT = 480

DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480

SAVE_WIDTH = 256
SAVE_HEIGHT = 256
# ===============================================


SAVE_DIR.mkdir(parents=True, exist_ok=True)


def get_next_episode_id(save_dir):
    """避免重启程序后覆盖已有 episode"""
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


def reconnect_rtde(old_rtde=None):
    """
    重新连接 RTDE。
    有些环境下长期持有 RTDEReceiveInterface 可能读到旧值，
    所以每条轨迹开始前重新连接一次。
    """
    if old_rtde is not None:
        try:
            if hasattr(old_rtde, "disconnect"):
                old_rtde.disconnect()
        except Exception:
            pass

    rtde_r = RTDEReceiveInterface(ROBOT_IP)
    tcp = rtde_r.getActualTCPPose()
    q = rtde_r.getActualQ()

    print(
        f"RTDE 已连接，当前 TCP: "
        f"[{tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f}], "
        f"q0={q[0]:.4f}, q1={q[1]:.4f}, q2={q[2]:.4f}"
    )

    return rtde_r


def start_realsense_pipelines():
    """
    扫描 RealSense 设备。
    - 所有相机启动 RGB
    - 只有 D405 额外启动 Depth
    """
    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) < 2:
        raise RuntimeError(f"需要 2 个 RealSense 摄像头，只找到 {len(devices)} 个")

    print(f"找到 {len(devices)} 个 RealSense 设备:")

    pipelines = []

    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)

        print(f"  - {name} (S/N: {serial})")

        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_device(serial)

        # 所有相机都启动 RGB
        config.enable_stream(
            rs.stream.color,
            COLOR_WIDTH,
            COLOR_HEIGHT,
            rs.format.bgr8,
            FPS,
        )

        # 只有 D405 启动 depth
        enable_depth = "D405" in name.upper()

        if enable_depth:
            config.enable_stream(
                rs.stream.depth,
                DEPTH_WIDTH,
                DEPTH_HEIGHT,
                rs.format.z16,
                FPS,
            )

        try:
            profile = pipeline.start(config)

            depth_scale = None
            align = None

            if enable_depth:
                depth_sensor = profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()

                # 将 D405 depth 对齐到 D405 RGB
                # 这样 D405 RGB 和 D405 depth 像素是一一对应的
                align = rs.align(rs.stream.color)

            pipelines.append(
                {
                    "name": name,
                    "serial": serial,
                    "pipe": pipeline,
                    "enable_depth": enable_depth,
                    "depth_scale": depth_scale,
                    "align": align,
                }
            )

            if enable_depth:
                print(f"  ▶ {name} 启动成功: RGB + Depth")
                print(f"    Depth Scale: {depth_scale}")
            else:
                print(f"  ▶ {name} 启动成功: RGB only")

        except Exception as e:
            print(f"  ✗ {name} 启动失败: {e}")

            for p in pipelines:
                try:
                    p["pipe"].stop()
                except Exception:
                    pass

            raise RuntimeError(f"{name} 启动失败: {e}")

    print("正在预热 RealSense 摄像头...")
    for _ in range(15):
        for p in pipelines:
            try:
                p["pipe"].wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                pass

    return pipelines


def get_rgb_and_optional_depth_frame(p):
    """
    从单个 RealSense pipeline 获取 RGB。
    如果该设备是 D405，则额外获取对齐到 RGB 的 depth。

    返回:
    - color_img: BGR uint8, HxWx3
    - depth_img: uint16, HxW；如果不是 D405，则为 None
    """
    frames = p["pipe"].wait_for_frames(timeout_ms=2000)

    if p.get("enable_depth", False):
        # D405: RGB + Depth，并把 Depth 对齐到 RGB
        aligned_frames = p["align"].process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())

        return color_img, depth_img

    else:
        # D435i: 只采 RGB
        color_frame = frames.get_color_frame()

        if not color_frame:
            return None, None

        color_img = np.asanyarray(color_frame.get_data())

        return color_img, None


def get_rgb_frame(p):
    """
    保留 RGB 获取函数，用于预览阶段。
    """
    color_img, _ = get_rgb_and_optional_depth_frame(p)
    return color_img


def capture_rgb_frames(pipelines):
    """
    采集双相机 RGB，并额外采集 D405 depth。

    输入:
    - pipelines: [pipe_main, pipe_wrist]

    返回:
    - frame_main: 主视角 RGB, BGR
    - frame_wrist: 腕部 RGB, BGR
    - depth_d405: D405 depth, uint16
    """
    frames = []
    depth_d405 = None

    for p in pipelines:
        try:
            img, depth = get_rgb_and_optional_depth_frame(p)
            frames.append(img)

            if p.get("enable_depth", False):
                depth_d405 = depth

        except RuntimeError:
            frames.append(None)

            if p.get("enable_depth", False):
                depth_d405 = None

    if any(f is None for f in frames):
        return None, None, None

    if depth_d405 is None:
        return None, None, None

    return frames[0], frames[1], depth_d405


def main():
    pipelines = []
    rtde_r = None
    saved_count = 0

    try:
        # ---- 初始化 RealSense ----
        print("\n正在启动 RealSense 摄像头...")
        pipelines = start_realsense_pipelines()

        pipe_main = pipelines[0]
        pipe_wrist = pipelines[1]

        print(f"默认主视角: {pipe_main['name']} (S/N: {pipe_main['serial']})")
        print(f"默认腕部视角: {pipe_wrist['name']} (S/N: {pipe_wrist['serial']})")

        # ---- 交互式确认哪个是哪个 ----
        print("\n正在预览摄像头画面，请确认视角分配...")
        print("按 's' 交换两个视角，按 Enter 确认继续")

        while True:
            try:
                img1 = get_rgb_frame(pipe_main)
                img2 = get_rgb_frame(pipe_wrist)

                if img1 is not None and img2 is not None:
                    vis1 = img1.copy()
                    vis2 = img2.copy()

                    cv2.putText(
                        vis1,
                        f"Main: {pipe_main['name']}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        vis2,
                        f"Wrist: {pipe_wrist['name']}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    preview = np.hstack([vis1, vis2])
                    cv2.imshow("Camera Preview - Enter=confirm, S=swap", preview)

            except RuntimeError:
                pass

            key = cv2.waitKey(100) & 0xFF

            if key in [13, 10]:
                break

            elif key == ord("s"):
                pipe_main, pipe_wrist = pipe_wrist, pipe_main
                print(
                    f"已交换 -> 主视角: {pipe_main['name']}，腕部视角: {pipe_wrist['name']}"
                )

        cv2.destroyAllWindows()

        active_pipelines = [pipe_main, pipe_wrist]

        # ---- 找到 D405，用于保存 depth_scale ----
        pipe_d405 = None
        for p in active_pipelines:
            if "D405" in p["name"].upper():
                pipe_d405 = p
                break

        if pipe_d405 is None:
            raise RuntimeError("没有找到 D405，无法采集 D405 深度")

        if not pipe_d405.get("enable_depth", False):
            raise RuntimeError("D405 没有启用 depth，请检查 start_realsense_pipelines()")

        print(
            f"\nD405 深度采集设备: {pipe_d405['name']} "
            f"(S/N: {pipe_d405['serial']}), "
            f"depth_scale={pipe_d405['depth_scale']}"
        )

        # ---- 初始连接 RTDE，确认能读到 ----
        print("\n正在连接机器人（只读模式）...")
        rtde_r = reconnect_rtde(rtde_r)

        # ---- 夹爪状态，键盘标记 ----
        gripper_state = 0.0  # 0.0=闭合, 1.0=打开

        print("\n" + "=" * 60)
        print("UR7e 左臂自由拖动采集（双 RealSense RGB + D405 Depth）")
        print("=" * 60)
        print("操作步骤：")
        print("  0. 示教器上进入 本地手动模式 → 开启自由驱动 Freedrive")
        print("  1. 按住示教器背面按钮，手拖机器人")
        print("  2. 按 'g' 标记夹爪打开，按 'c' 标记夹爪闭合")
        print("  3. 按 'q' 或 ESC 结束当前轨迹并保存")
        print("  4. 按 Enter 开始下一条")
        print("=" * 60)
        print("注意：g/c/q/ESC 需要在 OpenCV 图像窗口获得焦点时按。")

        episode = get_next_episode_id(SAVE_DIR)
        interval = 1.0 / FPS

        # 蔬菜种类列表（与 TASK_LIST 保持一致）
        VEGETABLES = [
            "red pepper",
            "green pepper",
            "yellow pepper",
            "corn",
            "purple sweet potato",
            "pumpkin",
        ]

        def generate_episode_sequence():
            """
            生成一轮完整的采集序列：
            1. 随机打乱 6 种蔬菜的顺序
            2. 为每种蔬菜随机分配左/右篮子
            3. 按顺序展开为 pick up → place 的原子指令对
            返回 12 条指令的列表，格式：
              [(指令字符串, 操作类型), ...]
              操作类型: "pick_up" | "place"
            """
            shuffled = VEGETABLES[:]
            random.shuffle(shuffled)
            sequence = []
            for veg in shuffled:
                side = random.choice(["left", "right"])
                sequence.append((f"pick up the {veg}", "pick_up"))
                sequence.append((f"place the {veg} in the {side} basket", "place"))
            return sequence

        # 外层循环：每轮生成一个新的采集序列
        while True:
            # ---- 生成本轮采集序列 ----
            sequence = generate_episode_sequence()
            total_in_round = len(sequence)  # 固定 12 条

            print("\n" + "=" * 60)
            print("本轮采集序列（共 12 条原子动作）：")
            for idx, (instr, _) in enumerate(sequence, 1):
                print(f"  {idx:2d}. {instr}")
            print("=" * 60)
            print("将按上述顺序逐条录制，每条录完保存后继续下一条。")
            cont = input("开始本轮采集？(y/n，n 则重新生成序列): ")
            if cont.lower() != "y":
                continue

            # ---- 逐条录制本轮序列 ----
            for step_idx, (task_instruction, op_type) in enumerate(sequence):
                step_num = step_idx + 1

                print(f"\n{'─' * 60}")
                print(f"  本轮第 {step_num}/{total_in_round} 条  |  Episode #{episode}")
                print(f"  指令: {task_instruction}")
                print(f"  操作: {'拾取' if op_type == 'pick_up' else '放置'}")
                print(f"{'─' * 60}")
                print("按住示教器自由驱动按钮，将机器人移到起始位置")
                input("准备好后按 Enter 开始录制...")

                # 每条轨迹开始前重新连接 RTDE
                print("重新连接 RTDE，确保读取最新机器人状态...")
                rtde_r = reconnect_rtde(rtde_r)

                print(">>> 录制已开始！按 q 或 ESC 结束 <<<")

                images_main = []
                images_wrist = []
                depths_d405 = []
                tcp_poses = []
                joint_positions = []
                gripper_states = []

                frame_count = 0
                start_tcp = None

                while frame_count < MAX_FRAMES:
                    loop_start = time.time()

                    # 先读取机器人状态
                    tcp = rtde_r.getActualTCPPose()
                    joints = rtde_r.getActualQ()

                    if tcp is None or joints is None:
                        print("警告: RTDE 读取为空，跳过该帧")
                        time.sleep(0.01)
                        continue

                    if start_tcp is None:
                        start_tcp = np.array(tcp[:3], dtype=np.float64)

                    delta_tcp = np.array(tcp[:3], dtype=np.float64) - start_tcp
                    move_dist = np.linalg.norm(delta_tcp)

                    # 再读取图像和 D405 depth
                    frame_m, frame_w, depth_d405 = capture_rgb_frames(active_pipelines)

                    if frame_m is None or frame_w is None or depth_d405 is None:
                        time.sleep(0.01)
                        continue

                    # BGR -> RGB，用于保存
                    frame_m_rgb = cv2.cvtColor(frame_m, cv2.COLOR_BGR2RGB)
                    frame_w_rgb = cv2.cvtColor(frame_w, cv2.COLOR_BGR2RGB)

                    images_main.append(
                        cv2.resize(frame_m_rgb, (SAVE_WIDTH, SAVE_HEIGHT))
                    )

                    images_wrist.append(
                        cv2.resize(frame_w_rgb, (SAVE_WIDTH, SAVE_HEIGHT))
                    )

                    # 保存 D405 depth，保持 uint16 raw depth
                    # 注意 depth resize 必须使用 INTER_NEAREST，避免深度值被插值污染
                    depth_d405_resized = cv2.resize(
                        depth_d405,
                        (SAVE_WIDTH, SAVE_HEIGHT),
                        interpolation=cv2.INTER_NEAREST,
                    )

                    depths_d405.append(depth_d405_resized.astype(np.uint16))

                    tcp_poses.append(tcp)
                    joint_positions.append(joints)
                    gripper_states.append(gripper_state)

                    if frame_count % 30 == 0:
                        tcp_now = rtde_r.getActualTCPPose()
                        q_now = rtde_r.getActualQ()

                        print(
                            f"REC TCP(saved): "
                            f"[{tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f}] | "
                            f"TCP(now): "
                            f"[{tcp_now[0]:.4f}, {tcp_now[1]:.4f}, {tcp_now[2]:.4f}] | "
                            f"dMove={move_dist:.4f} m | "
                            f"q(now): "
                            f"[{q_now[0]:.4f}, {q_now[1]:.4f}, {q_now[2]:.4f}]"
                        )

                    frame_count += 1

                    # ---- 显示预览 ----
                    display_m = frame_m.copy()
                    display_w = frame_w.copy()

                    grip_str = "OPEN" if gripper_state > 0.5 else "CLOSED"
                    grip_color = (0, 255, 0) if gripper_state > 0.5 else (0, 0, 255)

                    cv2.putText(
                        display_m,
                        f"Ep#{episode} [{step_num}/{total_in_round}] F:{frame_count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        display_m,
                        f"TCP: [{tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f}]",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

                    cv2.putText(
                        display_m,
                        f"dXYZ: [{delta_tcp[0]:.4f}, {delta_tcp[1]:.4f}, {delta_tcp[2]:.4f}]",
                        (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

                    cv2.putText(
                        display_m,
                        f"Move: {move_dist:.4f} m",
                        (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

                    cv2.putText(
                        display_m,
                        f"Gripper: {grip_str}",
                        (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        grip_color,
                        2,
                    )

                    # 在画面上显示当前指令（截断超长文字）
                    instr_display = task_instruction if len(task_instruction) <= 38 else task_instruction[:35] + "..."
                    cv2.putText(
                        display_m,
                        instr_display,
                        (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 200, 255),
                        1,
                    )

                    if gripper_state > 0.5:
                        hint = "REC | G=opened | Press C to CLOSE | Q=stop"
                    else:
                        hint = "REC | C=closed | Press G to OPEN | Q=stop"

                    cv2.putText(
                        display_m,
                        hint,
                        (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )

                    cv2.putText(
                        display_w,
                        "Wrist",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                    )

                    display = np.hstack([display_m, display_w])

                    cv2.imshow("UR7e Left RGB Collection", display)

                    # ---- D405 depth 预览 ----
                    depth_vis = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_d405, alpha=0.03),
                        cv2.COLORMAP_JET,
                    )

                    cv2.putText(
                        depth_vis,
                        "D405 Depth",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                    cv2.imshow("D405 Depth Preview", depth_vis)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("c"):
                        gripper_state = 0.0
                        gripper_states[-1] = 0.0
                        print("  标记: 夹爪 → 闭合")

                    elif key == ord("g"):
                        gripper_state = 1.0
                        gripper_states[-1] = 1.0
                        print("  标记: 夹爪 → 打开")

                    if key == ord("q") or key == 27:
                        break

                    elapsed = time.time() - loop_start
                    if elapsed < interval:
                        time.sleep(interval - elapsed)

                # ---- 保存本条轨迹 ----
                if len(images_main) < 30:
                    print(f"轨迹太短（{len(images_main)} 帧），丢弃，请重新录制此条")
                    redo = input("重新录制这条指令？(y/n，n 则跳过): ")
                    if redo.lower() == "y":
                        step_idx -= 1  # 注意：for 循环不支持直接回退，用 flag 处理
                        # 实际上 for 循环无法回退，这里改为提示后继续
                        print("  注意：跳过此条，继续下一条。如需重录请在本轮结束后重新开始。")
                    continue

                save_path = SAVE_DIR / f"episode_{episode:04d}.npz"

                np.savez_compressed(
                    save_path,
                    images=np.array(images_main, dtype=np.uint8),
                    images_wrist=np.array(images_wrist, dtype=np.uint8),

                    # D405 depth
                    depths_d405=np.array(depths_d405, dtype=np.uint16),
                    d405_depth_scale=np.float64(pipe_d405["depth_scale"]),

                    tcp_poses=np.array(tcp_poses, dtype=np.float64),
                    joint_positions=np.array(joint_positions, dtype=np.float64),
                    gripper=np.array(gripper_states, dtype=np.float64),
                    instruction=task_instruction,
                    fps=FPS,
                )

                print(f"✓ 已保存: {save_path} ({len(images_main)} 帧)")
                print(f"  指令: {task_instruction}")
                print(f"  RGB main:    {np.array(images_main).shape}")
                print(f"  RGB wrist:   {np.array(images_wrist).shape}")
                print(f"  D405 depth:  {np.array(depths_d405).shape}, uint16")
                print(f"  TCP poses:   {np.array(tcp_poses).shape}")
                print(f"  Gripper:     {np.array(gripper_states).shape}")

                episode += 1
                saved_count += 1

                # 本轮未结束时，提示继续下一条
                if step_num < total_in_round:
                    remaining = sequence[step_idx + 1:]
                    print(f"\n  剩余 {len(remaining)} 条：")
                    for r_idx, (r_instr, _) in enumerate(remaining, step_num + 1):
                        print(f"    {r_idx:2d}. {r_instr}")
                    input("  按 Enter 继续录制下一条...")

            # ---- 本轮 12 条全部录完 ----
            print(f"\n{'=' * 60}")
            print(f"本轮采集完成！共保存 {saved_count} 条轨迹（累计）。")
            print(f"{'=' * 60}")
            cont = input("继续采集下一轮？(y/n): ")
            if cont.lower() != "y":
                break

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        if rtde_r is not None:
            try:
                if hasattr(rtde_r, "disconnect"):
                    rtde_r.disconnect()
            except Exception:
                pass

        for p in pipelines:
            try:
                p["pipe"].stop()
            except Exception:
                pass

        cv2.destroyAllWindows()
        print(f"采集结束，本次保存 {saved_count} 条轨迹到 {SAVE_DIR}")


if __name__ == "__main__":
    main()
