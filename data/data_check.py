import time
import numpy as np
from pathlib import Path


# 当前 RGB-D 采集脚本的保存目录
demo_dir = Path(__file__).parent / "raw_demos_left_third"
files = sorted(demo_dir.glob("*.npz"))

print(f"正在检查目录: {demo_dir.resolve()}")
print(f"共 {len(files)} 条轨迹\n")

if len(files) == 0:
    print("没有找到 .npz 文件，请检查路径是否正确。")
    raise SystemExit


lengths = []

for idx, f in enumerate(files):
    data = np.load(f, allow_pickle=True)

    images = data["images"]
    tcp = data["tcp_poses"]
    grip = data["gripper"].reshape(-1)

    length = len(images)
    lengths.append(length)

    print("=" * 80)
    print(f"[{idx + 1}/{len(files)}] {f.name}")
    print(f"文件修改时间: {time.ctime(f.stat().st_mtime)}")
    print(f"帧数: {length}")

    # ---------- 基础字段 ----------
    print("\n基础字段:")
    print(f"  images: {images.shape}, dtype={images.dtype}")

    if "images_wrist" in data.files:
        images_wrist = data["images_wrist"]
        print(f"  images_wrist: {images_wrist.shape}, dtype={images_wrist.dtype}")
    else:
        images_wrist = None
        print("  images_wrist: 不存在")

    print(f"  tcp_poses: {tcp.shape}, dtype={tcp.dtype}")

    if "joint_positions" in data.files:
        joints = data["joint_positions"]
        print(f"  joint_positions: {joints.shape}, dtype={joints.dtype}")
    else:
        joints = None
        print("  joint_positions: 不存在")

    print(f"  gripper: {grip.shape}, dtype={grip.dtype}")

    if "instruction" in data.files:
        print(f"  instruction: {str(data['instruction'])}")

    if "fps" in data.files:
        print(f"  fps: {data['fps']}")

    if "arm" in data.files:
        print(f"  arm: {str(data['arm'])}")

    # ---------- TCP 运动范围 ----------
    print("\nTCP 位置:")
    print(f"  起始 XYZ: {[f'{v:.4f}' for v in tcp[0, :3]]}")
    print(f"  终止 XYZ: {[f'{v:.4f}' for v in tcp[-1, :3]]}")

    xyz_min = tcp[:, :3].min(axis=0)
    xyz_max = tcp[:, :3].max(axis=0)
    xyz_range = xyz_max - xyz_min

    print(f"  X: [{xyz_min[0]:.4f}, {xyz_max[0]:.4f}], dx={xyz_range[0]:.4f}")
    print(f"  Y: [{xyz_min[1]:.4f}, {xyz_max[1]:.4f}], dy={xyz_range[1]:.4f}")
    print(f"  Z: [{xyz_min[2]:.4f}, {xyz_max[2]:.4f}], dz={xyz_range[2]:.4f}")

    # ---------- 关节角范围 ----------
    if joints is not None:
        print("\n关节角范围:")
        joint_min = joints.min(axis=0)
        joint_max = joints.max(axis=0)
        joint_range = joint_max - joint_min

        for j in range(joints.shape[1]):
            print(
                f"  q{j}: [{joint_min[j]:.6f}, {joint_max[j]:.6f}], "
                f"range={joint_range[j]:.6f}"
            )

    # ---------- 夹爪状态 ----------
    print("\n夹爪状态:")

    unique_grip = np.unique(grip)

    # 按你采集代码里的定义:
    # 0.0 = 闭合
    # 1.0 = 打开
    open_frames = int(np.sum(grip > 0.5))
    closed_frames = int(np.sum(grip <= 0.5))

    print(f"  unique values: {unique_grip}")
    print(f"  打开帧数 gripper=1: {open_frames}/{length}")
    print(f"  闭合帧数 gripper=0: {closed_frames}/{length}")
    print(f"  grip.sum(): {float(grip.sum()):.1f}")

    # ---------- 图像变化检查 ----------
    print("\n图像变化:")

    img_diff = np.mean(
        np.abs(images[-1].astype(np.float32) - images[0].astype(np.float32))
    )
    print(f"  主视角首尾平均像素差异: {img_diff:.2f}")

    if images_wrist is not None:
        wrist_diff = np.mean(
            np.abs(
                images_wrist[-1].astype(np.float32)
                - images_wrist[0].astype(np.float32)
            )
        )
        print(f"  腕部视角首尾平均像素差异: {wrist_diff:.2f}")

    # ---------- 腕部 depth 检查 ----------
    print("\n腕部 depth:")

    if "depths_wrist" in data.files:
        depths_wrist = data["depths_wrist"]
        print(f"  depths_wrist: {depths_wrist.shape}, dtype={depths_wrist.dtype}")

        if "depth_scale_wrist" in data.files:
            depth_scale_wrist = float(data["depth_scale_wrist"])
        else:
            depth_scale_wrist = 1.0

        print(f"  depth_scale_wrist: {depth_scale_wrist}")

        valid = depths_wrist > 0
        valid_ratio = valid.mean()

        print(f"  有效 depth 像素比例: {valid_ratio * 100:.2f}%")

        if valid.any():
            depth_m = depths_wrist.astype(np.float32) * depth_scale_wrist
            print(
                f"  depth 范围: "
                f"[{depth_m[valid].min():.4f}, {depth_m[valid].max():.4f}] m"
            )
            print(f"  depth 平均值: {depth_m[valid].mean():.4f} m")
        else:
            print("  警告: depths_wrist 全部为 0，没有有效深度。")
    else:
        print("  不存在 depths_wrist 字段。")

    # ---------- 可用性判断 ----------
    print("\n可用性判断:")

    usable = True

    if length < 30:
        print("  不可用: 轨迹太短，少于 30 帧。")
        usable = False
    else:
        print("  帧数正常。")

    if xyz_range.max() < 0.005:
        print("  不可用: TCP XYZ 几乎没有变化。")
        usable = False
    else:
        print("  TCP XYZ 有变化。")

    if joints is not None:
        if joint_range.max() < 0.005:
            print("  不可用: 关节角几乎没有变化。")
            usable = False
        else:
            print("  关节角有变化。")

    if len(unique_grip) == 1:
        print("  提醒: 夹爪状态全程不变。")
    else:
        print("  夹爪状态有变化。")

    if "depths_wrist" in data.files:
        if valid_ratio < 0.05:
            print("  警告: 腕部 depth 有效像素比例太低。")
        else:
            print("  腕部 depth 看起来正常。")

    if usable:
        print("\n  结论: 这条轨迹基本可用。")
    else:
        print("\n  结论: 这条轨迹不建议用于训练。")

    print()


print("=" * 80)
print("总体统计:")
print(f"  轨迹数: {len(files)}")
print(f"  平均长度: {np.mean(lengths):.0f} 帧")
print(f"  最短: {min(lengths)} 帧")
print(f"  最长: {max(lengths)} 帧")
print(f"  检查目录: {demo_dir.resolve()}")
