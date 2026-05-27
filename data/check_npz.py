"""
检查 npz 文件内容

用法：
  python check_npz.py episode_0000.npz
  python check_npz.py ./raw_demos_left_third/episode_0000.npz
  python check_npz.py ./raw_demos_left_third/  (检查目录下所有 npz)
"""

import sys
import numpy as np
from pathlib import Path


def check_one(path: Path):
    print(f"\n{'='*60}")
    print(f"文件: {path}")
    print(f"大小: {path.stat().st_size / 1024:.1f} KB")
    print(f"{'='*60}")

    data = np.load(str(path), allow_pickle=True)

    print(f"\n字段列表 ({len(data.files)} 个):")
    for key in sorted(data.files):
        arr = data[key]
        if arr.ndim == 0:
            # 标量
            print(f"  {key:25s}  scalar  value={arr.item()}")
        else:
            print(f"  {key:25s}  shape={str(arr.shape):20s}  dtype={arr.dtype}")

    # 详细信息
    print(f"\n详细内容:")

    if "instruction" in data.files:
        print(f"  指令: {data['instruction']}")

    if "fps" in data.files:
        print(f"  FPS: {data['fps']}")

    if "images" in data.files:
        imgs = data["images"]
        print(f"  主视角图像: {imgs.shape} ({imgs.nbytes/1024:.0f} KB)")
        print(f"    范围: [{imgs.min()}, {imgs.max()}]")

    if "images_wrist" in data.files:
        imgs = data["images_wrist"]
        print(f"  腕部图像: {imgs.shape} ({imgs.nbytes/1024:.0f} KB)")

    if "depths_d405" in data.files:
        depths = data["depths_d405"]
        print(f"  D405 深度: {depths.shape} ({depths.nbytes/1024:.0f} KB)")
        print(f"    范围: [{depths.min()}, {depths.max()}]")
        if "d405_depth_scale" in data.files:
            scale = float(data["d405_depth_scale"])
            print(f"    深度尺度: {scale} (最近={depths[depths>0].min()*scale:.3f}m, "
                  f"最远={depths.max()*scale:.3f}m)")

    if "tcp_poses" in data.files:
        tcp = data["tcp_poses"]
        print(f"  TCP 位姿: {tcp.shape}")
        print(f"    起始: [{tcp[0,0]:.4f}, {tcp[0,1]:.4f}, {tcp[0,2]:.4f}]")
        print(f"    结束: [{tcp[-1,0]:.4f}, {tcp[-1,1]:.4f}, {tcp[-1,2]:.4f}]")
        dist = np.linalg.norm(tcp[-1,:3] - tcp[0,:3])
        print(f"    总移动距离: {dist:.4f} m")

    if "joint_positions" in data.files:
        joints = data["joint_positions"]
        print(f"  关节角: {joints.shape}")
        print(f"    起始: [{', '.join(f'{j:.3f}' for j in joints[0])}]")

    if "gripper" in data.files:
        grip = data["gripper"]
        print(f"  夹爪状态: {grip.shape}")
        changes = np.where(np.diff(grip) != 0)[0]
        print(f"    初始: {'打开' if grip[0] > 0.5 else '闭合'}")
        print(f"    切换次数: {len(changes)}")
        if len(changes) > 0:
            for c in changes[:10]:
                print(f"      帧 {c+1}: → {'打开' if grip[c+1] > 0.5 else '闭合'}")

    n_frames = len(data["images"]) if "images" in data.files else 0
    fps = int(data["fps"]) if "fps" in data.files else 30
    duration = n_frames / fps if fps > 0 else 0
    print(f"\n  总帧数: {n_frames}, 时长: {duration:.1f}s")


def main():
    if len(sys.argv) < 2:
        print("用法: python check_npz.py <npz文件或目录>")
        sys.exit(1)

    path = Path(sys.argv[1])

    if path.is_dir():
        files = sorted(path.glob("episode_*.npz"))
        if not files:
            files = sorted(path.glob("*.npz"))
        print(f"找到 {len(files)} 个 npz 文件")
        for f in files:
            check_one(f)
    elif path.is_file():
        check_one(path)
    else:
        print(f"路径不存在: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
