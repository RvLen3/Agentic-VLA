"""
将 raw_demos_left_rgbd/*.npz 转换为 HDF5 (robomind-ur RGB-D 格式)

原则:
    只新增 depth 相关字段，不改变任何旧字段的格式和含义。

输入:
    fine-tune/raw_demos_left_rgbd/episode_XXXX.npz

输出:
    fine-tune/training_data_left_rgbd/episode_XXXX.hdf5

原有 HDF5 结构保持不变:
    observations/images/cam_high      uint8,  T x 256 x 256 x 3
    observations/images/cam_wrist     uint8,  T x 256 x 256 x 3

    puppet/end_effector               float32, T x 6
        [x, y, z, roll, pitch, yaw]

    puppet/joint_position             float32, T x 7
        保持原格式: 前 6 维为 0，最后 1 维为 gripper

    language_instruction              string

新增 HDF5 字段:
    observations/depths/cam_high      uint16, T x 256 x 256
    observations/depths/cam_wrist     uint16, T x 256 x 256
    depth_scale_main                  float32
    depth_scale_wrist                 float32
    camera_info_json                  string
"""

import numpy as np
import h5py
from pathlib import Path
from scipy.spatial.transform import Rotation


INPUT_DIR = Path(__file__).parent / "raw_demos_left_rgbd"
OUTPUT_DIR = Path(__file__).parent / "training_data_left_rgbd"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def axisangle_to_euler_xyz(rvec):
    """
    UR RTDE getActualTCPPose() 返回:
        [x, y, z, rx, ry, rz]

    其中 [rx, ry, rz] 是 axis-angle / rotation vector。
    这里保持原逻辑，转成 euler xyz。
    """
    R = Rotation.from_rotvec(rvec)
    return R.as_euler("xyz")


def read_string_from_npz(data, key, default=""):
    """
    兼容 np.savez 保存字符串后的读取。
    """
    if key not in data.files:
        return default

    value = data[key]

    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        return str(value)

    return str(value)


def main():
    npz_files = sorted(INPUT_DIR.glob("*.npz"))

    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"找到 {len(npz_files)} 条轨迹\n")

    if len(npz_files) == 0:
        print("没有找到 .npz 文件，请检查 INPUT_DIR 是否正确。")
        return

    start_idx = 0
    total_frames = 0

    for i, npz_path in enumerate(npz_files):
        data = np.load(npz_path, allow_pickle=True)

        # -------- 原有基础数据：保持不变 --------
        images = data["images"]
        tcp_poses = data["tcp_poses"]
        gripper = data["gripper"]
        instruction = str(data["instruction"])

        images_wrist = data.get("images_wrist", None)

        T = len(images)
        total_frames += T

        # -------- 新增 depth 数据：只读，不影响旧字段 --------
        depths = data["depths"] if "depths" in data.files else None
        depths_wrist = data["depths_wrist"] if "depths_wrist" in data.files else None

        depth_scale_main = (
            float(data["depth_scale_main"])
            if "depth_scale_main" in data.files
            else 1.0
        )

        depth_scale_wrist = (
            float(data["depth_scale_wrist"])
            if "depth_scale_wrist" in data.files
            else 1.0
        )

        camera_info_json = read_string_from_npz(data, "camera_info_json", "{}")

        # -------- 长度检查 --------
        if len(tcp_poses) != T:
            raise ValueError(
                f"{npz_path.name}: images 和 tcp_poses 长度不一致: "
                f"{T} vs {len(tcp_poses)}"
            )

        if len(gripper) != T:
            raise ValueError(
                f"{npz_path.name}: images 和 gripper 长度不一致: "
                f"{T} vs {len(gripper)}"
            )

        if images_wrist is not None and len(images_wrist) != T:
            raise ValueError(
                f"{npz_path.name}: images 和 images_wrist 长度不一致: "
                f"{T} vs {len(images_wrist)}"
            )

        if depths is not None and len(depths) != T:
            raise ValueError(
                f"{npz_path.name}: images 和 depths 长度不一致: "
                f"{T} vs {len(depths)}"
            )

        if depths_wrist is not None and len(depths_wrist) != T:
            raise ValueError(
                f"{npz_path.name}: images 和 depths_wrist 长度不一致: "
                f"{T} vs {len(depths_wrist)}"
            )

        # -------- 原有 end_effector 逻辑：保持不变 --------
        eulers = np.array(
            [axisangle_to_euler_xyz(tcp[3:6]) for tcp in tcp_poses]
        )

        end_effector = np.concatenate(
            [tcp_poses[:, :3], eulers],
            axis=1
        ).astype(np.float32)

        # -------- 原有 joint_position 逻辑：保持不变 --------
        # 注意:
        # 这里不要写入真实 6 维关节角。
        # 保持原 robomind-ur 格式:
        #   前 6 维为 0
        #   最后 1 维为 gripper
        joint_position = np.zeros((T, 7), dtype=np.float32)
        joint_position[:, -1] = gripper.flatten()

        # -------- 输出 HDF5 --------
        out_path = OUTPUT_DIR / f"episode_{start_idx + i:04d}.hdf5"

        with h5py.File(out_path, "w") as f:
            # 原有 observations/images 结构：保持不变
            obs_grp = f.create_group("observations")
            img_grp = obs_grp.create_group("images")

            img_grp.create_dataset(
                "cam_high",
                data=images,
                dtype=np.uint8,
                compression="gzip"
            )

            if images_wrist is not None:
                img_grp.create_dataset(
                    "cam_wrist",
                    data=images_wrist,
                    dtype=np.uint8,
                    compression="gzip"
                )

            # 新增 observations/depths
            # 只新增，不影响 observations/images
            if depths is not None or depths_wrist is not None:
                depth_grp = obs_grp.create_group("depths")

                if depths is not None:
                    depth_grp.create_dataset(
                        "cam_high",
                        data=depths,
                        dtype=np.uint16,
                        compression="gzip"
                    )

                if depths_wrist is not None:
                    depth_grp.create_dataset(
                        "cam_wrist",
                        data=depths_wrist,
                        dtype=np.uint16,
                        compression="gzip"
                    )

            # 原有 puppet 结构：保持不变
            puppet_grp = f.create_group("puppet")

            puppet_grp.create_dataset(
                "end_effector",
                data=end_effector
            )

            puppet_grp.create_dataset(
                "joint_position",
                data=joint_position
            )

            # 原有 language_instruction：保持不变
            f.create_dataset(
                "language_instruction",
                data=instruction
            )

            # 新增 depth metadata
            f.create_dataset(
                "depth_scale_main",
                data=np.float32(depth_scale_main)
            )

            f.create_dataset(
                "depth_scale_wrist",
                data=np.float32(depth_scale_wrist)
            )

            str_dtype = h5py.string_dtype(encoding="utf-8")
            f.create_dataset(
                "camera_info_json",
                data=camera_info_json,
                dtype=str_dtype
            )

        print(
            f"  [{i + 1}/{len(npz_files)}] {out_path.name} — {T} 帧 "
            f"xyz=[{tcp_poses[0,0]:.3f},{tcp_poses[0,1]:.3f},{tcp_poses[0,2]:.3f}] "
            f"depth={'yes' if depths is not None else 'no'} "
            f"wrist_depth={'yes' if depths_wrist is not None else 'no'}"
        )

    print(f"\n转换完成: {len(npz_files)} 条轨迹, 共 {total_frames} 帧")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
