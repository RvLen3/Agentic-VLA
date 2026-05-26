import os
import shutil
import multiprocessing
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np

# 核心真机环境组件
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.common.precise_sleep import precise_wait

# --- 环境补丁 ---
os.environ['OPENCV_FORK_TERMINATE'] = '1'
try:
    if multiprocessing.get_start_method(allow_none=True) != 'fork':
        multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    pass

# ======================================================
# 【映射逻辑：代码1验证成功的三步旋转映射】
# ======================================================
ROTATION_DEG = 100.0
TILT_DEG = -36.0
XY_ROT_DEG = 36.0

def get_teleop_matrix(theta_deg, tilt_deg, xy_rot_deg):
    theta, tilt, phi = np.radians([theta_deg, tilt_deg, xy_rot_deg])
    
    # 1. 第一步：水平基础向量
    u_base = np.array([0, -np.sin(theta), np.cos(theta)])
    zenith = np.array([1, 0, 0])
    
    # 2. 第二步：正交向量 + Tilt 倾斜
    v_base_h = np.cross(zenith, u_base)
    u_orig = u_base
    v_orig = v_base_h * np.cos(tilt) + zenith * np.sin(tilt)
    
    u_orig /= np.linalg.norm(u_orig)
    v_orig /= np.linalg.norm(v_orig)
    
    # 3. 第三步：平面内旋转 (phi)
    u_final = u_orig * np.cos(phi) - v_orig * np.sin(phi)
    v_final = u_orig * np.sin(phi) + v_orig * np.cos(phi)
    
    return u_final, v_final

# ======================================================

@click.command()
@click.option('--output', '-o', required=True, help="数据集保存目录")
@click.option('--robot_ip', '-ri', required=True, help="UR5e IP")
@click.option('--frequency', '-f', default=10, type=float, help="控制频率 (Hz)")
def main(output, robot_ip, frequency):
    # --- 关键修复：预防 Zarr 空文件报错 ---
    # 如果目录存在，先彻底删除，确保 RealEnv 初始化的是干净的结构
    if os.path.exists(output):
        print(f"[*] 清理旧数据目录: {output}")
        shutil.rmtree(output)

    cv2.setNumThreads(1)
    dt = 1 / frequency
    
    # 预计算映射矩阵向量
    u_unit, v_unit = get_teleop_matrix(ROTATION_DEG, TILT_DEG, XY_ROT_DEG)

    with SharedMemoryManager() as shm_manager:
        # 初始化环境
        env = RealEnv(
            output_dir=output,
            robot_ip=robot_ip,
            obs_image_resolution=(640, 480), 
            frequency=frequency,
            init_joints=False,
            enable_multi_cam_vis=True, 
            record_raw_video=True,
            shm_manager=shm_manager
        )

        with env:
            print("[*] 正在同步相机并设置曝光...")
            env.realsense.set_exposure(exposure=200, gain=64)
            
            # 等待相机数据流稳定
            obs_ready = False
            for _ in range(60):
                try:
                    obs = env.get_obs()
                    if 'camera_0' in obs:
                        obs_ready = True
                        break
                except: pass
                time.sleep(0.1)
            
            if not obs_ready:
                print(" [!] 错误：相机无法建立数据流。")
                return

            print('='*60)
            print('【操控与录制就绪】')
            print(f' - 映射向量: W/S={u_unit}, A/D={v_unit}')
            print(' - 操作说明:')
            print('   WASD: 移动机械臂')
            print('   Space: 切换夹爪开合')
            print('   C: [开始录制] | V: [保存并停止] | X: [舍弃录制]')
            print('   Q: 退出程序')
            print('='*60)

            # 获取初始位姿
            state = env.get_robot_state()
            target_pose = state['TargetTCPPose'].copy()

            win_name = 'Teleop_Control'
            cv2.namedWindow(win_name)

            t_start = time.monotonic()
            iter_idx = 0
            is_recording = False
            current_stage = 0 

            while True:
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - 0.01 
                t_command_target = t_cycle_end + dt

                # 获取最新观测
                obs = env.get_obs()
                if 'camera_0' in obs:
                    # 将 BGR 转为 RGB 显示 (取决于你的环境)
                    img = obs['camera_0'][-1][:,:,::-1].copy()
                    
                    # 绘制 UI 状态
                    status_text = "● RECORDING" if is_recording else "IDLE"
                    color = (0, 0, 255) if is_recording else (0, 255, 0)
                    cv2.putText(img, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(img, f"Stage: {current_stage}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    cv2.imshow(win_name, img)

                # 捕获键盘输入
                key = cv2.waitKey(1) & 0xFF
                u_in, v_in = 0.0, 0.0
                step = 0.08 / frequency 

                if key == ord('q'): 
                    break
                elif key == ord('c'):
                    if not is_recording:
                        env.start_episode(time.time())
                        is_recording = True
                        print(">> ● 开始录制 Episode")
                elif key == ord('v'):
                    if is_recording:
                        env.end_episode()
                        is_recording = False
                        print(">> ■ 录制已保存")
                elif key == ord('x'):
                    if is_recording:
                        env.drop_episode()
                        is_recording = False
                        print(">> × 录制已丢弃")
                elif key == ord(' '):
                    current_stage = 1 - current_stage
                    print(f">> 夹爪状态: {'闭合' if current_stage==1 else '开启'}")
                
                # 操控逻辑
                if key == ord('w'): u_in = step
                elif key == ord('s'): u_in = -step
                elif key == ord('a'): v_in = step
                elif key == ord('d'): v_in = -step

                # 应用映射
                if u_in != 0 or v_in != 0:
                    delta = u_in * u_unit + v_in * v_unit
                    target_pose[:3] += delta

                # 精确推送指令
                precise_wait(t_sample)
                env.exec_actions(
                    actions=[target_pose],
                    timestamps=[t_command_target - time.monotonic() + time.time()],
                    stages=[current_stage]
                )

                precise_wait(t_cycle_end)
                iter_idx += 1

    cv2.destroyAllWindows()
    print("[*] 程序已正常退出。")

if __name__ == '__main__':
    main()