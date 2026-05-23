"""
3D Mouse 诊断脚本（自动探测字段名）
"""
import time
import pyspacemouse

print("正在打开 3D Mouse...")
device = pyspacemouse.open()
print(f"设备: {device.product_name}  ({device.vendor_name})")

# 先读一帧，打印完整字段
print("\n读取一帧，探测字段名...")
for _ in range(50):
    s = device.read()
    if s:
        print("SpaceMouseState 字段:", [f for f in dir(s) if not f.startswith("_")])
        print("完整内容:", s)
        break
    time.sleep(0.02)

print("\n移动 3D Mouse，观察输出（Ctrl+C 退出）：")
print("-" * 60)

try:
    while True:
        s = device.read()
        if s:
            # 打印所有数值字段
            fields = {f: getattr(s, f) for f in dir(s)
                      if not f.startswith("_") and isinstance(getattr(s, f), (int, float))}
            if any(abs(v) > 0.01 for v in fields.values()):
                print(fields)
        time.sleep(0.02)
except KeyboardInterrupt:
    pass
finally:
    device.close()
    print("\n已关闭")
