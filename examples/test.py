import spiceypy as spice
import os

# === 设置 kernel 路径 ===
base_path = "/mnt/d/Github_Code/basilisk/basilisk/dist3/Basilisk/supportData/EphemerisData"
leap_seconds_kernel = os.path.join(base_path, "naif0012.tls")   # 或其他 .tls 文件
ephemeris_kernel = os.path.join(base_path, "de430.bsp")          # 星历文件

# === 加载 SPICE 内核 ===
if spice.ktotal("ALL") == 0:  # 避免重复加载
    spice.furnsh(leap_seconds_kernel)
    spice.furnsh(ephemeris_kernel)

# === 使用 SPICE 查询地球相对于太阳的位置 ===
et = spice.str2et("2023 JAN 1")
pos, _ = spice.spkezr("EARTH", et, "J2000", "NONE", "SUN")

# === 输出结果 ===
print("ET:", et)
print("Position of EARTH from SUN [km]:")
print(f"X: {pos[0]:.2f} km, Y: {pos[1]:.2f} km, Z: {pos[2]:.2f} km")

# === 卸载内核（可选）===
spice.kclear()
