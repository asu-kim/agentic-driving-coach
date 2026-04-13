import matplotlib.pyplot as plt

# ============================
# CONFIG
# ============================
LOG_PATH = "src/HardwareIntegration/accelaration.log"
OUT_PEDAL_PNG = "accelerator_signal.png"
OUT_ACCEL_PNG = "estimated_acceleration.png"

# Assumed full pedal angle and max acceleration
MAX_PEDAL_ANGLE_DEG = 30.0
MAX_ACCEL_MPS2 = 4.0

# ============================
# READ VALUES
# ============================
values = []

with open(LOG_PATH, "r", errors="ignore") as f:
    for line in f:
        s = line.strip()
        if s and s != "None":
            try:
                values.append(float(s))
            except:
                pass

if not values:
    raise RuntimeError("No numeric pedal values found in log.")

# ============================
# PEDAL ANGLE + ACCELERATION
# ============================
pedal_angle_deg = [v * MAX_PEDAL_ANGLE_DEG for v in values]
accel_mps2 = [v * MAX_ACCEL_MPS2 for v in values]

# ============================
# PLOT 1: PEDAL SIGNAL
# ============================
plt.figure(figsize=(14, 5))
plt.plot(values, linewidth=2)
plt.xlabel("Sample index", fontsize=14, fontweight="bold")
plt.ylabel("Normalized pedal value", fontsize=14, fontweight="bold")
plt.title("Accelerator Pedal Signal", fontsize=15, fontweight="bold")
plt.ylim(0, 1.05)
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_PEDAL_PNG, dpi=300)
plt.show()

# ============================
# PLOT 2: PEDAL ANGLE
# ============================
plt.figure(figsize=(14, 5))
plt.plot(pedal_angle_deg, linewidth=2)
plt.xlabel("Sample index", fontsize=14, fontweight="bold")
plt.ylabel("Pedal angle (deg)", fontsize=14, fontweight="bold")
plt.title("Estimated Pedal Angle", fontsize=15, fontweight="bold")
plt.ylim(0, MAX_PEDAL_ANGLE_DEG * 1.05)
plt.grid(True)
plt.tight_layout()
plt.savefig("pedal_angle.png", dpi=300)
plt.show()

# ============================
# PLOT 3: ESTIMATED ACCELERATION
# ============================
plt.figure(figsize=(14, 5))
plt.plot(accel_mps2, linewidth=2)
plt.xlabel("Sample index", fontsize=14, fontweight="bold")
plt.ylabel("Estimated acceleration (m/s²)", fontsize=14, fontweight="bold")
plt.title("Estimated Longitudinal Acceleration from Pedal Input", fontsize=15, fontweight="bold")
plt.ylim(0, MAX_ACCEL_MPS2 * 1.05)
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_ACCEL_PNG, dpi=300)
plt.show()

print(f"Saved: {OUT_PEDAL_PNG}")
print("Saved: pedal_angle.png")
print(f"Saved: {OUT_ACCEL_PNG}")