import matplotlib.pyplot as plt

# ============================
# CONFIG
# ============================
LOG_PATH = "src/HardwareIntegration/brake.log"   # <-- your brake log
OUT_BRAKE_PNG = "brake_signal.png"
OUT_BRAKE_ACCEL_PNG = "brake_deceleration.png"

MAX_PEDAL_ANGLE_DEG = 30.0
MAX_BRAKE_DECEL = 9.0   # m/s² (typical strong braking)

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
    raise RuntimeError("No numeric brake values found.")

# ============================
# COMPUTE ANGLE + DECELERATION
# ============================
brake_angle_deg = [v * MAX_PEDAL_ANGLE_DEG for v in values]

# negative because braking slows down
decel_mps2 = [-v * MAX_BRAKE_DECEL for v in values]

# ============================
# PLOT 1: BRAKE SIGNAL
# ============================
plt.figure(figsize=(14, 5))
plt.plot(values, linewidth=2)
plt.xlabel("Sample index", fontsize=14, fontweight="bold")
plt.ylabel("Normalized brake value", fontsize=14, fontweight="bold")
plt.title("Brake Pedal Signal", fontsize=15, fontweight="bold")
plt.ylim(0, 1.05)
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_BRAKE_PNG, dpi=300)
plt.show()

# ============================
# PLOT 2: BRAKE ANGLE
# ============================
plt.figure(figsize=(14, 5))
plt.plot(brake_angle_deg, linewidth=2)
plt.xlabel("Sample index", fontsize=14, fontweight="bold")
plt.ylabel("Brake pedal angle (deg)", fontsize=14, fontweight="bold")
plt.title("Estimated Brake Pedal Angle", fontsize=15, fontweight="bold")
plt.ylim(0, MAX_PEDAL_ANGLE_DEG * 1.05)
plt.grid(True)
plt.tight_layout()
plt.savefig("brake_angle.png", dpi=300)
plt.show()

# ============================
# PLOT 3: DECELERATION
# ============================
plt.figure(figsize=(14, 5))
plt.plot(decel_mps2, linewidth=2)
plt.xlabel("Sample index", fontsize=14, fontweight="bold")
plt.ylabel("Deceleration (m/s²)", fontsize=14, fontweight="bold")
plt.title("Estimated Braking Deceleration", fontsize=15, fontweight="bold")
plt.ylim(-MAX_BRAKE_DECEL * 1.05, 0)
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_BRAKE_ACCEL_PNG, dpi=300)
plt.show()

print(f"Saved: {OUT_BRAKE_PNG}")
print("Saved: brake_angle.png")
print(f"Saved: {OUT_BRAKE_ACCEL_PNG}")