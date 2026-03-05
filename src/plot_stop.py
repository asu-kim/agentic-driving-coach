import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# CONFIG
# ============================
LOG_PATH = "src/stop_sign.log"
OUT_SPEED_SVG = "stop_sign.svg"
UNIT = "m/s"   

STOP_SIGN_AT_M = 100.0
MAX_PLOT_TIME = 9.0  

# Start at 12 and 8 (Center 10.0 +/- 2.0)
V_CRUISE = 10.0
ENV_W = 2.0 
BRAKE_START_RD = 25.0
ENVELOPE_COLOR = "#FB0000" # Professional Grey

# Marker styling
VERBAL_MARKER = "^"          
MARKER_SIZE = 240
COLOR_WARNING = "#2ca02c" # Green
COLOR_ACTUATE = "#ff7f0e" # Orange

# ============================
# Regex & Helpers
# ============================
re_dist = re.compile(r"\[Relative DISTANCE\]:\s*([0-9]*\.?[0-9]+)")
re_speed = re.compile(r"\[speed\]:\s*([0-9]*\.?[0-9]+)")
re_physical = re.compile(r"^physical\s+([0-9]*\.?[0-9]+)")
re_decision = re.compile(r"(WARNING|ACTUATE)")

def to_unit(v_mps): return v_mps * 3.6 if UNIT.lower() == "km/h" else v_mps
def unit_label(): return "km/h" if UNIT.lower() == "km/h" else "m/s"

def smoothstep01(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)

BRAKE_BLEND_RD = 12.0  # meters of smoothing zone ABOVE BRAKE_START_RD (tune 8..20)

def ideal_speed(rd):
    """
    Smooth center-line:
    - braking curve for rd <= BRAKE_START_RD
    - smoothly blends to V_CRUISE over [BRAKE_START_RD, BRAKE_START_RD + BRAKE_BLEND_RD]
    - flat V_CRUISE beyond that
    """
    if rd <= 0.0:
        return 0.0

    # raw braking curve (sqrt)
    v_brake = V_CRUISE * math.sqrt(max(0.0, rd) / BRAKE_START_RD)

    # far away: cruise
    if rd >= BRAKE_START_RD + BRAKE_BLEND_RD:
        return V_CRUISE

    # near: pure braking curve
    if rd <= BRAKE_START_RD:
        return min(V_CRUISE, v_brake)

    # blend zone
    t = (rd - BRAKE_START_RD) / BRAKE_BLEND_RD   # 0..1
    w = smoothstep01(t)                          # smooth easing
    return (1.0 - w) * min(V_CRUISE, v_brake) + w * V_CRUISE

# ============================
# Parse Log
# ============================
rows = []
cur = {}
with open(LOG_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if m := re_dist.search(line):
            if "distance" in cur: rows.append(cur.copy())
            cur = {"distance": float(m.group(1)), "verbal_tok": None, "has_verbal": False}
        elif m := re_speed.search(line): cur["speed"] = float(m.group(1))
        elif m := re_physical.search(line): cur["physical_ms"] = float(m.group(1))
        
        if "[VERBAL]" in line:
            cur["has_verbal"] = True
            if m := re_decision.search(line): cur["verbal_tok"] = m.group(1)

if "distance" in cur: rows.append(cur)

df_all = pd.DataFrame(rows)
# Time Normalization
df_all["time_s"] = (df_all["physical_ms"].interpolate().ffill().bfill() / 1000.0)
df_all["time_s"] -= df_all["time_s"].min()
# Coordinate Conversion
df_all["x_m"] = (STOP_SIGN_AT_M - df_all["distance"]).clip(0, STOP_SIGN_AT_M)
df_all["speed_u"] = df_all["speed"].apply(to_unit)

# Filter for the 9s markers
df_markers = df_all[(df_all["has_verbal"] == True) & (df_all["time_s"] <= MAX_PLOT_TIME)].copy()

# ============================
# Plotting
# ============================
fig, ax = plt.subplots(figsize=(18, 9))

# Tapered Envelope Math
rd_grid = np.linspace(0, STOP_SIGN_AT_M, 1000)
x_grid = STOP_SIGN_AT_M - rd_grid
ideal_mps = np.array([ideal_speed(rd) for rd in rd_grid])

# Tapering: scales the allowed deviation from ENV_W (at 0m) down to 0 (at 100m)
# ----------------------------
# Smooth envelopes
# ----------------------------

# === Smooth braking envelopes (flat far away, parabolic/√ drop near stop) ===
UPPER_START = 12.0   # upper envelope when far (x near 0)
LOWER_START = 8.0    # lower envelope when far (x near 0)
BRAKE_START_RD = 25.0  # start braking when distance-to-stop <= 25m

rd_grid = np.linspace(0, STOP_SIGN_AT_M, 1000)   # rd: distance-to-stop (0..100)
x_grid  = STOP_SIGN_AT_M - rd_grid               # x: displacement from start (0..100)

def brake_curve(start_speed, rd):
    # flat at start_speed until braking starts, then √ curve to 0 at rd=0
    out = np.empty_like(rd, dtype=float)
    mask_far = rd >= BRAKE_START_RD
    out[mask_far] = start_speed
    out[~mask_far] = start_speed * np.sqrt(np.clip(rd[~mask_far], 0, BRAKE_START_RD) / BRAKE_START_RD)
    return out

upper = to_unit(brake_curve(UPPER_START, rd_grid))
lower = to_unit(brake_curve(LOWER_START, rd_grid))

# Plot Envelopes
ax.plot(x_grid, upper, color=ENVELOPE_COLOR, label="Upper Envelope", alpha=0.7, linewidth=1.8)
ax.plot(x_grid, lower, color=ENVELOPE_COLOR, ls="--", label="Lower Envelope", alpha=0.7, linewidth=1.8)

# Plot Speed Trace
ax.plot(df_all["x_m"], df_all["speed_u"], color="#1f77b4", lw=2.5, label="Measured Speed", zorder=3)

# Plot Verbal Markers
if not df_markers.empty:
    for tok, col in [("WARNING", COLOR_WARNING), ("ACTUATE", COLOR_ACTUATE)]:
        sub = df_markers[df_markers["verbal_tok"] == tok]
        ax.scatter(sub["x_m"], sub["speed_u"], marker=VERBAL_MARKER, s=MARKER_SIZE, 
                   color=col, edgecolors="black", zorder=5, label=f"Verbal {tok}")

# Physical Time Axis
time_ticks = np.arange(0, int(MAX_PLOT_TIME) + 1)
tick_pos_x = np.interp(time_ticks, df_all["time_s"], df_all["x_m"])
secax = ax.secondary_xaxis("bottom")
secax.spines["bottom"].set_position(("outward", 65))
secax.set_xticks(tick_pos_x)
secax.set_xticklabels([f"{t}s" for t in time_ticks])
secax.set_xlabel("Physical Time (s)", fontsize=16, fontweight="bold", labelpad=15)

# Axis Labels & Formatting
ax.set_xlim(0, 100)
ax.set_ylim(0, 14)
ax.set_xlabel("Displacement from start (m)", fontsize=16, fontweight="bold")
ax.set_ylabel(f"Speed ({unit_label()})", fontsize=16, fontweight="bold")
ax.tick_params(axis='both', labelsize=14)
ax.legend(loc="upper right", fontsize=12, framealpha=0.9)

plt.tight_layout()
plt.savefig(OUT_SPEED_SVG, dpi=300)
plt.show()