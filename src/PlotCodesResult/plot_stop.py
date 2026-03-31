import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# CONFIG
# ============================
LOG_PATH = "/home/asurite.ad.asu.edu/dprahlad/agentic-driving-coach/src/logs/stop_sign_novice_1b.log"
OUT_SPEED_SVG = "/home/asurite.ad.asu.edu/dprahlad/agentic-driving-coach/src/plot2/stop_novice_1b.svg"
UNIT = "m/s"

STOP_SIGN_AT_M = 100.0
MAX_PLOT_TIME = 9.0

# Envelope speeds (far away)
UPPER_START = 12.0
LOWER_START = 8.0

# Braking starts when distance-to-stop <= this
BRAKE_START_RD = 25.0

ENVELOPE_COLOR = "#FB0000"

# Marker styling
VERBAL_MARKER = "^"
MARKER_SIZE = 4000
COLOR_WARNING = "#2ca02c"
COLOR_ACTUATE = "#ff7f0e"

# Deadline marker styling
DEADLINE_MARKER = "X"
DEADLINE_SIZE = 1500
DEADLINE_EDGE_LW = 2.5
DEADLINE_COLOR = "#d62728"

ALLOWED_TOKENS = {"WARNING", "ACTUATE"}

# ============================
# Regex
# ============================
re_dist     = re.compile(r"\[Relative DISTANCE\]:\s*([0-9]*\.?[0-9]+)")
re_speed    = re.compile(r"\[speed\]:\s*([0-9]*\.?[0-9]+)")
re_physical = re.compile(r"^physical\s+([0-9]*\.?[0-9]+)")

# LLM line: [LLM] 966.7 ms -> WARNING | message...
re_llm = re.compile(r"\[LLM\]\s*([0-9]*\.?[0-9]+)\s*ms\s*->\s*([A-Z]+)\s*\|\s*(.*)$")

# Deadline line: [DEADLINE] fallback -> WARNING | message...
re_deadline = re.compile(r"\[DEADLINE\]\s*fallback\s*->\s*([A-Z]+)\s*\|\s*(.*)$")

# Verbal line: [VERBAL] WARNING | message...
re_verbal = re.compile(r"\[VERBAL\]\s*([A-Z]+)\s*\|\s*(.*)$")

# ============================
# Helpers
# ============================
def to_unit(v_mps):
    return v_mps * 3.6 if UNIT.lower() == "km/h" else v_mps

def unit_label():
    return "km/h" if UNIT.lower() == "km/h" else "m/s"

def brake_curve(start_speed, rd_arr):
    """Flat at start_speed when rd >= BRAKE_START_RD, then sqrt curve down to 0 at rd=0."""
    out = np.empty_like(rd_arr, dtype=float)
    mask_far = rd_arr >= BRAKE_START_RD
    out[mask_far] = start_speed
    out[~mask_far] = start_speed * np.sqrt(np.clip(rd_arr[~mask_far], 0, BRAKE_START_RD) / BRAKE_START_RD)
    return out

# ============================
# Parse log
# ============================
rows = []
cur = {}

def flush_cur():
    if "distance" in cur:
        rows.append(cur.copy())

with open(LOG_PATH, "r", errors="ignore") as f:
    for raw in f:
        line = raw.strip()

        m = re_dist.search(line)
        if m:
            flush_cur()
            cur = {
                "distance": float(m.group(1)),
                "speed": None,
                "physical_ms": None,

                # LLM decision
                "llm_ms": None,
                "llm_tok": None,
                "llm_msg": None,

                # deadline decision
                "has_deadline": False,
                "deadline_tok": None,
                "deadline_msg": None,

                # verbal decision
                "has_verbal": False,
                "verbal_tok": None,
                "verbal_msg": None,
            }
            continue

        m = re_speed.search(line)
        if m:
            cur["speed"] = float(m.group(1))
            continue

        m = re_physical.search(line)
        if m:
            cur["physical_ms"] = float(m.group(1))
            continue

        m = re_llm.search(line)
        if m:
            cur["llm_ms"] = float(m.group(1))
            cur["llm_tok"] = (m.group(2) or "").strip().upper()
            cur["llm_msg"] = (m.group(3) or "").strip().rstrip("]")
            continue

        m = re_deadline.search(line)
        if m:
            cur["has_deadline"] = True
            cur["deadline_tok"] = (m.group(1) or "").strip().upper()
            cur["deadline_msg"] = (m.group(2) or "").strip().rstrip("]")
            continue

        m = re_verbal.search(line)
        if m:
            cur["has_verbal"] = True
            cur["verbal_tok"] = (m.group(1) or "").strip().upper()
            cur["verbal_msg"] = (m.group(2) or "").strip().rstrip("]")
            continue

# flush last
flush_cur()

df_all = pd.DataFrame(rows)
if df_all.empty:
    raise RuntimeError("Parsed 0 rows. Check LOG_PATH and regex patterns.")

# Keep rows that have speed
df_all = df_all[df_all["speed"].notna()].copy()
df_all["physical_ms"] = pd.to_numeric(df_all["physical_ms"], errors="coerce")

# Time normalize (physical)
df_all["time_s"] = (df_all["physical_ms"].interpolate().ffill().bfill() / 1000.0)
df_all["time_s"] -= df_all["time_s"].min()

# Coordinates
df_all["x_m"] = (STOP_SIGN_AT_M - df_all["distance"]).clip(0, STOP_SIGN_AT_M)
df_all["speed_u"] = df_all["speed"].apply(to_unit)
df_all.to_csv("CSV/stop_novice_1b_data_all.csv", index=False)

# Marker datasets
df_verbal = df_all[(df_all["has_verbal"] == True) & (df_all["time_s"] <= MAX_PLOT_TIME)].copy()
df_verbal = df_verbal[df_verbal["verbal_tok"].isin(ALLOWED_TOKENS)].copy()
df_verbal.to_csv("CSV/stop_novice_1b_data_verbal.csv", index=False)

df_deadline = df_all[(df_all["has_deadline"] == True) & (df_all["time_s"] <= MAX_PLOT_TIME)].copy()
df_deadline = df_deadline[df_deadline["deadline_tok"].isin(ALLOWED_TOKENS)].copy()
df_deadline.to_csv("CSV/stop_novice_1b_data_deadline.csv", index=False)

# ============================
# Plot
# ============================
fig, ax = plt.subplots(figsize=(30, 15))

# Envelopes
rd_grid = np.linspace(0, STOP_SIGN_AT_M, 1000)   # rd: distance-to-stop (0..100)
x_grid  = STOP_SIGN_AT_M - rd_grid               # x: displacement from start (0..100)

upper = to_unit(brake_curve(UPPER_START, rd_grid))
lower = to_unit(brake_curve(LOWER_START, rd_grid))

ax.plot(x_grid, upper, color=ENVELOPE_COLOR, label="Upper Envelope", alpha=0.8, linewidth=4.0)
ax.plot(x_grid, lower, color=ENVELOPE_COLOR, ls="--", label="Lower Envelope", alpha=0.8, linewidth=4.0)

# Speed trace
ax.plot(df_all["x_m"], df_all["speed_u"], color="#1f77b4", lw=5.0, label="Measured Speed", zorder=3)

# VERBAL markers
for tok, col in [("WARNING", COLOR_WARNING), ("ACTUATE", COLOR_ACTUATE)]:
    sub = df_verbal[df_verbal["verbal_tok"] == tok]
    if not sub.empty:
        ax.scatter(
            sub["x_m"], sub["speed_u"],
            marker=VERBAL_MARKER, s=MARKER_SIZE,
            color=col, edgecolors="black", zorder=6, label=f"Verbal {tok}"
        )

# DEADLINE markers
if not df_deadline.empty:
    ax.scatter(
        df_deadline["x_m"], df_deadline["speed_u"],
        marker=DEADLINE_MARKER, s=DEADLINE_SIZE,
        color=DEADLINE_COLOR, edgecolors="black", linewidths=DEADLINE_EDGE_LW,
        zorder=7, label="Deadline fallback"
    )

# Physical time axis (secondary)
time_ticks = np.arange(0, int(MAX_PLOT_TIME) + 1)
tick_pos_x = np.interp(time_ticks, df_all["time_s"], df_all["x_m"])

valid_ticks = []
valid_pos = []
last_x = -999.0
for t, x in zip(time_ticks, tick_pos_x):
    if abs(x - last_x) >= 4.0:  # Prevent ticks from getting too close horizontally
        valid_ticks.append(t)
        valid_pos.append(x)
        last_x = x

secax = ax.secondary_xaxis("bottom")
secax.spines["bottom"].set_position(("outward", 250))
secax.set_xticks(valid_pos)
secax.set_xticklabels([f"{t}s" for t in valid_ticks])
secax.set_xlabel("Time (s)", fontsize=75, fontweight="bold", labelpad=40 )

secax.tick_params(axis='x', labelsize=75, length=12, width=3, pad=18)
for label in secax.get_xticklabels():
    label.set_fontweight('bold')
for spine in secax.spines.values():
    spine.set_linewidth(1.5)

# Axis formatting
ax.set_xlim(0, 100)
ax.set_ylim(0, 14)
ax.set_xlabel("Displacement (m)", fontsize=75, fontweight="bold", labelpad=30)
ax.set_ylabel(f"Velocity ({unit_label()})", fontsize=75, fontweight="bold", labelpad=30)

ax.tick_params(axis='both', labelsize=85, length=12, width=3, pad=18)
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# ax.legend(loc="upper right", fontsize=12, framealpha=0.95)

plt.tight_layout()
plt.savefig(OUT_SPEED_SVG, dpi=300)
plt.show()

print(f"Saved: {OUT_SPEED_SVG}")