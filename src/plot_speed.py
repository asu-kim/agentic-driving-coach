import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# CONFIG
# ============================
LOG_PATH = "src/speed_change.log"
OUT_SPEED_SVG = "speed_change.svg"
UNIT = "m/s"

EVENT_AT_M = 100.0
MAX_PLOT_TIME = 9.0

ENVELOPE_COLOR = "#FB0000"

# Marker styling
VERBAL_MARKER = "^"
MARKER_SIZE = 240
COLOR_WARNING = "#2ca02c"
COLOR_ACTUATE = "#ff7f0e"

# Deadline marker styling
DEADLINE_MARKER = "X"
DEADLINE_SIZE = 220
DEADLINE_EDGE_LW = 1.8
DEADLINE_COLOR = "#d62728"

ALLOWED_TOKENS = {"WARNING", "ACTUATE"}

# ============================
# Regex
# ============================
re_dist     = re.compile(r"\[Relative DISTANCE\]:\s*([0-9]*\.?[0-9]+)")
re_speed    = re.compile(r"\[speed\]:\s*([0-9]*\.?[0-9]+)")
re_physical = re.compile(r"^physical\s+([0-9]*\.?[0-9]+)")

re_llm = re.compile(r"\[LLM\]\s*([0-9]*\.?[0-9]+)\s*ms\s*->\s*([A-Z]+)\s*\|\s*(.*)$")
re_deadline = re.compile(r"\[DEADLINE\]\s*fallback\s*->\s*([A-Z]+)\s*\|\s*(.*)$")
re_verbal = re.compile(r"\[VERBAL\]\s*([A-Z]+)\s*\|\s*(.*)$")

# ============================
# Helpers
# ============================
def to_unit(v_mps):
    return v_mps * 3.6 if UNIT.lower() == "km/h" else v_mps

def unit_label():
    return "km/h" if UNIT.lower() == "km/h" else "m/s"

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

flush_cur()

df_all = pd.DataFrame(rows)
if df_all.empty:
    raise RuntimeError("Parsed 0 rows. Check LOG_PATH and regex patterns.")

# Keep rows that have speed
df_all = df_all[df_all["speed"].notna()].copy()

# Ensure numeric time
df_all["physical_ms"] = pd.to_numeric(df_all["physical_ms"], errors="coerce")

# If there are missing physical_ms, interpolate safely
df_all["physical_ms"] = df_all["physical_ms"].interpolate(limit_direction="both")

# Drop anything still missing
df_all = df_all[df_all["physical_ms"].notna()].copy()

# Normalize time
df_all["time_s"] = df_all["physical_ms"] / 1000.0
df_all["time_s"] -= df_all["time_s"].min()

# Coordinates
df_all["x_m"] = (EVENT_AT_M - df_all["distance"]).clip(0, EVENT_AT_M)
df_all["speed_u"] = df_all["speed"].apply(to_unit)

# IMPORTANT: for interpolation, make time strictly increasing
# sort by time, then drop duplicate time samples (keep last)
df_all = df_all.sort_values("time_s").drop_duplicates(subset=["time_s"], keep="last").reset_index(drop=True)

# Marker datasets
df_verbal = df_all[(df_all["has_verbal"] == True) & (df_all["time_s"] <= MAX_PLOT_TIME)].copy()
df_verbal = df_verbal[df_verbal["verbal_tok"].isin(ALLOWED_TOKENS)].copy()

df_deadline = df_all[(df_all["has_deadline"] == True) & (df_all["time_s"] <= MAX_PLOT_TIME)].copy()
df_deadline = df_deadline[df_deadline["deadline_tok"].isin(ALLOWED_TOKENS)].copy()

# ============================
# SPEED CHANGE ENVELOPE (18->11 and 16->11 bands)
# ============================
FAR_LOW = 16.0
FAR_HIGH = 18.0
NEAR_LOW = 11.0
NEAR_HIGH = 12.0

DECEL_START_RD = 80.0
DECEL_END_RD = 25.0

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)

def speed_change_curve(start, end, rd):
    rd = np.asarray(rd, dtype=float)

    # piecewise with vectorization (no loop)
    out = np.empty_like(rd)

    far = rd >= DECEL_START_RD
    near = rd <= DECEL_END_RD
    mid = (~far) & (~near)

    out[far] = start
    out[near] = end

    t = (DECEL_START_RD - rd[mid]) / (DECEL_START_RD - DECEL_END_RD)
    t = smoothstep(t)
    out[mid] = start + (end - start) * t
    return out

rd_grid = np.linspace(0, EVENT_AT_M, 1000)   # distance-to-event
x_grid  = EVENT_AT_M - rd_grid               # displacement

upper = to_unit(speed_change_curve(FAR_HIGH, NEAR_HIGH, rd_grid))
lower = to_unit(speed_change_curve(FAR_LOW,  NEAR_LOW,  rd_grid))

# ============================
# Plot
# ============================
fig, ax = plt.subplots(figsize=(18, 9))

ax.plot(x_grid, upper, color=ENVELOPE_COLOR, label="Upper Envelope", alpha=0.8, linewidth=2.2)
ax.plot(x_grid, lower, color=ENVELOPE_COLOR, ls="--", label="Lower Envelope", alpha=0.8, linewidth=2.2)

ax.plot(df_all["x_m"], df_all["speed_u"], color="#1f77b4", lw=2.8, label="Measured Speed", zorder=3)

# VERBAL markers
for tok, col in [("WARNING", COLOR_WARNING), ("ACTUATE", COLOR_ACTUATE)]:
    sub = df_verbal[df_verbal["verbal_tok"] == tok]
    if not sub.empty:
        ax.scatter(sub["x_m"], sub["speed_u"],
                   marker=VERBAL_MARKER, s=MARKER_SIZE,
                   color=col, edgecolors="black", zorder=6, label=f"Verbal {tok}")

# DEADLINE markers
if not df_deadline.empty:
    ax.scatter(df_deadline["x_m"], df_deadline["speed_u"],
               marker=DEADLINE_MARKER, s=DEADLINE_SIZE,
               color=DEADLINE_COLOR, edgecolors="black", linewidths=DEADLINE_EDGE_LW,
               zorder=7, label="Deadline fallback")

# ===== Secondary physical time axis (safe) =====
time_ticks = np.arange(0, int(MAX_PLOT_TIME) + 1, 1)

# clamp ticks to available time range to avoid NaN tick positions
tmin, tmax = float(df_all["time_s"].min()), float(df_all["time_s"].max())
time_ticks = time_ticks[(time_ticks >= tmin) & (time_ticks <= tmax)]

tick_pos_x = np.interp(time_ticks, df_all["time_s"].to_numpy(), df_all["x_m"].to_numpy())

secax = ax.secondary_xaxis("bottom")
secax.spines["bottom"].set_position(("outward", 65))
secax.set_xticks(tick_pos_x)
secax.set_xticklabels([f"{int(t)}s" for t in time_ticks])
secax.set_xlabel("Physical Time (s)", fontsize=18, fontweight="bold", labelpad=15)

secax.tick_params(axis='x', labelsize=18, length=8, width=2)
for label in secax.get_xticklabels():
    label.set_fontweight('bold')
secax.minorticks_off()  # ✅ avoids minor-tick NaN warnings
for spine in secax.spines.values():
    spine.set_linewidth(1.5)

# Axis formatting
ax.set_xlim(0, 100)
ax.set_ylim(10, 19)
ax.set_xlabel("Displacement from start (m)", fontsize=18, fontweight="bold")
ax.set_ylabel(f"Speed ({unit_label()})", fontsize=18, fontweight="bold")

ax.tick_params(axis='both', labelsize=18, length=8, width=2)
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

ax.minorticks_off()  #

plt.tight_layout()
plt.savefig(OUT_SPEED_SVG, dpi=300)
plt.show()
print(f"Saved: {OUT_SPEED_SVG}")