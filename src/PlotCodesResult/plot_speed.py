import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# CONFIG
# ============================
LOG_PATH = "/home/asurite.ad.asu.edu/dprahlad/agentic-driving-coach/src/logs/speed_change_novice_8b.log"
OUT_SPEED_SVG = "/home/asurite.ad.asu.edu/dprahlad/agentic-driving-coach/src/plot2/speed_change_novice_8b.svg"

UNIT = "m/s"
EVENT_AT_M = 100.0
MAX_PLOT_TIME = 9.0

# Envelope
ENVELOPE_COLOR = "#FB0000"
FAR_LOW, FAR_HIGH = 16.0, 18.0
NEAR_LOW, NEAR_HIGH = 8.0, 12.0
DECEL_START_RD = 19.0
DECEL_END_RD = 0.0

# Marker styling
VERBAL_MARKER = "^"
MARKER_SIZE = 800
COLOR_WARNING = "#2ca02c"
COLOR_ACTUATE = "#ff7f0e"

DEADLINE_MARKER = "X"
DEADLINE_SIZE = 300
DEADLINE_EDGE_LW = 1.8
DEADLINE_COLOR = "#d62728"

ALLOWED_TOKENS = {"WARNING", "ACTUATE"}

# Plot styling
FIGSIZE = (18, 8)
AXIS_LABEL_SIZE = 30
TICK_LABEL_SIZE = 30
LINE_WIDTH = 2.8
ENVELOPE_WIDTH = 2.2
SPINE_WIDTH = 1.5
TIME_AXIS_OFFSET = 65

# ============================
# REGEX
# ============================
re_dist = re.compile(r"\[Relative DISTANCE\]:\s*([0-9]*\.?[0-9]+)")
re_speed = re.compile(r"\[speed\]:\s*([0-9]*\.?[0-9]+)")
re_physical = re.compile(r"^physical\s+([0-9]*\.?[0-9]+)")
re_llm = re.compile(r"\[LLM\]\s*([0-9]*\.?[0-9]+)\s*ms\s*->\s*([A-Z]+)\s*\|\s*(.*)$")
re_deadline = re.compile(r"\[DEADLINE\]\s*fallback\s*->\s*([A-Z]+)\s*\|\s*(.*)$")
re_verbal = re.compile(r"\[VERBAL\]\s*([A-Z]+)\s*\|\s*(.*)$")

# ============================
# HELPERS
# ============================
def to_unit(v_mps: float) -> float:
    return v_mps * 3.6 if UNIT.lower() == "km/h" else v_mps

def unit_label() -> str:
    return "km/h" if UNIT.lower() == "km/h" else "m/s"

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def speed_change_curve(start: float, end: float, rd):
    """
    Smooth transition from 'start' to 'end' between:
      rd >= DECEL_START_RD -> start
      rd <= DECEL_END_RD   -> end
    """
    rd = np.asarray(rd, dtype=float)
    out = np.empty_like(rd)

    far_mask = rd >= DECEL_START_RD
    near_mask = rd <= DECEL_END_RD
    mid_mask = ~(far_mask | near_mask)

    out[far_mask] = start
    out[near_mask] = end

    t = (DECEL_START_RD - rd[mid_mask]) / (DECEL_START_RD - DECEL_END_RD)
    out[mid_mask] = start + (end - start) * smoothstep(t)
    return out

def new_block(distance: float):
    return {
        "distance": distance,
        "speed": None,
        "physical_ms": None,
        "llm_ms": None,
        "llm_tok": None,
        "llm_msg": None,
        "has_deadline": False,
        "deadline_tok": None,
        "deadline_msg": None,
        "has_verbal": False,
        "verbal_tok": None,
        "verbal_msg": None,
    }

# ============================
# PARSE LOG
# ============================
rows = []
cur = None

with open(LOG_PATH, "r", errors="ignore") as f:
    for raw in f:
        line = raw.strip()

        m = re_dist.search(line)
        if m:
            if cur is not None and cur["speed"] is not None:
                rows.append(cur.copy())
            cur = new_block(float(m.group(1)))
            continue

        if cur is None:
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

if cur is not None and cur["speed"] is not None:
    rows.append(cur.copy())

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError("Parsed 0 rows. Check LOG_PATH and regex patterns.")

# ============================
# CLEAN DATA
# ============================
df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
df["physical_ms"] = pd.to_numeric(df["physical_ms"], errors="coerce")

df = df[df["speed"].notna()].copy()
df["physical_ms"] = df["physical_ms"].interpolate(limit_direction="both")
df = df[df["physical_ms"].notna()].copy()

# Normalize physical time
df["time_s"] = df["physical_ms"] / 1000.0
df["time_s"] -= df["time_s"].min()

# Coordinates
df["x_m"] = (EVENT_AT_M - df["distance"]).clip(0.0, EVENT_AT_M)
df["speed_u"] = df["speed"].apply(to_unit)

# Sort by time and remove duplicate timestamps for safe interpolation
df = df.sort_values("time_s").drop_duplicates(subset=["time_s"], keep="last").reset_index(drop=True)

# Marker subsets
df_verbal = df[(df["has_verbal"] == True) & (df["time_s"] <= MAX_PLOT_TIME)].copy()
df_verbal = df_verbal[df_verbal["verbal_tok"].isin(ALLOWED_TOKENS)].copy()

df_deadline = df[(df["has_deadline"] == True) & (df["time_s"] <= MAX_PLOT_TIME)].copy()
df_deadline = df_deadline[df_deadline["deadline_tok"].isin(ALLOWED_TOKENS)].copy()

# ============================
# ENVELOPE
# ============================
rd_grid = np.linspace(0.0, EVENT_AT_M, 1000)
x_grid = EVENT_AT_M - rd_grid

upper = to_unit(speed_change_curve(FAR_HIGH, NEAR_HIGH, rd_grid))
lower = to_unit(speed_change_curve(FAR_LOW, NEAR_LOW, rd_grid))

# ============================
# PLOT
# ============================
fig, ax = plt.subplots(figsize=FIGSIZE)

# Envelope
ax.plot(
    x_grid, upper,
    color=ENVELOPE_COLOR,
    linewidth=ENVELOPE_WIDTH,
    alpha=0.8,
    label="Upper Envelope",
    zorder=2
)
ax.plot(
    x_grid, lower,
    color=ENVELOPE_COLOR,
    linewidth=ENVELOPE_WIDTH,
    linestyle="--",
    alpha=0.8,
    label="Lower Envelope",
    zorder=2
)

# Measured speed
ax.plot(
    df["x_m"], df["speed_u"],
    color="#1f77b4",
    linewidth=LINE_WIDTH,
    label="Measured Speed",
    zorder=3
)

# Verbal markers
for tok, col in [("WARNING", COLOR_WARNING), ("ACTUATE", COLOR_ACTUATE)]:
    sub = df_verbal[df_verbal["verbal_tok"] == tok]
    if not sub.empty:
        ax.scatter(
            sub["x_m"], sub["speed_u"],
            marker=VERBAL_MARKER,
            s=MARKER_SIZE,
            color=col,
            edgecolors="black",
            zorder=6,
            label=f"Verbal {tok}"
        )

# Deadline markers
if not df_deadline.empty:
    ax.scatter(
        df_deadline["x_m"], df_deadline["speed_u"],
        marker=DEADLINE_MARKER,
        s=DEADLINE_SIZE,
        color=DEADLINE_COLOR,
        edgecolors="black",
        linewidths=DEADLINE_EDGE_LW,
        zorder=7,
        label="Deadline fallback"
    )

# ============================
# AXES FORMATTING
# ============================
ax.set_xlim(0, EVENT_AT_M)
ax.set_ylim(10, 19)

ax.set_xlabel("Displacement (m)", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
ax.set_ylabel(f"Velocity ({unit_label()})", fontsize=AXIS_LABEL_SIZE, fontweight="bold")

ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, length=8, width=2)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")

for spine in ax.spines.values():
    spine.set_linewidth(SPINE_WIDTH)

ax.minorticks_off()
ax.grid(False)

# ============================
# SECONDARY PHYSICAL TIME AXIS
# ============================
time_ticks = np.arange(0, int(MAX_PLOT_TIME) + 1, 1)

tmin = float(df["time_s"].min())
tmax = float(df["time_s"].max())
time_ticks = time_ticks[(time_ticks >= tmin) & (time_ticks <= tmax)]

if len(time_ticks) > 0:
    tick_pos_x = np.interp(
        time_ticks,
        df["time_s"].to_numpy(),
        df["x_m"].to_numpy()
    )

    secax = ax.secondary_xaxis("bottom")
    secax.spines["bottom"].set_position(("outward", TIME_AXIS_OFFSET))
    secax.set_xticks(tick_pos_x)
    secax.set_xticklabels([f"{int(t)}s" for t in time_ticks])

    secax.set_xlabel("Time (s)", fontsize=AXIS_LABEL_SIZE, fontweight="bold", labelpad=25)
    secax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE, length=8, width=2)

    for label in secax.get_xticklabels():
        label.set_fontweight("bold")

    for spine in secax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)

    secax.minorticks_off()

plt.tight_layout()
plt.savefig(OUT_SPEED_SVG, dpi=300)
plt.show()

print(f"Saved: {OUT_SPEED_SVG}")