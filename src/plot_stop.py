import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker

# ============================
# CONFIG
# ============================
LOG_PATH = "src/stop_sign.log"
OUT_PNG  = "stop_sign.svg"
UNIT     = "m/s"   # "m/s" or "km/h"

STOP_SIGN_AT_M = 100.0

# Plot only points where a [VERBAL] line occurred
PLOT_TOKENS = {"WARNING", "ACTUATE"}

# Marker styling
VERBAL_MARKER = "^"          # same shape for both
MARKER_EDGE_LW = 1.0
MARKER_SIZE_WARNING = 240
MARKER_SIZE_ACTUATE = 240
COLOR_WARNING = "#2ca02c"
COLOR_ACTUATE = "#ff7f0e"

# Deadline miss styling (make obvious)
DEADLINE_MARKER = "X"
DEADLINE_SIZE = 220
DEADLINE_EDGE_LW = 1.8
DEADLINE_COLOR = "#d62728"

# Time axis (below displacement)
TIME_AXIS_BELOW = True
TIME_AXIS_OUTWARD_PTS = 50       # pushes time axis down (separate line)
TIME_TICK_PAD = 6
TIME_MAX_TICKS = 9               # reduces clutter so 9/10 don't overlap

# ============================
# IDEAL SPEED + ENVELOPE
# ============================
BRAKE_START_RD = 25.0
V_CRUISE       = 10.0
ENV_W          = 2.5
ENVELOPE_COLOR = "#d62728"

# ============================
# REGEX patterns (your log format)
# ============================
re_dist     = re.compile(r"\[Relative DISTANCE\]:\s*([0-9]*\.?[0-9]+)")
re_speed    = re.compile(r"\[speed\]:\s*([0-9]*\.?[0-9]+)")
re_llm      = re.compile(r"\[LLM\]\s*([0-9]*\.?[0-9]+)\s*ms\s*->\s*([A-Z]+)\s*\|")
re_deadline = re.compile(r"\[DEADLINE\]\s*fallback\s*->\s*([A-Z]+)\s*\|")
re_verbal   = re.compile(r"\[VERBAL\]\s*([A-Z]+)\s*\|\s*(.*)$")
re_logical  = re.compile(r"^logical\s+([0-9]*\.?[0-9]+)")
re_physical = re.compile(r"^physical\s+([0-9]*\.?[0-9]+)")

# ============================
# Helpers
# ============================
def to_unit(v_mps: float) -> float:
    return v_mps * 3.6 if UNIT.lower() == "km/h" else v_mps

def unit_label() -> str:
    return "km/h" if UNIT.lower() == "km/h" else "m/s"

def ideal_speed(rd: float) -> float:
    if rd >= BRAKE_START_RD:
        return V_CRUISE
    if rd <= 0.0:
        return 0.0
    return V_CRUISE * math.sqrt(rd / BRAKE_START_RD)

def add_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    have_logical = df["logical_ms"].notna().mean() > 0.7
    if have_logical:
        df["time_s"] = df["logical_ms"] / 1000.0
    else:
        STEP_PERIOD_SEC = 0.1
        df["time_s"] = df["step"] * STEP_PERIOD_SEC
    return df

# ============================
# Parse log into rows
# ============================
rows_all = []
rows_verbal = []

cur = {
    "step": 0,
    "distance": None,
    "speed": None,
    "llm_ms": None,
    "llm_tok": None,
    "deadline_tok": None,
    "deadline_miss": False,
    "logical_ms": None,
    "physical_ms": None,
    "verbal": False,
    "verbal_tok": None,
    "verbal_msg": None,
}

def reset_cur_for_next():
    cur["step"] += 1
    cur["distance"] = None
    cur["speed"] = None
    cur["llm_ms"] = None
    cur["llm_tok"] = None
    cur["deadline_tok"] = None
    cur["deadline_miss"] = False
    cur["logical_ms"] = None
    cur["physical_ms"] = None
    cur["verbal"] = False
    cur["verbal_tok"] = None
    cur["verbal_msg"] = None

def flush_if_ready(require_decision=True):
    if cur["distance"] is None or cur["speed"] is None:
        return
    if require_decision and cur["llm_tok"] is None and cur["deadline_tok"] is None:
        return

    row = cur.copy()
    rows_all.append(row)
    if cur["verbal"]:
        rows_verbal.append(row)

    reset_cur_for_next()

with open(LOG_PATH, "r", errors="ignore") as f:
    for line in f:
        line = line.strip()

        m = re_dist.search(line)
        if m:
            if (
                cur["distance"] is not None
                and cur["speed"] is not None
                and cur["deadline_tok"] is not None
                and cur["llm_tok"] is None
            ):
                flush_if_ready()
            cur["distance"] = float(m.group(1))
            continue

        m = re_speed.search(line)
        if m:
            cur["speed"] = float(m.group(1))
            continue

        m = re_logical.search(line)
        if m:
            cur["logical_ms"] = float(m.group(1))
            continue

        m = re_physical.search(line)
        if m:
            cur["physical_ms"] = float(m.group(1))
            continue

        m = re_deadline.search(line)
        if m:
            cur["deadline_tok"] = m.group(1).upper()
            cur["deadline_miss"] = True
            continue

        m = re_llm.search(line)
        if m:
            cur["llm_ms"] = float(m.group(1))
            cur["llm_tok"] = m.group(2).upper()
            continue

        m = re_verbal.search(line)
        if m:
            cur["verbal"] = True
            cur["verbal_tok"] = m.group(1).upper().strip()
            cur["verbal_msg"] = (m.group(2) or "").strip()
            flush_if_ready()
            continue

# flush tail
if cur["distance"] is not None and cur["speed"] is not None and (cur["llm_tok"] is not None or cur["deadline_tok"] is not None):
    rows_all.append(cur.copy())
    if cur["verbal"]:
        rows_verbal.append(cur.copy())

df_all = pd.DataFrame(rows_all)
if df_all.empty:
    raise RuntimeError("Parsed 0 rows. Check LOG_PATH and regex patterns vs your log.")

df_verbal = pd.DataFrame(rows_verbal)

df_all = add_time_cols(df_all)
df_verbal = add_time_cols(df_verbal)

# units + displacement axis
df_all["speed_u"] = df_all["speed"].apply(to_unit)
df_all["x_m"] = (STOP_SIGN_AT_M - df_all["distance"]).clip(0.0, STOP_SIGN_AT_M)

if not df_verbal.empty:
    df_verbal["speed_u"] = df_verbal["speed"].apply(to_unit)
    df_verbal["x_m"] = (STOP_SIGN_AT_M - df_verbal["distance"]).clip(0.0, STOP_SIGN_AT_M)

# sort for interpolation stability
df_all.sort_values("x_m", inplace=True)
if not df_verbal.empty:
    df_verbal.sort_values("x_m", inplace=True)

# envelope grid in displacement coords
rd_grid = np.linspace(0.0, STOP_SIGN_AT_M, 900)
x_grid  = np.sort(STOP_SIGN_AT_M - rd_grid)
ideal_arr = np.array([ideal_speed(rd) for rd in (STOP_SIGN_AT_M - x_grid)])
upper_env = np.maximum(0.0, ideal_arr + ENV_W)
lower_env = np.maximum(0.0, ideal_arr - ENV_W)

# verbal points (only tokens you care about)
dfp = df_verbal.copy()
if not dfp.empty:
    dfp["tok"] = (
        dfp["verbal_tok"]
        .fillna(dfp["llm_tok"])
        .fillna(dfp["deadline_tok"])
        .fillna("NONE")
        .str.upper()
    )
    dfp = dfp[dfp["tok"].isin(PLOT_TOKENS)].copy()

# ============================
# Plot (ONE axes + secondary bottom time axis)
# ============================
fig, ax = plt.subplots(figsize=(18, 7))

# envelopes
ax.plot(x_grid, to_unit(upper_env), linewidth=2.0, color=ENVELOPE_COLOR, zorder=2, label="Upper envelope")
ax.plot(x_grid, to_unit(lower_env), linewidth=2.0, color=ENVELOPE_COLOR, linestyle="--", zorder=2, label="Lower envelope")

# speed trace
ax.plot(df_all["x_m"], df_all["speed_u"], marker=".", linewidth=1.2, color="#1f77b4", zorder=3, label=f"Speed ({unit_label()})")

# deadline misses
miss = df_all[df_all["deadline_miss"] == True]
if not miss.empty:
    ax.scatter(
        miss["x_m"], miss["speed_u"],
        marker=DEADLINE_MARKER, s=DEADLINE_SIZE,
        color=DEADLINE_COLOR, edgecolors="black", linewidths=DEADLINE_EDGE_LW,
        zorder=6, label="Deadline miss",
    )

# verbal markers only
if not dfp.empty:
    gw = dfp[dfp["tok"] == "WARNING"]
    ga = dfp[dfp["tok"] == "ACTUATE"]

    if not gw.empty:
        ax.scatter(
            gw["x_m"], gw["speed_u"],
            marker=VERBAL_MARKER, s=MARKER_SIZE_WARNING,
            color=COLOR_WARNING, edgecolors="black", linewidths=MARKER_EDGE_LW,
            zorder=5, label="VERBAL: WARNING",
        )

    if not ga.empty:
        ax.scatter(
            ga["x_m"], ga["speed_u"],
            marker=VERBAL_MARKER, s=MARKER_SIZE_ACTUATE,
            color=COLOR_ACTUATE, edgecolors="black", linewidths=MARKER_EDGE_LW,
            zorder=5, label="VERBAL: ACTUATE",
        )

# main axes labels
ax.set_xlabel("Displacement of car from stop sign (m)", fontsize=18, fontweight="bold", labelpad=10)
ax.set_ylabel(f"Speed ({unit_label()})", fontsize=18, fontweight="bold")
ax.tick_params(axis="both", labelsize=16)
ax.set_xlim(0.0, STOP_SIGN_AT_M)
ax.set_ylim(bottom=0.0)
ax.grid(False)

# ===== secondary time axis below (with its own axis line) =====
x_vals = df_all["x_m"].to_numpy()
t_vals = df_all["time_s"].to_numpy()

# ensure monotonic for interp
order = np.argsort(x_vals)
x_sorted = x_vals[order]
t_sorted = t_vals[order]

def x_to_time(x):
    return np.interp(x, x_sorted, t_sorted)

def time_to_x(t):
    # invert approximately by interpolating t->x using sorted t
    ord_t = np.argsort(t_sorted)
    t_s = t_sorted[ord_t]
    x_s = x_sorted[ord_t]
    return np.interp(t, t_s, x_s)

if TIME_AXIS_BELOW:
    secax = ax.secondary_xaxis("bottom", functions=(x_to_time, time_to_x))
    secax.spines["bottom"].set_position(("outward", TIME_AXIS_OUTWARD_PTS))
    secax.spines["bottom"].set_visible(True)
    secax.spines["bottom"].set_linewidth(1.2)

    secax.set_xlabel("Time (s)", fontsize=18, fontweight="bold", labelpad=18)
    secax.tick_params(axis="x", labelsize=14, pad=TIME_TICK_PAD)

    # >>> REPLACEMENT STARTS HERE <<<
    t_max = float(np.nanmax(t_sorted)) if len(t_sorted) else 0.0
    t_end = math.floor(t_max)
    secax.set_xticks(np.arange(0, t_end + 1, 1))
    secax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    # >>> REPLACEMENT ENDS HERE <<<

else:
    secax = ax.secondary_xaxis("top", functions=(x_to_time, time_to_x))
    secax.set_xlabel("Time (s)", fontsize=18, fontweight="bold")
    secax.tick_params(axis="x", labelsize=14, pad=8)

    t_max = float(np.nanmax(t_sorted)) if len(t_sorted) else 0.0
    t_end = math.floor(t_max)
    secax.set_xticks(np.arange(0, t_end + 1, 1))
    secax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))

ax.legend(loc="lower left", fontsize=14)

# important: leave room at bottom for the moved time axis
plt.subplots_adjust(bottom=0.22)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
plt.show()
print(f"\nSaved: {OUT_PNG}")