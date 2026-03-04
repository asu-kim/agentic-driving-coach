import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ============================
# CONFIG (SPEED CHANGE)
# ============================
LOG_PATH = "src/speed_change.log"     # <-- change if needed
OUT_PNG  = "speed_change.svg"         # <-- output file
UNIT     = "m/s"                      # "m/s" or "km/h"

# "Event location" in meters (same role as STOP_SIGN_AT_M = 100 in stop_sign)
# If your speed change scenario is also 100m from start, keep 100.
EVENT_AT_M = 100.0

# Plot only points where [VERBAL] occurred, and only these tokens:
PLOT_TOKENS = {"WARNING", "ACTUATE"}

# Marker styling (same as your stop_sign style)
VERBAL_MARKER = "^"          # same shape for both WARNING + ACTUATE
MARKER_EDGE_LW = 1.0
MARKER_SIZE_WARNING = 240
MARKER_SIZE_ACTUATE = 240
COLOR_WARNING = "#2ca02c"
COLOR_ACTUATE = "#ff7f0e"

# Deadline miss styling (obvious)
DEADLINE_MARKER = "X"
DEADLINE_SIZE = 220
DEADLINE_EDGE_LW = 1.8
DEADLINE_COLOR = "#d62728"

# Time axis BELOW displacement (uniform integer ticks 0,1,2,...)
TIME_AXIS_BELOW = True
TIME_AXIS_OUTWARD_PTS = 50    # pushes the time axis down (separate line)
TIME_TICK_PAD = 6

# ============================
# OPTIONAL: IDEAL SPEED + ENVELOPE (same as stop_sign)
# If you do NOT want envelopes for speed-change, set SHOW_ENVELOPE=False
# ============================
SHOW_ENVELOPE = True
ENVELOPE_COLOR = "#d62728"

# thresholds from your rules (all in m/s)
FAR_RD     = 80.0
WARN_RD_HI = 60.0
WARN_RD_LO = 50.0
NEAR_RD    = 25.0

FAR_LOW,  FAR_HIGH  = 16.0, 18.0
WARN_LOW, WARN_HIGH = 13.0, 15.0
NEAR_LOW, NEAR_HIGH = 11.0, 12.0

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

def lerp(a, b, t):
    return a + (b - a) * t

def envelope_band(rd: float):
    """
    Returns (low, high) allowed speed band at relative distance rd (meters).
    Piecewise + linear ramps between regions to look 'decelerating'.
    """
    # rd is distance-to-event/sign (your logged "Relative DISTANCE")

    # Far: rd >= 80 => [16,18]
    if rd >= FAR_RD:
        return FAR_LOW, FAR_HIGH

    # Ramp FAR -> WARN_HIGH as rd goes 80 -> 60
    if WARN_RD_HI < rd < FAR_RD:
        t = (FAR_RD - rd) / (FAR_RD - WARN_RD_HI)  # 0 at 80, 1 at 60
        low  = lerp(FAR_LOW,  WARN_LOW,  t)
        high = lerp(FAR_HIGH, WARN_HIGH, t)
        return low, high

    # Warning window: 60..50 => [13,15]
    if WARN_RD_LO <= rd <= WARN_RD_HI:
        return WARN_LOW, WARN_HIGH

    # Ramp WARN_LOW -> NEAR as rd goes 50 -> 25
    if NEAR_RD < rd < WARN_RD_LO:
        t = (WARN_RD_LO - rd) / (WARN_RD_LO - NEAR_RD)  # 0 at 50, 1 at 25
        low  = lerp(WARN_LOW,  NEAR_LOW,  t)
        high = lerp(WARN_HIGH, NEAR_HIGH, t)
        return low, high

    # Near: rd <= 25 => [11,12]
    return NEAR_LOW, NEAR_HIGH

def add_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Use logical time if present, else fall back to step*0.1s."""
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
# We flush a "row" when [VERBAL] occurs (so we can plot only those),
# but we also keep df_all to draw the speed trace + deadline misses.
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
            # if deadline already happened and we're starting a new block before llm line
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

        # IMPORTANT: we flush on VERBAL so points correspond exactly
        m = re_verbal.search(line)
        if m:
            cur["verbal"] = True
            cur["verbal_tok"] = m.group(1).upper().strip()
            cur["verbal_msg"] = (m.group(2) or "").strip()
            flush_if_ready()
            continue

# flush tail (rarely needed)
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
df_all["x_m"] = (EVENT_AT_M - df_all["distance"]).clip(0.0, EVENT_AT_M)

if not df_verbal.empty:
    df_verbal["speed_u"] = df_verbal["speed"].apply(to_unit)
    df_verbal["x_m"] = (EVENT_AT_M - df_verbal["distance"]).clip(0.0, EVENT_AT_M)

# sort for interpolation stability
df_all.sort_values("x_m", inplace=True)
if not df_verbal.empty:
    df_verbal.sort_values("x_m", inplace=True)

# envelope grid in displacement coords
# ============================
# Envelope grid (decelerating, based on your RULES)
# ============================
if SHOW_ENVELOPE:
    rd_grid = np.linspace(0.0, EVENT_AT_M, 900)          # rd: 0..EVENT_AT_M
    x_grid  = EVENT_AT_M - rd_grid                       # displacement
    x_grid  = np.sort(x_grid)                            # 0..EVENT_AT_M
    rd_for_x = EVENT_AT_M - x_grid                        # convert back

    low_arr  = np.array([envelope_band(rd)[0] for rd in rd_for_x], dtype=float)
    high_arr = np.array([envelope_band(rd)[1] for rd in rd_for_x], dtype=float)

    upper_env = high_arr
    lower_env = low_arr

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

# envelopes (optional)
if SHOW_ENVELOPE:
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

# main axes labels (same style)
ax.set_xlabel("Displacement of car from speed change point (m)", fontsize=18, fontweight="bold", labelpad=10)
ax.set_ylabel(f"Speed ({unit_label()})", fontsize=18, fontweight="bold")
ax.tick_params(axis="both", labelsize=16)
ax.set_xlim(0.0, EVENT_AT_M)
ax.set_ylim(bottom=0.0)
ax.grid(False)

# ===== secondary time axis below (uniform 0,1,2,3...) =====
x_vals = df_all["x_m"].to_numpy()
t_vals = df_all["time_s"].to_numpy()

order = np.argsort(x_vals)
x_sorted = x_vals[order]
t_sorted = t_vals[order]

def x_to_time(x):
    return np.interp(x, x_sorted, t_sorted)

def time_to_x(t):
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

    # FORCE integer ticks 0,1,2,... (no 0,1.3,3,...)
    t_max = float(np.nanmax(t_sorted)) if len(t_sorted) else 0.0
    t_end = math.floor(t_max)
    secax.set_xticks(np.arange(0, t_end + 1, 1))
    secax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
else:
    secax = ax.secondary_xaxis("top", functions=(x_to_time, time_to_x))
    secax.set_xlabel("Time (s)", fontsize=18, fontweight="bold")
    secax.tick_params(axis="x", labelsize=14, pad=8)

    t_max = float(np.nanmax(t_sorted)) if len(t_sorted) else 0.0
    t_end = math.floor(t_max)
    secax.set_xticks(np.arange(0, t_end + 1, 1))
    secax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))

ax.legend(loc="lower left", fontsize=14)

# leave room for the moved time axis
plt.subplots_adjust(bottom=0.22)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
plt.show()
print(f"\nSaved: {OUT_PNG}")