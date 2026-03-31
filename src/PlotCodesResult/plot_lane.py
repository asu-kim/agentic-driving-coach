import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================
# CONFIG
# ============================
LOG_PATH = "/home/asurite.ad.asu.edu/dprahlad/agentic-driving-coach/src/logs/lane_change_novice_70b.log"
OUT_PNG  = "/home/asurite.ad.asu.edu/dprahlad/agentic-driving-coach/src/plot2/lane_change_novice_70b.svg"

UNIT = "m/s"  # "m/s" or "km/h"

EVENT_AT_M = 100.0
MAX_PLOT_TIME = 9.0          # seconds shown on the time axis
TIME_SOURCE = "physical"     # "physical" or "logical"

# Ideal lane-change speed band (constant)
IDEAL_V = 18.5               # m/s
ENV_W   = 1.5                # +/- envelope half-width (m/s)

ENVELOPE_COLOR = "#d62728"

# tokens
ALLOWED_TOKENS = {"NONE", "NOTIFY", "WARNING", "ACTUATE"}
PLOT_TOKENS    = {"NOTIFY", "WARNING", "ACTUATE"}  # markers

# marker styling
TOKEN_MARKER = {"NOTIFY": "s", "WARNING": "^", "ACTUATE": "^"}
TOKEN_COLOR  = {"NOTIFY": "#1f77b4", "WARNING": "#2ca02c", "ACTUATE": "#ff7f0e"}

DEADLINE_MARKER   = "X"
DEADLINE_SIZE     = 300
DEADLINE_EDGE_LW  = 1.8
DEADLINE_COLOR    = "#d62728"

# ============================
# REGEX patterns (match your log)
# ============================
re_dist     = re.compile(r"\[Relative DISTANCE\]:\s*([0-9]*\.?[0-9]+)")
re_speed    = re.compile(r"\[speed\]:\s*([0-9]*\.?[0-9]+)")
re_logical  = re.compile(r"^logical\s+([0-9]*\.?[0-9]+)")
re_physical = re.compile(r"^physical\s+([0-9]*\.?[0-9]+)")

# [LLM] 1811.1 ms -> NONE |
re_llm      = re.compile(r"\[LLM\]\s*([0-9]*\.?[0-9]+)\s*ms\s*->\s*([A-Z]+)\s*\|")

# [DEADLINE] fallback -> WARNING | message...
re_deadline = re.compile(r"\[DEADLINE\]\s*fallback\s*->\s*([A-Z]+)\s*\|\s*(.*)$")

# [VERBAL] ACTUATE | message...
re_verbal   = re.compile(r"\[VERBAL\]\s*([A-Z]+)\s*\|\s*(.*)$")

# ============================
# Helpers
# ============================
def to_unit(v_mps: float) -> float:
    return v_mps * 3.6 if UNIT.lower() == "km/h" else v_mps

def unit_label() -> str:
    return "km/h" if UNIT.lower() == "km/h" else "m/s"

# ============================
# Parse log into per-block rows
# We finalize ("flush") when a NEW [Relative DISTANCE] starts.
# ============================
rows = []
cur = None

def start_new_block(dist_val: float):
    return {
        "distance": dist_val,
        "speed": None,
        "logical_ms": None,
        "physical_ms": None,

        "llm_ms": None,
        "llm_tok": None,

        "has_deadline": False,
        "deadline_tok": None,
        "deadline_msg": None,

        "has_verbal": False,
        "verbal_tok": None,
        "verbal_msg": None,
    }

def flush_block():
    nonlocal_cur = cur  # just for clarity
    if nonlocal_cur is None:
        return
    # keep only blocks that have distance + speed
    if nonlocal_cur["distance"] is None or nonlocal_cur["speed"] is None:
        return
    rows.append(nonlocal_cur.copy())

with open(LOG_PATH, "r", errors="ignore") as f:
    for raw in f:
        line = raw.strip()

        m = re_dist.search(line)
        if m:
            # new block starts -> flush previous
            if cur is not None:
                flush_block()
            cur = start_new_block(float(m.group(1)))
            continue

        if cur is None:
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
            cur["has_deadline"] = True
            cur["deadline_tok"] = (m.group(1) or "").strip().upper()
            cur["deadline_msg"] = (m.group(2) or "").strip().rstrip("]")
            continue

        m = re_llm.search(line)
        if m:
            cur["llm_ms"] = float(m.group(1))
            cur["llm_tok"] = (m.group(2) or "").strip().upper()
            continue

        m = re_verbal.search(line)
        if m:
            cur["has_verbal"] = True
            cur["verbal_tok"] = (m.group(1) or "").strip().upper()
            cur["verbal_msg"] = (m.group(2) or "").strip().rstrip("]")
            continue

# flush last
if cur is not None:
    flush_block()

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError("Parsed 0 rows. Check LOG_PATH and regex patterns.")

# ============================
# Clean + derive columns
# ============================
df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
df = df[df["speed"].notna()].copy()

# time source
if TIME_SOURCE == "physical":
    t_ms = pd.to_numeric(df["physical_ms"], errors="coerce")
else:
    t_ms = pd.to_numeric(df["logical_ms"], errors="coerce")

# normalize time (seconds) robustly
df["time_s"] = (t_ms.interpolate().ffill().bfill() / 1000.0)
df["time_s"] -= float(df["time_s"].min())

# displacement axis (0..100): x = EVENT_AT_M - distance_to_event
df["x_m"] = (EVENT_AT_M - df["distance"]).clip(0.0, EVENT_AT_M)
df["speed_u"] = df["speed"].apply(to_unit)

# Effective token (what "acted"):
# Prefer deadline token when deadline fired; else LLM token; else NONE
df["tok_eff"] = np.where(df["has_deadline"], df["deadline_tok"], df["llm_tok"])
df["tok_eff"] = df["tok_eff"].fillna("NONE").str.upper()
df = df[df["tok_eff"].isin(ALLOWED_TOKENS)].copy()

# Verbal token for markers
df["tok_verbal"] = df["verbal_tok"].fillna("").str.upper()

# deadline misses mask (for X markers)
df["deadline_miss"] = df["has_deadline"] == True

# sort by displacement increasing (nice lines)
df.sort_values("x_m", inplace=True)

# ============================
# Envelope grid (constant band)
# ============================
x_grid = np.linspace(0.0, EVENT_AT_M, 900)
upper_env = np.full_like(x_grid, IDEAL_V + ENV_W, dtype=float)
lower_env = np.full_like(x_grid, IDEAL_V - ENV_W, dtype=float)

# ============================
# Plot
# ============================
fig, ax = plt.subplots(figsize=(18, 7))

# envelopes
ax.plot(x_grid, to_unit(upper_env), linewidth=2.2, color=ENVELOPE_COLOR, alpha=0.85, label="Upper envelope", zorder=2)
ax.plot(x_grid, to_unit(lower_env), linewidth=2.2, color=ENVELOPE_COLOR, alpha=0.85, linestyle="--", label="Lower envelope", zorder=2)

# speed trace
ax.plot(df["x_m"], df["speed_u"], color="#1f77b4", lw=2.8, label=f"Measured speed ({unit_label()})", zorder=3)

# deadline misses
miss = df[df["deadline_miss"]]
# Speed trace (lower zorder)
ax.plot(df["x_m"], df["speed_u"],
        color="#1f77b4", lw=2.8, zorder=3, label="Measured Speed")

# Verbal markers (middle zorder)
# ax.scatter(..., zorder=10, label="VERBAL: WARNING")
# ax.scatter(..., zorder=10, label="VERBAL: ACTUATE")

# Deadline misses (TOP zorder)  
miss = df[df["deadline_miss"] == True]
if not miss.empty:
    ax.scatter(
        miss["x_m"], miss["speed_u"],
        marker="X", s=DEADLINE_SIZE,
        color=DEADLINE_COLOR,
        edgecolors="black", linewidths=DEADLINE_EDGE_LW,
        zorder=50,   # <<< highest
        label="Deadline miss"
    )

# verbal markers (only when [VERBAL] exists)
df_verbal = df[df["has_verbal"] == True].copy()
df_verbal = df_verbal[df_verbal["tok_verbal"].isin(PLOT_TOKENS)]

for tok in ["NOTIFY", "WARNING", "ACTUATE"]:
    sub = df_verbal[df_verbal["tok_verbal"] == tok]
    if sub.empty:
        continue
    ax.scatter(
        sub["x_m"], sub["speed_u"],
        marker=TOKEN_MARKER[tok],
        s=800,
        color=TOKEN_COLOR[tok],
        edgecolors="black",
        linewidths=0.9,
        zorder=7,
        label=f"VERBAL: {tok}",
    )

# axes formatting
ax.set_xlim(0, EVENT_AT_M)
ax.set_ylim(16, 22)
ax.set_xlabel("Displacement (m)", fontsize=28, fontweight="bold")
ax.set_ylabel(f"Velocity ({unit_label()})", fontsize=28, fontweight="bold")
ax.tick_params(axis="both", labelsize=28, length=8, width=2)
for lab in ax.get_xticklabels() + ax.get_yticklabels():
    lab.set_fontweight("bold")
for sp in ax.spines.values():
    sp.set_linewidth(1.5)

ax.grid(False)

# ============================
# Secondary time axis (bottom)
# ============================
# Need monotonic arrays for interp; x_m is sorted already
x_vals = df["x_m"].to_numpy()
t_vals = df["time_s"].to_numpy()

# guard against empty
if len(x_vals) >= 2 and np.isfinite(t_vals).any():
    time_ticks = np.arange(0, int(MAX_PLOT_TIME) + 1, 1)
    # only ticks within data time range
    t_max = float(np.nanmax(t_vals))
    time_ticks = time_ticks[time_ticks <= t_max + 1e-9]

    tick_pos_x = np.interp(time_ticks, t_vals, x_vals)

    secax = ax.secondary_xaxis("bottom")
    secax.spines["bottom"].set_position(("outward", 65))
    secax.set_xticks(tick_pos_x)
    secax.set_xticklabels([f"{int(t)}s" for t in time_ticks])
    secax.set_xlabel(f"{TIME_SOURCE.capitalize()} Time (s)", fontsize=28, fontweight="bold", labelpad=15)
    secax.tick_params(axis="x", labelsize=28, length=8, width=2)
    for lab in secax.get_xticklabels():
        lab.set_fontweight("bold")
    for sp in secax.spines.values():
        sp.set_linewidth(1.5)

# Legend (force entries even if absent)
legend_force = [
    Line2D([0], [0], marker='s', linestyle='None', markerfacecolor=TOKEN_COLOR["NOTIFY"],
           markeredgecolor='black', label='VERBAL: NOTIFY'),
    Line2D([0], [0], marker='^', linestyle='None', markerfacecolor=TOKEN_COLOR["WARNING"],
           markeredgecolor='black', label='VERBAL: WARNING'),
    Line2D([0], [0], marker='D', linestyle='None', markerfacecolor=TOKEN_COLOR["ACTUATE"],
           markeredgecolor='black', label='VERBAL: ACTUATE'),
    Line2D([0], [0], marker='X', linestyle='None', markerfacecolor=DEADLINE_COLOR,
           markeredgecolor='black', label='Deadline fallback'),
]

h1, l1 = ax.get_legend_handles_labels()
all_h = h1 + legend_force
all_l = l1 + [h.get_label() for h in legend_force]

seen = set()
uniq_h, uniq_l = [], []
for h, l in zip(all_h, all_l):
    if l not in seen:
        uniq_h.append(h)
        uniq_l.append(l)
        seen.add(l)

# ax.legend(uniq_h, uniq_l, loc="lower left", fontsize=14)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=250)
plt.show()
print(f"Saved: {OUT_PNG}")

# fig, ax = plt.subplots(figsize=(14, 2))

# # Hide axes
# ax.axis("off")

# # Draw legend only (horizontal)
# # ax.legend(
# #     uniq_h,
# #     uniq_l,
# #     loc="center",
# #     ncol=len(uniq_h),   # makes it horizontal
# #     fontsize=14,
# #     frameon=False
# # )

# plt.tight_layout()
# plt.savefig("lane_change_legend.svg", dpi=250)
# plt.show()