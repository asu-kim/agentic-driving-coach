import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================
# CONFIG
# ============================
LOG_PATH = "src/stop_sign.log"
BEHAVIOR_PATH = "src/DriverBehaviorStop.txt"
OUT_PNG = "stop_sign.svg"

UNIT = "m/s"  # "m/s" or "km/h"
USE_DEADLINE_TOKEN_AS_EFFECTIVE = False  # if True, token=deadline token on miss
ALLOWED_TOKENS = {"NONE", "NOTIFY", "WARNING", "ACTUATE"}

# ============================
# SPEED BUFFER / ALLOWED BANDS
# ============================
# (rd_min, rd_max, v_min, v_max, label)
SPEED_BANDS = [
    (98.0, None, 8.0, 13.0, "Allowed (rd>=98): 3–13"),
    (50.0, 60.0, 5.0, 9.0,  "Allowed (50<rd<=60): 5–9"),
    (0.0, 25.0, 0.0, 2.5,   "Allowed (rd<=25): 0–2.5"),
]

ACTUATION_THRESHOLD_M = 25.0

# ============================
# REGEX patterns
# ============================
re_dist = re.compile(r"\[Relative DISTANCE\]:\s*([0-9]*\.?[0-9]+)")
re_speed = re.compile(r"\[speed\]:\s*([0-9]*\.?[0-9]+)")
re_llm = re.compile(r"\[LLM\]\s*([0-9]*\.?[0-9]+)\s*ms\s*->\s*([A-Z]+)\s*\|")
re_deadline = re.compile(r"\[DEADLINE\]\s*fallback\s*->\s*([A-Z]+)\s*\|")

# ============================
# HELPERS
# ============================
def to_unit(v_mps: float) -> float:
    return v_mps * 3.6 if UNIT.lower() == "km/h" else v_mps

def unit_label() -> str:
    return "km/h" if UNIT.lower() == "km/h" else "m/s"

def parse_behavior_file(path: str):
    vals = []
    try:
        with open(path, "r", errors="ignore") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    vals.append(int(ln))
                except Exception:
                    pass
    except FileNotFoundError:
        return None
    return vals if vals else None

def band_mask(x, rd_min, rd_max):
    """Mask of x values that fall into the distance band."""
    if rd_max is None:
        return x >= rd_min
    # middle rule is 50 < rd <= 60 (exclude 50)
    if rd_min == 50.0 and rd_max == 60.0:
        return (x > rd_min) & (x <= rd_max)
    return (x >= rd_min) & (x <= rd_max)

def speed_band_limits(rd):
    """
    Return (vmin, vmax) based on SPEED_BANDS for this rd.
    If rd is not in any band, returns (None, None).
    """
    for (rd_min, rd_max, vmin, vmax, _lab) in SPEED_BANDS:
        if rd_max is None:
            if rd >= rd_min:
                return vmin, vmax
        else:
            if rd_min == 50.0 and rd_max == 60.0:
                if (rd > rd_min) and (rd <= rd_max):
                    return vmin, vmax
            else:
                if (rd >= rd_min) and (rd <= rd_max):
                    return vmin, vmax
    return None, None

# ============================
# PARSE LOG INTO DATAFRAME
# ============================
rows = []
cur = {
    "step": 0,
    "distance": None,
    "speed": None,
    "llm_ms": None,
    "llm_tok": None,
    "deadline_tok": None,
    "deadline_miss": False,
}

def flush_if_ready():
    # push row if we have distance+speed and any decision signal
    if cur["distance"] is None or cur["speed"] is None:
        return
    if cur["llm_tok"] is None and cur["deadline_tok"] is None:
        return

    rows.append(cur.copy())

    # reset per-step fields
    cur["step"] += 1
    cur["distance"] = None
    cur["speed"] = None
    cur["llm_ms"] = None
    cur["llm_tok"] = None
    cur["deadline_tok"] = None
    cur["deadline_miss"] = False

with open(LOG_PATH, "r", errors="ignore") as f:
    for line in f:
        line = line.strip()

        # NEW: if a new step begins (new distance) but previous step had deadline decision only,
        # flush that row so ACTUATE/others from deadline are not dropped.
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

        m = re_deadline.search(line)
        if m:
            cur["deadline_tok"] = m.group(1).upper()
            cur["deadline_miss"] = True
            continue

        m = re_llm.search(line)
        if m:
            cur["llm_ms"] = float(m.group(1))
            cur["llm_tok"] = m.group(2).upper()
            flush_if_ready()
            continue

# flush last partial step if possible
flush_if_ready()

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError("Parsed 0 rows. Check LOG_PATH and regex patterns vs your log.")

# Effective token
if USE_DEADLINE_TOKEN_AS_EFFECTIVE:
    df["tok"] = np.where(df["deadline_miss"], df["deadline_tok"], df["llm_tok"])
else:
    df["tok"] = df["llm_tok"].fillna(df["deadline_tok"])

df["tok"] = df["tok"].fillna("NONE").str.upper()
df = df[df["tok"].isin(ALLOWED_TOKENS)].copy()

# Units
df["speed_u"] = df["speed"].apply(to_unit)

# Sort far->near (100 -> 0) for plotting, but we will invert x-axis to read 100 on left
df.sort_values("distance", ascending=False, inplace=True)

# Attach behavior (best effort)
beh = parse_behavior_file(BEHAVIOR_PATH)
df["behavior"] = np.nan
if beh is not None:
    df["behavior"] = df["step"].apply(lambda i: beh[i] if i < len(beh) else np.nan)

print(df.head(10))
print("\nCounts:\n", df["tok"].value_counts())
print("\nDeadline misses:", int(df["deadline_miss"].sum()))
print("\nBehavior rows parsed:", 0 if beh is None else len(beh))

# ============================
# IDEAL DECELERATION LINE (piecewise inside buffer)
# ============================
# We create a piecewise ideal line that always stays within the buffer bands.
# Strategy:
# - At rd>=98: aim mid of [8,13] -> 10.5
# - Through general region (60..25): linearly ramp down toward near-stop band
# - Within rd<=25: ramp to 0 by rd=0, staying <=2.5
def ideal_speed_mps(rd: float) -> float:
    # Pick a target inside the bands when they exist
    vmin, vmax = speed_band_limits(rd)
    if vmin is not None:
        return 0.5 * (vmin + vmax)

    # Otherwise, interpolate between anchor points chosen to match your buffers
    # Anchors (rd, v): (98, 10.5), (60, 7.0), (25, 1.25), (0, 0)
    anchors = [(98.0, 10.5), (60.0, 7.0), (25.0, 1.25), (0.0, 0.0)]
    # clamp rd to [0,98] for interpolation
    r = max(0.0, min(98.0, rd))
    for (r0, v0), (r1, v1) in zip(anchors[:-1], anchors[1:]):
        if r <= r0 and r >= r1:
            # linear interpolation in rd
            t = (r0 - r) / (r0 - r1 + 1e-9)
            return v0 + t * (v1 - v0)
    return 0.0

# build ideal arrays on a grid
x_min = float(df["distance"].min())
x_max = float(df["distance"].max())
x_grid = np.linspace(x_min, x_max, 900)

ideal = np.array([ideal_speed_mps(x) for x in x_grid])

# envelope around ideal (like your red curves), clipped to nonnegative
# "buffer envelope" width: choose a fixed margin, but don't violate physical min 0
ENV_W = 2.5  # adjust if you want wider/narrower red bands
upper_env = np.maximum(0.0, ideal + ENV_W)
lower_env = np.maximum(0.0, ideal - ENV_W)

# ============================
# PLOT
# ============================
fig, ax = plt.subplots(figsize=(14, 7))

# Allowed bands
for (rd_min, rd_max, vmin, vmax, lab) in SPEED_BANDS:
    mask = band_mask(x_grid, rd_min, rd_max)
    if not np.any(mask):
        continue
    ax.fill_between(
        x_grid[mask],
        to_unit(vmin),
        to_unit(vmax),
        alpha=0.18,
        label=f"{lab} ({unit_label()})",
        zorder=1,
    )

# Speed trace
ax.plot(
    df["distance"],
    df["speed_u"],
    marker=".",
    linewidth=1.2,
    label=f"Speed ({unit_label()})",
    zorder=3,
)

# Ideal + envelopes
ax.plot(
    x_grid,
    to_unit(ideal),
    linestyle="--",
    linewidth=2.0,
    label="Ideal speed (within allowed buffer)",
    zorder=2,
)
ax.plot(
    x_grid,
    to_unit(upper_env),
    linewidth=2.0,
    label="Upper envelope (buffer limit)",
    zorder=2,
)
ax.plot(
    x_grid,
    to_unit(lower_env),
    linewidth=2.0,
    label="Lower envelope (buffer limit)",
    zorder=2,
)

# Actuation threshold line
ax.axvline(
    ACTUATION_THRESHOLD_M,
    linestyle=":",
    linewidth=2.0,
    label=f"Actuation threshold ({ACTUATION_THRESHOLD_M:g}m)",
    zorder=2,
)

# Tokens
marker_map = {"NONE": "o", "NOTIFY": "s", "WARNING": "^", "ACTUATE": "D"}
for tok, g in df.groupby("tok"):
    ax.scatter(
        g["distance"],
        g["speed_u"],
        marker=marker_map.get(tok, "o"),
        s=55,
        label=f"LLM: {tok}",
        zorder=4,
    )

# Deadline misses
miss = df[df["deadline_miss"]]
if not miss.empty:
    ax.scatter(
        miss["distance"],
        miss["speed_u"],
        marker="x",
        s=85,
        label="Deadline miss",
        zorder=5,
    )

ax.set_xlabel("Relative distance (m)")
ax.set_ylabel(f"Speed ({unit_label()})")
ax.grid(True)

# show far->near (100 on left, 0 on right)
ax.invert_xaxis()

# Behavior on secondary axis
ax2 = ax.twinx()
beh_g = df.dropna(subset=["behavior"])
if not beh_g.empty:
    ax2.plot(
        beh_g["distance"],
        beh_g["behavior"],
        linestyle="None",
        marker="|",
        markersize=14,
        label="Behavior (DriverBehavior.txt)",
        zorder=2,
    )
ax2.set_ylabel("Behavior code")

# Combine legends and force token entries even if absent (optional)
legend_force = [
    Line2D([0], [0], marker='o', linestyle='None', label='LLM: NONE'),
    Line2D([0], [0], marker='s', linestyle='None', label='LLM: NOTIFY'),
    Line2D([0], [0], marker='^', linestyle='None', label='LLM: WARNING'),
    Line2D([0], [0], marker='D', linestyle='None', label='LLM: ACTUATE'),
    Line2D([0], [0], marker='x', linestyle='None', label='Deadline miss'),
]

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()

all_h = h1 + h2 + legend_force
all_l = l1 + l2 + [h.get_label() for h in legend_force]

seen = set()
uniq_h, uniq_l = [], []
for h, l in zip(all_h, all_l):
    if l not in seen:
        uniq_h.append(h)
        uniq_l.append(l)
        seen.add(l)

ax.legend(uniq_h, uniq_l, loc="upper right")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
plt.show()

print(f"\nSaved: {OUT_PNG}")