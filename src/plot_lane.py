import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================
# CONFIG
# ============================
LOG_PATH = "src/lane_change.log"                # <-- your lane-change runtime log
BEHAVIOR_PATH = "src/DriverBehaviorLane.txt"    # <-- accel,head,steer per line (e.g., 4,9,14)
OUT_PNG = "lane_change.svg"

UNIT = "m/s"  # or "km/h"
USE_DEADLINE_TOKEN_AS_EFFECTIVE = False  # if True, effective token = deadline token when miss

# Only what you want
ALLOWED_TOKENS = {"NONE", "NOTIFY", "WARNING", "ACTUATE"}

# ============================
# ALLOWED SPEED BANDS (EDIT)
# ============================
# Format: (rd_min, rd_max, v_min, v_max, label)
# rd is "distance_to_maneuver" i.e. [Relative DISTANCE]
SPEED_BANDS = [
    (80.0, None, 17.0, 20.0, "Allowed (rd>=80): 15–25"),
    (50.0, 60.0, 17.0, 20.0, "Allowed (50<rd<=60): 15–25"),
    (0.0, 25.0, 17.0, 20.0, "Allowed (rd<=25): 15–25"),
]

# ============================
# REGEX patterns
# ============================
re_dist = re.compile(r"\[Relative DISTANCE\]:\s*([0-9]*\.?[0-9]+)")
re_speed = re.compile(r"\[speed\]:\s*([0-9]*\.?[0-9]+)")
re_llm = re.compile(r"\[LLM\]\s*([0-9]*\.?[0-9]+)\s*ms\s*->\s*([A-Z]+)\s*\|")
re_deadline = re.compile(r"\[DEADLINE\]\s*fallback\s*->\s*([A-Z]+)\s*\|")

# ============================
# Helpers
# ============================
def to_unit(v_mps: float) -> float:
    return v_mps * 3.6 if UNIT.lower() == "km/h" else v_mps

def unit_label() -> str:
    return "km/h" if UNIT.lower() == "km/h" else "m/s"

def parse_lane_behavior(path: str):
    """
    DriverBehaviorLane.txt lines: accel,head,steer
    Example: 4,9,14
    Returns dict of lists: accel[], head[], steer[]
    """
    accel, head, steer = [], [], []
    try:
        with open(path, "r", errors="ignore") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                parts = [p.strip() for p in ln.split(",")]
                if len(parts) != 3:
                    continue
                try:
                    a = int(parts[0]); h = int(parts[1]); s = int(parts[2])
                    accel.append(a); head.append(h); steer.append(s)
                except Exception:
                    pass
    except FileNotFoundError:
        return None
    if not accel:
        return None
    return {"accel": accel, "head": head, "steer": steer}

# ============================
# Parse log into dataframe
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
    if cur["distance"] is None or cur["speed"] is None:
        return
    if cur["llm_tok"] is None and cur["deadline_tok"] is None:
        return

    rows.append(cur.copy())

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

        m = re_dist.search(line)
        if m:
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

flush_if_ready()

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError("Parsed 0 rows. Check LOG_PATH and patterns vs your log output.")

# Effective token
if USE_DEADLINE_TOKEN_AS_EFFECTIVE:
    df["tok"] = np.where(df["deadline_miss"], df["deadline_tok"], df["llm_tok"])
else:
    df["tok"] = df["llm_tok"].fillna(df["deadline_tok"])

df["tok"] = df["tok"].fillna("NONE").str.upper()
df = df[df["tok"].isin(ALLOWED_TOKENS)].copy()

df["speed_u"] = df["speed"].apply(to_unit)

# Sort by distance (we'll invert axis later to show 100 -> 0)
df.sort_values("distance", ascending=True, inplace=True)

# Attach lane behavior by step
beh = parse_lane_behavior(BEHAVIOR_PATH)
df["beh_accel"] = np.nan
df["beh_head"] = np.nan
df["beh_steer"] = np.nan
if beh is not None:
    df["beh_accel"] = df["step"].apply(lambda i: beh["accel"][i] if i < len(beh["accel"]) else np.nan)
    df["beh_head"]  = df["step"].apply(lambda i: beh["head"][i]  if i < len(beh["head"]) else np.nan)
    df["beh_steer"] = df["step"].apply(lambda i: beh["steer"][i] if i < len(beh["steer"]) else np.nan)

print(df.head(10))
print("\nCounts:\n", df["tok"].value_counts())
print("\nDeadline misses:", int(df["deadline_miss"].sum()))

# ============================
# Plot ALL-IN-ONE
# ============================
fig, ax = plt.subplots(figsize=(14, 7))

# --- Allowed speed bands (shaded) ---
# IMPORTANT: build grid from max->min so shading visually matches approach 100->0
x_min = float(df["distance"].min())
x_max = float(df["distance"].max())
x_grid = np.linspace(x_max, x_min, 600)  # descending grid

for (rd_min, rd_max, vmin, vmax, lab) in SPEED_BANDS:
    if rd_max is None:
        mask = (x_grid >= rd_min)
    else:
        mask = (x_grid > rd_min) & (x_grid <= rd_max)

    if not np.any(mask):
        continue

    ax.fill_between(
        x_grid[mask],
        to_unit(vmin),
        to_unit(vmax),
        alpha=0.15,
        label=f"{lab} ({unit_label()})",
        zorder=1,
    )

# --- Speed trace ---
ax.plot(
    df["distance"],
    df["speed_u"],
    marker=".",
    linewidth=1.0,
    label=f"Speed ({unit_label()})",
    zorder=2,
)

# --- Token markers ---
marker_map = {"NONE": "o", "NOTIFY": "s", "WARNING": "^", "ACTUATE": "D"}
for tok, g in df.groupby("tok"):
    ax.scatter(
        g["distance"], g["speed_u"],
        marker=marker_map.get(tok, "o"),
        label=f"LLM: {tok}",
        zorder=3,
    )

# --- Deadline miss overlay ---
miss = df[df["deadline_miss"]]
if not miss.empty:
    ax.scatter(
        miss["distance"],
        miss["speed_u"],
        marker="x",
        label="Deadline miss",
        zorder=4,
    )

ax.set_xlabel("Relative distance to lane-change (m)")
ax.set_ylabel(f"Speed ({unit_label()})")
ax.set_title("Lane Change: Speed vs Distance + Allowed Buffer + LLM Token + Deadline Miss + Behavior")
ax.grid(True)

# Show as 100 -> 0
ax.invert_xaxis()

# --- Behavior on secondary y-axis (codes) ---
ax2 = ax.twinx()
ax2.set_ylabel("Behavior codes")

beh_plot = df.dropna(subset=["beh_accel", "beh_head", "beh_steer"], how="all")
if not beh_plot.empty:
    ax2.scatter(beh_plot["distance"], beh_plot["beh_accel"], marker="|", label="Behavior: accel", zorder=2)
    ax2.scatter(beh_plot["distance"], beh_plot["beh_head"],  marker="_", label="Behavior: head",  zorder=2)
    ax2.scatter(beh_plot["distance"], beh_plot["beh_steer"], marker="+", label="Behavior: steer", zorder=2)

# ============================
# FORCE legend entries even if token missing (e.g., ACTUATE not present)
# ============================
want_tokens_in_legend = ["NOTIFY", "WARNING", "ACTUATE"]
existing = set(df["tok"].unique().tolist())

dummy_handles = []
dummy_labels = []
for t in want_tokens_in_legend:
    if t not in existing:
        dummy_handles.append(Line2D([0], [0], marker=marker_map[t], linestyle="None"))
        dummy_labels.append(f"LLM: {t}")

# Combine legends from both axes + dummy handles
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()

ax.legend(h1 + h2 + dummy_handles, l1 + l2 + dummy_labels, loc="upper right")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
plt.show()

print(f"\nSaved: {OUT_PNG}")