import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# CONFIG
# ============================
LOG_PATH = "src/stop_sign.log"            # runtime log file
BEHAVIOR_PATH = "src/DriverBehaviorStop.txt"  # driver behavior file (one int per line)
OUT_PNG = "stop_sign.svg"

# Units: "m/s" or "km/h"
UNIT = "m/s"

# If True: when deadline miss happens, use deadline_tok as effective token
# If False: keep llm_tok as effective token, but still mark misses with X
USE_DEADLINE_TOKEN_AS_EFFECTIVE = False

# Keep only these tokens for plotting
ALLOWED_TOKENS = {"NONE", "NOTIFY", "WARNING", "ACTUATE"}

# ============================
# SPEED BUFFER / ALLOWED BANDS
# ============================
# (rd_min, rd_max, v_min, v_max, label)
# Interpretation:
# - rd in [rd_min, rd_max]  (inclusive, except where your logic uses 50<rd<=60)
# - If rd_max is None => rd >= rd_min
SPEED_BANDS = [
    (98.0, None, 8.0, 13.0, "Allowed (rd>=98): 3–13"),
    (50.0, 60.0, 5.0, 9.0,  "Allowed (50<rd<=60): 5–9"),
    (0.0, 25.0, 0.0, 2.5,   "Allowed (rd<=25): 0–2.5"),
]

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
    if UNIT.lower() == "km/h":
        return v_mps * 3.6
    return v_mps

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
            cur["deadline_tok"] = m.group(1)
            cur["deadline_miss"] = True
            continue

        m = re_llm.search(line)
        if m:
            cur["llm_ms"] = float(m.group(1))
            cur["llm_tok"] = m.group(2)
            flush_if_ready()
            continue

flush_if_ready()

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError("Parsed 0 rows. Check LOG_PATH and patterns vs your log output.")

# Effective token (final decision for plotting)
if USE_DEADLINE_TOKEN_AS_EFFECTIVE:
    df["tok"] = np.where(df["deadline_miss"], df["deadline_tok"], df["llm_tok"])
else:
    df["tok"] = df["llm_tok"].fillna(df["deadline_tok"])

df["tok"] = df["tok"].fillna("NONE")
df = df[df["tok"].isin(ALLOWED_TOKENS)].copy()

# Convert speed units
df["speed_u"] = df["speed"].apply(to_unit)

# Sort by distance descending (far -> near)
df.sort_values("distance", ascending=False, inplace=True)

# Attach behavior by step index (best effort)
beh = parse_behavior_file(BEHAVIOR_PATH)
if beh is not None:
    df["behavior"] = df["step"].apply(lambda i: beh[i] if i < len(beh) else np.nan)
else:
    df["behavior"] = np.nan

print(df.head(10))
print("\nCounts:\n", df["tok"].value_counts())
print("\nDeadline misses:", int(df["deadline_miss"].sum()))
print("\nBehavior rows parsed:", 0 if beh is None else len(beh))

# ============================
# PLOT: ALL IN ONE FIGURE
# ============================
fig, ax = plt.subplots(figsize=(14, 7))

# --- Allowed speed bands (shaded) ---
x_min = float(df["distance"].min())
x_max = float(df["distance"].max())
x_grid = np.linspace(x_min, x_max, 800)

def band_mask(x, rd_min, rd_max):
    """Mask of x values that fall into the distance band."""
    if rd_max is None:
        return x >= rd_min
    # Your middle rule is 50 < rd <= 60 (exclude 50)
    if rd_min == 50.0 and rd_max == 60.0:
        return (x > rd_min) & (x <= rd_max)
    # Otherwise inclusive range
    return (x >= rd_min) & (x <= rd_max)

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

# --- Speed trace ---
ax.plot(
    df["distance"],
    df["speed_u"],
    marker=".",
    linewidth=1.2,
    label=f"Speed ({unit_label()})",
    zorder=2,
)

# --- Token markers ---
marker_map = {"NONE": "o", "NOTIFY": "s", "WARNING": "^", "ACTUATE": "D"}
for tok, g in df.groupby("tok"):
    ax.scatter(
        g["distance"],
        g["speed_u"],
        marker=marker_map.get(tok, "o"),
        label=f"LLM: {tok}",
        zorder=3,
    )

# --- Deadline misses overlay ---
miss = df[df["deadline_miss"]]
if not miss.empty:
    ax.scatter(
        miss["distance"],
        miss["speed_u"],
        marker="x",
        s=70,
        label="Deadline miss",
        zorder=4,
    )

ax.set_xlabel("Relative distance (m)")
ax.set_ylabel(f"Speed ({unit_label()})")
ax.grid(True)

# IMPORTANT: distance decreases 100 -> 0
# This makes the plot read like: left=far (100), right=near (0)
ax.invert_xaxis()

# --- Behavior on secondary y-axis ---
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

# Combine legends from both axes
from matplotlib.lines import Line2D

# --- Force legend entries for all tokens ---
legend_elements = [
    Line2D([0], [0], marker='o', linestyle='None', label='LLM: NONE'),
    Line2D([0], [0], marker='s', linestyle='None', label='LLM: NOTIFY'),
    Line2D([0], [0], marker='^', linestyle='None', label='LLM: WARNING'),
    Line2D([0], [0], marker='D', linestyle='None', label='LLM: ACTUATE'),
    Line2D([0], [0], marker='x', linestyle='None', label='Deadline miss'),
]

# get existing handles
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# combine everything
all_handles = handles1 + handles2 + legend_elements
all_labels = labels1 + labels2 + [h.get_label() for h in legend_elements]

# remove duplicates while preserving order
seen = set()
unique = []
for h, l in zip(all_handles, all_labels):
    if l not in seen:
        unique.append((h, l))
        seen.add(l)

ax.legend([u[0] for u in unique], [u[1] for u in unique], loc="upper right")

# ax.set_title("Stop Sign: Speed vs Distance + Bands + LLM Token + Deadline Miss + Behavior")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
plt.show()

print(f"\nSaved: {OUT_PNG}")