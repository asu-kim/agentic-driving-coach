import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================
# CONFIG
# ============================
LOG_PATH = "src/lane_change.log"
OUT_PNG = "lane_change.svg"

UNIT = "m/s"  # or "km/h"
USE_DEADLINE_TOKEN_AS_EFFECTIVE = False
ALLOWED_TOKENS = {"NONE", "NOTIFY", "WARNING", "ACTUATE"}

# ============================
# IDEAL SPEED + ENVELOPE
# ============================
# Lane change: maintain constant speed ~18.5 m/s throughout
# Piecewise-linear ideal: (rd, v_mps)
IDEAL_ANCHORS = [(100.0, 18.5), (0.0, 18.5)]
ENV_W = 1.5  # envelope half-width (m/s)

ENVELOPE_COLOR = "#d62728"

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

def ideal_speed(rd: float) -> float:
    """Piecewise-linear ideal speed at distance rd (m/s)."""
    anchors = IDEAL_ANCHORS
    r = max(anchors[-1][0], min(anchors[0][0], rd))
    for (r0, v0), (r1, v1) in zip(anchors[:-1], anchors[1:]):
        if r <= r0 and r >= r1:
            t = (r0 - r) / (r0 - r1 + 1e-9)
            return v0 + t * (v1 - v0)
    return anchors[-1][1]

# ============================
# Parse log
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
df.sort_values("distance", ascending=False, inplace=True)

print(df.head(10))
print("\nCounts:\n", df["tok"].value_counts())
print("\nDeadline misses:", int(df["deadline_miss"].sum()))

# ============================
# Build envelope grid
# ============================
x_min = float(df["distance"].min())
x_max = float(df["distance"].max())
x_grid = np.linspace(x_min, x_max, 900)

ideal_arr = np.array([ideal_speed(x) for x in x_grid])
upper_env = np.maximum(0.0, ideal_arr + ENV_W)
lower_env = np.maximum(0.0, ideal_arr - ENV_W)

# ============================
# PLOT
# ============================
TOKEN_MARKER = {"NONE": "o", "NOTIFY": "s", "WARNING": "^", "ACTUATE": "D"}

fig, ax = plt.subplots(figsize=(14, 7))

# Upper & lower envelopes
ax.plot(x_grid, to_unit(upper_env), linewidth=2.0, color=ENVELOPE_COLOR,
        label="Upper envelope", zorder=2)
ax.plot(x_grid, to_unit(lower_env), linewidth=2.0, color=ENVELOPE_COLOR,
        linestyle="--", label="Lower envelope", zorder=2)

# Speed trace
ax.plot(
    df["distance"],
    df["speed_u"],
    marker=".",
    linewidth=1.2,
    label=f"Speed ({unit_label()})",
    zorder=3,
)

# Token markers
for tok, g in df.groupby("tok"):
    ax.scatter(
        g["distance"],
        g["speed_u"],
        marker=TOKEN_MARKER.get(tok, "o"),
        s=160,
        edgecolors="black",
        linewidths=0.8,
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

ax.set_xlabel("Relative distance to lane-change (m)", fontsize=16, fontweight="bold")
ax.set_ylabel(f"Speed ({unit_label()})", fontsize=16, fontweight="bold")
ax.tick_params(axis="both", labelsize=18)
ax.grid(True)
ax.set_ylim(bottom=0)
ax.invert_xaxis()

# Force legend entries for all tokens even if absent
legend_force = [
    Line2D([0], [0], marker='o', linestyle='None', label='LLM: NONE'),
    Line2D([0], [0], marker='s', linestyle='None', label='LLM: NOTIFY'),
    Line2D([0], [0], marker='^', linestyle='None', label='LLM: WARNING'),
    Line2D([0], [0], marker='D', linestyle='None', label='LLM: ACTUATE'),
    Line2D([0], [0], marker='x', linestyle='None', label='Deadline miss'),
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

ax.legend(uniq_h, uniq_l, loc="lower left", fontsize=14)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
plt.show()
print(f"\nSaved: {OUT_PNG}")
