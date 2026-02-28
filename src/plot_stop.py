import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================
# CONFIG
# ============================
LOG_PATH = "src/stop_sign.log"
OUT_PNG = "stop_sign.svg"

UNIT = "m/s"  # "m/s" or "km/h"
USE_DEADLINE_TOKEN_AS_EFFECTIVE = False
ALLOWED_TOKENS = {"NONE", "NOTIFY", "WARNING", "ACTUATE"}

# ============================
# IDEAL SPEED + ENVELOPE
# ============================
# Shape derived from standard braking curve (v ∝ √distance).
# Cruise at 10 m/s, smooth deceleration in last 25 m to v=0.
import math
BRAKE_START_RD = 25.0   # braking begins at this relative distance
V_CRUISE       = 10.0   # cruise speed (m/s)
ENV_W          = 2.5    # envelope half-width (m/s)

ENVELOPE_COLOR = "#d62728"
IDEAL_COLOR = "#2ca02c"

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
    """Ideal speed at relative distance rd.  Cruise, then √-curve braking."""
    if rd >= BRAKE_START_RD:
        return V_CRUISE
    if rd <= 0.0:
        return 0.0
    return V_CRUISE * math.sqrt(rd / BRAKE_START_RD)

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
df["speed_u"] = df["speed"].apply(to_unit)
df.sort_values("distance", ascending=False, inplace=True)

print(df.head(10))
print("\nCounts:\n", df["tok"].value_counts())
print("\nDeadline misses:", int(df["deadline_miss"].sum()))

# ============================
# Build envelope grid
# ============================
x_grid = np.linspace(0.0, 100.0, 900)  # fixed range, independent of data

ideal_arr = np.array([ideal_speed(x) for x in x_grid])
upper_env = np.maximum(0.0, ideal_arr + ENV_W)
lower_env = np.maximum(0.0, ideal_arr - ENV_W)

# ============================
# PLOT
# ============================
TOKEN_ORDER = ["ACTUATE", "NOTIFY", "WARNING", "NONE"]
TOKEN_MARKER = {"NONE": "o", "NOTIFY": "s", "WARNING": "^", "ACTUATE": "D"}
TOKEN_COLOR = {
    "ACTUATE": "#1f77b4",
    "NOTIFY": "#ff7f0e",
    "WARNING": "#2ca02c",
    "NONE": "#1f77b4",
}
SPEED_COLOR = "#1f77b4"
DEADLINE_COLOR = "#d62728"
ENVELOPE_COLOR = "#d62728"
SPEED_COLOR = "#1f77b4"
fig, ax = plt.subplots(figsize=(14, 7))

# Upper & lower envelopes
ax.plot(x_grid, to_unit(upper_env), linewidth=2.0, color=ENVELOPE_COLOR,
        label="Upper envelope", zorder=2)
ax.plot(x_grid, to_unit(lower_env), linewidth=2.0, color=ENVELOPE_COLOR,
        linestyle="--", label="Lower envelope", zorder=2)

# Speed trace
ax.plot(df["distance"], df["speed_u"], marker=".", linewidth=1.2,
        color=SPEED_COLOR, label="_nolegend_", zorder=3)

# Token markers
for tok in TOKEN_ORDER:
    g = df[df["tok"] == tok]
    if g.empty:
        continue
    ax.scatter(
        g["distance"], g["speed_u"],
        marker=TOKEN_MARKER[tok],
        color=TOKEN_COLOR[tok],
        s=160, edgecolors="black", linewidths=0.8,
        label="_nolegend_", zorder=4,
    )


# Deadline misses
miss = df[df["deadline_miss"]]
if not miss.empty:
    ax.scatter(
        miss["distance"], miss["speed_u"],
        marker="x", s=85, color=DEADLINE_COLOR,
        label="_nolegend_", zorder=5,
    )

ax.set_xlabel("Relative distance (m)", fontsize=16, fontweight="bold")
ax.set_ylabel(f"Speed ({unit_label()})", fontsize=16, fontweight="bold")
ax.tick_params(axis="both", labelsize=18)
ax.set_ylim(bottom=0)
ax.grid(True)
ax.invert_xaxis()
x_left = max(100.0, float(df["distance"].max()))
ax.set_xlim(x_left, 0.0)   # reversed x-axis, hard stop at 0
ax.margins(x=0)

# Force legend entries for all tokens even if absent
legend_handles = [
    Line2D([0], [0], color=ENVELOPE_COLOR, linewidth=2.0, label="Upper envelope"),
    Line2D([0], [0], color=ENVELOPE_COLOR, linewidth=2.0, linestyle="--", label="Lower envelope"),
    Line2D([0], [0], color=SPEED_COLOR, linewidth=1.2, marker=".", label=f"Speed ({unit_label()})"),
    Line2D([0], [0], marker='D', linestyle='None', markerfacecolor=TOKEN_COLOR["ACTUATE"], markeredgecolor='black', label='LLM: ACTUATE'),
    Line2D([0], [0], marker='s', linestyle='None', markerfacecolor=TOKEN_COLOR["NOTIFY"], markeredgecolor='black', label='LLM: NOTIFY'),
    Line2D([0], [0], marker='^', linestyle='None', markerfacecolor=TOKEN_COLOR["WARNING"], markeredgecolor='black', label='LLM: WARNING'),
    Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=TOKEN_COLOR["NONE"], markeredgecolor='black', label='LLM: NONE'),
    Line2D([0], [0], marker='x', linestyle='None', color=DEADLINE_COLOR, label='Deadline miss'),
]
ax.legend(handles=legend_handles, loc="lower left", fontsize=14)


h1, l1 = ax.get_legend_handles_labels()
all_h = h1 + legend_handles
all_l = l1 + [h.get_label() for h in legend_handles]

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
