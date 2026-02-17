import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================
# CONFIG
# ============================
LOG_PATH = "src/speed_change.log"                # <-- your speed-change runtime log
BEHAVIOR_PATH = "src/DriverBehaviorSpeed.txt"    # optional; one int per line
OUT_PREFIX = "speed_change"

UNIT = "m/s"  # "m/s" or "km/h"
USE_DEADLINE_TOKEN_AS_EFFECTIVE = False  # if True, tok = deadline_tok when miss else llm_tok

# Keep only these tokens (you said you want only these)
ALLOWED_TOKENS = {"NONE", "NOTIFY", "WARNING", "ACTUATE"}

# ============================
# SPEED BANDS (VISUAL ONLY)
# ============================
# Format:
#   ("phase", rd_min, rd_max, v_min, v_max, label)
# phase: "approach" uses Relative distance (rd)
#        "after"    uses Distance-after-sign (d_after)
#
# IMPORTANT:
# - For approach, rd decreases from ~100 to 0.
# - We'll still compute bands in normal numeric rd, and we will invert the x-axis.
SPEED_BANDS = [
    # Example for speed-change target ~11 m/s
    ("approach", 50.0, 60.0, 10.5, 11.5, "Allowed (50<rd<=60): 11±0.5"),
    ("approach", 0.0, 25.0,  0.0, 11.5, "Allowed (rd<=25): <=11.5"),
    ("after",    0.0, 50.0, 10.5, 11.5, "Allowed (after): 11±0.5"),
]

# ============================
# REGEX patterns
# ============================
re_dist = re.compile(r"\[Relative DISTANCE\]:\s*([0-9]*\.?[0-9]+)")
re_after = re.compile(r"\[DISTANCE after speed\]:\s*([0-9]*\.?[0-9]+)")
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
# Parse log into rows
# ============================
rows = []
cur = {
    "step": 0,
    "distance": None,       # rd
    "dist_after": None,     # after sign
    "speed": None,
    "llm_ms": None,
    "llm_tok": None,
    "deadline_tok": None,
    "deadline_miss": False,
}

def flush_if_ready():
    # Need speed + (rd or dist_after) + (LLM or deadline)
    if cur["speed"] is None:
        return
    if cur["distance"] is None and cur["dist_after"] is None:
        return
    if cur["llm_tok"] is None and cur["deadline_tok"] is None:
        return

    rows.append(cur.copy())

    cur["step"] += 1
    cur["distance"] = None
    cur["dist_after"] = None
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

        m = re_after.search(line)
        if m:
            cur["dist_after"] = float(m.group(1))
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
    raise RuntimeError("Parsed 0 rows. Check LOG_PATH and regex patterns vs your log output.")

# Effective token
if USE_DEADLINE_TOKEN_AS_EFFECTIVE:
    df["tok"] = np.where(df["deadline_miss"], df["deadline_tok"], df["llm_tok"])
else:
    df["tok"] = df["llm_tok"].fillna(df["deadline_tok"])

df["tok"] = df["tok"].fillna("NONE").str.upper()
df = df[df["tok"].isin(ALLOWED_TOKENS)].copy()

# Convert units
df["speed_u"] = df["speed"].apply(to_unit)

# Attach behavior (optional)
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
# Plotting
# ============================
marker_map = {"NONE": "o", "NOTIFY": "s", "WARNING": "^", "ACTUATE": "D"}

def shade_bands(ax, x_vals: np.ndarray, phase: str):
    """Shade speed bands on an axis for a given phase."""
    x_vals = x_vals[np.isfinite(x_vals)]
    if x_vals.size == 0:
        return

    x_min = float(np.min(x_vals))
    x_max = float(np.max(x_vals))
    # grid across actual observed range
    x_grid = np.linspace(x_min, x_max, 800)

    for (p, xmin, xmax, vmin, vmax, label) in SPEED_BANDS:
        if p != phase:
            continue
        # Bands are defined as: (xmin, xmax] for your rule-like zones
        mask = (x_grid > xmin) & (x_grid <= xmax)
        if not np.any(mask):
            continue

        ax.fill_between(
            x_grid[mask],
            to_unit(vmin),
            to_unit(vmax),
            alpha=0.18,
            zorder=1,
        )
        # Add a hidden proxy for legend (so band label appears once)
        ax.plot([], [], linewidth=8, alpha=0.18, label=f"{label} ({unit_label()})")

def force_full_legend(ax, ax2=None):
    """Force legend entries for all tokens + deadline miss + behavior, even if absent."""
    proxies = []
    labels = []

    # Token proxies
    for tok in ["NONE", "NOTIFY", "WARNING", "ACTUATE"]:
        proxies.append(Line2D([0], [0], marker=marker_map[tok], linestyle="None"))
        labels.append(f"LLM: {tok}")

    # Deadline proxy
    proxies.append(Line2D([0], [0], marker="x", linestyle="None"))
    labels.append("Deadline miss")

    # Behavior proxy (secondary axis)
    if ax2 is not None:
        proxies.append(Line2D([0], [0], marker="|", linestyle="None"))
        labels.append("Behavior (DriverBehavior*.txt)")

    # Existing handles (bands + speed line)
    h1, l1 = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
    else:
        h2, l2 = [], []

    ax.legend(h1 + h2 + proxies, l1 + l2 + labels, loc="best")

def plot_phase(df_phase: pd.DataFrame, xcol: str, phase_name: str,  out_path: str):
    if df_phase.empty:
        # print(f"[SKIP] No rows for {title}")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    # Shade bands
    shade_bands(ax, df_phase[xcol].to_numpy(dtype=float), phase_name)

    # Plot speed trace (sorted for a clean line)
    # For approach (rd), sort descending (far->near). We'll invert x-axis for "100 -> 0" view.
    if xcol == "distance":
        df_line = df_phase.sort_values(xcol, ascending=False).copy()
    else:
        df_line = df_phase.sort_values(xcol, ascending=True).copy()

    ax.plot(df_line[xcol], df_line["speed_u"], marker=".", linewidth=1.0, label=f"Speed ({unit_label()})", zorder=2)

    # Token markers
    for tok, g in df_phase.groupby("tok"):
        ax.scatter(g[xcol], g["speed_u"], marker=marker_map.get(tok, "o"), label=f"LLM: {tok}", zorder=3)

    # Deadline misses overlay
    miss = df_phase[df_phase["deadline_miss"]]
    if not miss.empty:
        ax.scatter(miss[xcol], miss["speed_u"], marker="x", label="Deadline miss", zorder=4)

    ax.set_xlabel("Relative distance (m)" if xcol == "distance" else "Distance after sign (m)")
    ax.set_ylabel(f"Speed ({unit_label()})")
    # ax.set_title(title)
    ax.grid(True)

    # Make approach plot read like 100 -> 0 (approaching the sign)
    if xcol == "distance":
        ax.invert_xaxis()

    # Behavior on secondary axis
    ax2 = ax.twinx()
    beh_g = df_phase.dropna(subset=["behavior"])
    if not beh_g.empty:
        ax2.scatter(beh_g[xcol], beh_g["behavior"], marker="|", label="Behavior (DriverBehavior*.txt)", zorder=2)
    ax2.set_ylabel("Behavior code")

    # Force legend to always include ACTUATE etc.
    force_full_legend(ax, ax2=ax2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"Saved: {out_path}")

# ============================
# Split phases
# ============================
df_approach = df[df["distance"].fillna(0.0) > 0.0].copy()
df_after = df[df["dist_after"].fillna(0.0) > 0.0].copy()

plot_phase(
    df_approach,
    xcol="distance",
    phase_name="approach",
    # title="Speed Change (Approach): Speed vs Relative Distance + Bands + Tokens + Deadline + Behavior",
    out_path=f"{OUT_PREFIX}.svg",
)

# plot_phase(
#     df_after,
#     xcol="dist_after",
#     phase_name="after",
#     title="Speed Change (After): Speed vs Distance After Sign + Bands + Tokens + Deadline + Behavior",
#     out_path=f"{OUT_PREFIX}_after.svg",
# )