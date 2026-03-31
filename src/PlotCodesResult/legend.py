import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Create legend handles manually
legend_items = [

    Line2D([0], [0],
           color="#1f77b4", lw=15,
           label="Car velocity (m/s)"),

    Line2D([0], [0],
           color="#b41f1f", lw=15,
           label="Upper bound (m/s)"),

    Line2D([0], [0],
       color="#b41f1f",
       lw=15,
       linestyle="--",
       label="Lower bound (m/s)"),

    Line2D([0], [0],
           marker="X",
           color="w",
           markerfacecolor="#d62728",
           markeredgecolor="black",
           markersize=80,
           linestyle="None",
           label="Deadline miss"),

    Line2D([0], [0],
           marker="^",
           color="w",
           markerfacecolor="#2ca02c",
           markeredgecolor="black",
           markersize=75,
           linestyle="None",
           label="Instruction"),

    Line2D([0], [0],
           marker="^",
           color="w",
           markerfacecolor="#ff7f0e",
           markeredgecolor="black",
           markersize=75,
           linestyle="None",
           label="Actuation command"),
           
]

fig, ax = plt.subplots(figsize=(50, 7))
ax.axis("off")

legend = ax.legend(
    handles=legend_items,
    loc="center",
    ncol=3,          # 6 rows, 1 column
    fontsize=75,
    frameon=False
)

for text in legend.get_texts():
    text.set_fontweight("bold")

plt.savefig("lane_change_legend.svg", dpi=300, bbox_inches="tight")
plt.show()