import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("keyboard_log.csv")

event = df["event_time_ms"]
lf = df["lf_time_ms"]
state = df["state"]

plt.figure(figsize=(12,3))

y_event = [1]*len(df)
y_lf = [2]*len(df)

colors = ["red" if s == "DOWN" else "blue" for s in state]

plt.scatter(event, y_event, c=colors, s=20)
plt.scatter(lf, y_lf, c=colors, marker="x", s=20)

for i in range(len(df)):
    plt.plot([event[i], lf[i]], [1, 2], color="gray", alpha=0.3)

plt.yticks([1,2], ["Event", "LF"])
plt.xlabel("Time (ms)")
plt.title("Event vs LF Timing")

plt.grid(True)
plt.savefig("timeline.png", dpi=300)
plt.show()