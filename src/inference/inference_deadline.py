import time
import random
import statistics
import numpy as np
import csv
import ollama

# -----------------------------
# Config
# -----------------------------
MODEL = "llama3.2:1b"          # change to "llama3.1:1b", "llama3.1:8b", "llama3.1:70b", etc.
NUM_RUNS = 300
WARMUP_RUNS = 10             # warmup to avoid cold-start bias
TEMPERATURE = 0.0
NUM_PREDICT = 30
SEED = 42
SAVE_CSV = True
CSV_PATH = f"latency_{MODEL.replace(':', '_')}.csv"

random.seed(SEED)

SYSTEM_PROMPT = (
    "Output exactly ONE line: TOKEN|MSG\n"
    "TOKEN must be one of: NONE, NOTIFY, WARNING, ACTUATE\n"
    "MSG must be <= 12 words, about stopping only.\n"
    "Use meters and m/s only.\n\n"
    "Rules for TOKEN:\n"
    "- if distance_to_stop <= 25 and speed > 2.5: TOKEN=ACTUATE\n"
    "- else if 50 < distance_to_stop <= 60 and (speed > 9 or speed < 5): TOKEN=WARNING\n"
    "- else if distance_to_stop >= 99 and (speed > 13 or speed < 3): TOKEN=NOTIFY\n"
    "- else TOKEN=NONE\n\n"
    "If TOKEN=NONE, output: NONE|\n"
    "Do not output anything else."
)

def one_inference(rd_val: float, v_val: float) -> tuple[float, str]:
    """Run one Ollama chat call and return (latency_ms, raw_text)."""
    t0 = time.perf_counter()
    r = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"distance_to_stop={rd_val:.2f}m speed={v_val:.2f}m/s"},
        ],
        options={"temperature": TEMPERATURE, "num_predict": NUM_PREDICT},
    )
    ms = (time.perf_counter() - t0) * 1000.0
    raw = (r.get("message", {}).get("content", "") or "").strip()
    return ms, raw

def sample_inputs() -> tuple[float, float]:
    """
    Generate a mix of distances/speeds across your rule regions so you don't
    only benchmark one trivial case.
    """
    # 3 bands roughly corresponding to your rules:
    band = random.choice(["far", "mid", "near"])
    if band == "far":
        rd = random.uniform(99.0, 120.0)
        v = random.uniform(0.0, 18.0)
    elif band == "mid":
        rd = random.uniform(50.0, 60.0)
        v = random.uniform(0.0, 18.0)
    else:
        rd = random.uniform(0.0, 25.0)
        v = random.uniform(0.0, 18.0)
    return rd, v

def main():
    # Warmup
    print(f"Model: {MODEL}")
    print(f"Warmup runs: {WARMUP_RUNS}  |  Measured runs: {NUM_RUNS}\n")
    for _ in range(WARMUP_RUNS):
        rd, v = sample_inputs()
        _ms, _raw = one_inference(rd, v)

    # Measurement
    latencies = []
    rows = []

    for i in range(NUM_RUNS):
        rd, v = sample_inputs()
        ms, raw = one_inference(rd, v)
        latencies.append(ms)
        rows.append((i + 1, rd, v, ms, raw))
        print(f"Run {i+1:3d}/{NUM_RUNS}: {ms:8.2f} ms | {raw}")

    # Stats
    mean_ms = statistics.mean(latencies)
    std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    p95_ms = float(np.percentile(latencies, 95))
    p99_ms = float(np.percentile(latencies, 99))
    max_ms = max(latencies)

    print("\n----- Inference Latency Summary -----")
    print(f"Mean      : {mean_ms:.2f} ms")
    print(f"Std       : {std_ms:.2f} ms")
    print(f"P95       : {p95_ms:.2f} ms")
    print(f"P99       : {p99_ms:.2f} ms")
    print(f"Max       : {max_ms:.2f} ms")

    if SAVE_CSV:
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "distance_to_stop_m", "speed_mps", "latency_ms", "raw_output"])
            w.writerows(rows)
        print(f"\nSaved CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()