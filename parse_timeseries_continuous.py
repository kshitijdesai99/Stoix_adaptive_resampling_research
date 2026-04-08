import re
import csv
import json
import os

TIMESERIES_DIR = "continous/timeseries"
OUTPUT         = "continous/timeseries.csv"

ABS_KEY_MAP = {
    "Steps per second":    "steps_per_second",
    "Episode length mean": "episode_length_mean",
    "Episode length std":  "episode_length_std",
    "Episode length min":  "episode_length_min",
    "Episode length max":  "episode_length_max",
    "Episode return mean": "episode_return_mean",
    "Episode return std":  "episode_return_std",
    "Episode return min":  "episode_return_min",
    "Episode return max":  "episode_return_max",
}

TRAINER_SCALAR_MAP = {
    "Adaptive temperature":      "adaptive_temperature",
    "Alpha mean":                "alpha_mean",
    "Alpha stddev":              "alpha_stddev",
    "Kl mean":                   "kl_mean",
    "Kl stddev":                 "kl_stddev",
    "Kl nonparametric":          "kl_nonparametric",
    "Kl nonparametric relative": "kl_nonparametric_relative",
    "Loss alpha mean":           "loss_alpha_mean",
    "Loss alpha stddev":         "loss_alpha_stddev",
    "Loss kl mean":              "loss_kl_mean",
    "Loss kl stddev":            "loss_kl_stddev",
    "Loss policy mean":          "loss_policy_mean",
    "Loss policy stddev":        "loss_policy_stddev",
    "Loss temperature":          "loss_temperature",
    "Steps per second":          "steps_per_second",
    "Value loss":                "value_loss",
    "Value pred mean":           "value_pred_mean",
    "Value pred std":            "value_pred_std",
}

TRAINER_DEPTH_PREFIXES = [
    "Entropy depth",
    "Ess fraction depth",
    "Mean td weights depth",
    "Particles alive depth",
    "Resample depth",
]
TRAINER_DEPTH_COL_MAP = {p: p.lower().replace(" ", "_") for p in TRAINER_DEPTH_PREFIXES}

ACTOR_COLS    = ["exp_id", "timestep"] + [f"{v}_actor"    for v in ABS_KEY_MAP.values()]
EVAL_COLS     = [f"{v}_eval"     for v in ABS_KEY_MAP.values()]
TRAINER_COLS  = ([f"{v}_trainer" for v in TRAINER_SCALAR_MAP.values()]
                 + list(TRAINER_DEPTH_COL_MAP.values()))

ALL_COLUMNS = ACTOR_COLS + EVAL_COLS + TRAINER_COLS


def parse_kv_line(prefix, line):
    result = {}
    for field in line[len(prefix):].split(" | "):
        if ": " in field:
            k, v = field.split(": ", 1)
            result[k.strip()] = v.strip()
    return result


rows = []

for fname in sorted(os.listdir(TIMESERIES_DIR)):
    if not fname.endswith(".txt"):
        continue
    exp_id = int(os.path.splitext(fname)[0])
    path = os.path.join(TIMESERIES_DIR, fname)

    current_timestep = None
    current_row = None

    with open(path) as f:
        for line in f:
            m = re.match(r'^MISC - Timestep:\s*(\d+)', line)
            if m:
                if current_row is not None:
                    rows.append(current_row)
                current_timestep = int(m.group(1))
                current_row = {col: "" for col in ALL_COLUMNS}
                current_row["exp_id"]    = exp_id
                current_row["timestep"] = current_timestep
                continue

            if current_row is None:
                continue

            if line.startswith("ACTOR - "):
                kv = parse_kv_line("ACTOR - ", line)
                for src, dst in ABS_KEY_MAP.items():
                    if src in kv:
                        current_row[f"{dst}_actor"] = kv[src]

            elif line.startswith("EVALUATOR - "):
                kv = parse_kv_line("EVALUATOR - ", line)
                for src, dst in ABS_KEY_MAP.items():
                    if src in kv:
                        current_row[f"{dst}_eval"] = kv[src]

            elif line.startswith("TRAINER - "):
                kv = parse_kv_line("TRAINER - ", line)
                depth_buckets = {p: {} for p in TRAINER_DEPTH_PREFIXES}
                for k, v in kv.items():
                    dm = re.match(r'^(.+):(\d+)$', k)
                    if dm:
                        prefix, idx = dm.group(1), int(dm.group(2))
                        if prefix in depth_buckets:
                            depth_buckets[prefix][idx] = float(v)
                    elif k in TRAINER_SCALAR_MAP:
                        current_row[f"{TRAINER_SCALAR_MAP[k]}_trainer"] = v
                for prefix, bucket in depth_buckets.items():
                    col = TRAINER_DEPTH_COL_MAP[prefix]
                    current_row[col] = json.dumps([bucket[i] for i in sorted(bucket)])

    if current_row is not None:
        rows.append(current_row)

with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

print(f"Written {len(rows)} rows to {OUTPUT}")
