import re
import csv
import json

INPUT             = "continous/continous-exp.txt"
OUTPUT_CONFIG     = "continous/config.csv"
OUTPUT_EVALUATOR  = "continous/evaluator.csv"
OUTPUT_ACTOR      = "continous/actor.csv"
OUTPUT_TRAINER    = "continous/trainer.csv"

ABS_COLUMNS = [
    "exp_id",
    "steps_per_second",
    "episode_length_mean", "episode_length_std", "episode_length_min", "episode_length_max",
    "episode_return_mean", "episode_return_std", "episode_return_min", "episode_return_max",
]

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
    "Adaptive temperature": "adaptive_temperature",
    "Alpha mean":           "alpha_mean",
    "Alpha stddev":         "alpha_stddev",
    "Kl mean":              "kl_mean",
    "Kl stddev":            "kl_stddev",
    "Kl nonparametric":     "kl_nonparametric",
    "Kl nonparametric relative": "kl_nonparametric_relative",
    "Loss alpha mean":      "loss_alpha_mean",
    "Loss alpha stddev":    "loss_alpha_stddev",
    "Loss kl mean":         "loss_kl_mean",
    "Loss kl stddev":       "loss_kl_stddev",
    "Loss policy mean":     "loss_policy_mean",
    "Loss policy stddev":   "loss_policy_stddev",
    "Loss temperature":     "loss_temperature",
    "Steps per second":     "steps_per_second",
    "Value loss":           "value_loss",
    "Value pred mean":      "value_pred_mean",
    "Value pred std":       "value_pred_std",
}

TRAINER_DEPTH_PREFIXES = [
    "Entropy depth",
    "Ess fraction depth",
    "Mean td weights depth",
    "Particles alive depth",
    "Resample depth",
]

TRAINER_DEPTH_COL_MAP = {p: p.lower().replace(" ", "_") for p in TRAINER_DEPTH_PREFIXES}

TRAINER_COLUMNS = (
    ["exp_id"]
    + list(TRAINER_SCALAR_MAP.values())
    + list(TRAINER_DEPTH_COL_MAP.values())
)

COLUMNS = [
    "exp_id", "env", "total_num_envs", "total_timesteps",
    "num_particles", "search_depth", "resampling_mode",
    "ess_threshold", "period", "use_residual",
    "total_batch_size", "rollout_length",
    "actor_lr", "critic_lr", "gamma", "gae_lambda",
    "timestep",
]

KEY_MAP = {
    "env":                            "env",
    "arch.total_num_envs":            "total_num_envs",
    "arch.total_timesteps":           "total_timesteps",
    "system.num_particles":           "num_particles",
    "system.search_depth":            "search_depth",
    "system.resampling.mode":         "resampling_mode",
    "system.resampling.ess_threshold":"ess_threshold",
    "system.resampling.period":       "period",
    "system.resampling.use_residual": "use_residual",
    "system.total_batch_size":        "total_batch_size",
    "system.rollout_length":          "rollout_length",
    "system.actor_lr":                "actor_lr",
    "system.critic_lr":               "critic_lr",
    "system.gamma":                   "gamma",
    "system.gae_lambda":              "gae_lambda",
}

rows = []
eval_rows = []
actor_rows = []
trainer_rows = []
current_row = None
current_exp_id = None

with open(INPUT) as f:
    for line in f:
        m = re.match(r'^(\d+)\.\s+python stoix/', line)
        if m:
            current_exp_id = int(m.group(1))
            current_row = {col: "" for col in COLUMNS}
            current_row["exp_id"] = current_exp_id
            current_row["use_residual"] = "false"
            for token in line.split():
                if "=" not in token:
                    continue
                k, v = token.split("=", 1)
                if k in KEY_MAP:
                    current_row[KEY_MAP[k]] = v
            rows.append(current_row)
            continue

        if current_row is not None:
            m2 = re.match(r'^MISC - Timestep:\s*(\d+)', line)
            if m2:
                current_row["timestep"] = int(m2.group(1))
                current_row = None

        if current_exp_id is not None and line.startswith("EVALUATOR - "):
            eval_row = {col: "" for col in ABS_COLUMNS}
            eval_row["exp_id"] = current_exp_id
            for field in line[len("EVALUATOR - "):].split(" | "):
                if ": " in field:
                    k, v = field.split(": ", 1)
                    if k.strip() in ABS_KEY_MAP:
                        eval_row[ABS_KEY_MAP[k.strip()]] = v.strip()
            eval_rows.append(eval_row)

        if current_exp_id is not None and line.startswith("ACTOR - "):
            actor_row = {col: "" for col in ABS_COLUMNS}
            actor_row["exp_id"] = current_exp_id
            for field in line[len("ACTOR - "):].split(" | "):
                if ": " in field:
                    k, v = field.split(": ", 1)
                    if k.strip() in ABS_KEY_MAP:
                        actor_row[ABS_KEY_MAP[k.strip()]] = v.strip()
            actor_rows.append(actor_row)

        if current_exp_id is not None and line.startswith("TRAINER - "):
            trainer_row = {col: "" for col in TRAINER_COLUMNS}
            trainer_row["exp_id"] = current_exp_id
            depth_buckets = {p: {} for p in TRAINER_DEPTH_PREFIXES}
            for field in line[len("TRAINER - "):].split(" | "):
                if ": " not in field:
                    continue
                k, v = field.split(": ", 1)
                k, v = k.strip(), v.strip()
                dm = re.match(r'^(.+):(\d+)$', k)
                if dm:
                    prefix, idx = dm.group(1), int(dm.group(2))
                    if prefix in depth_buckets:
                        depth_buckets[prefix][idx] = float(v)
                elif k in TRAINER_SCALAR_MAP:
                    trainer_row[TRAINER_SCALAR_MAP[k]] = v
            for prefix, bucket in depth_buckets.items():
                col = TRAINER_DEPTH_COL_MAP[prefix]
                trainer_row[col] = json.dumps([bucket[i] for i in sorted(bucket)])
            trainer_rows.append(trainer_row)

with open(OUTPUT_CONFIG, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

with open(OUTPUT_EVALUATOR, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=ABS_COLUMNS)
    writer.writeheader()
    writer.writerows(eval_rows)

with open(OUTPUT_ACTOR, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=ABS_COLUMNS)
    writer.writeheader()
    writer.writerows(actor_rows)

with open(OUTPUT_TRAINER, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=TRAINER_COLUMNS)
    writer.writeheader()
    writer.writerows(trainer_rows)

print(f"Written {len(rows)} rows to {OUTPUT_CONFIG}")
print(f"Written {len(eval_rows)} rows to {OUTPUT_EVALUATOR}")
print(f"Written {len(actor_rows)} rows to {OUTPUT_ACTOR}")
print(f"Written {len(trainer_rows)} rows to {OUTPUT_TRAINER}")
