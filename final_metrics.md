# Final Metrics for Experiment Comparison

Experiments vary: `resampling.mode` (periodic / adaptive), `num_particles`, `search_depth`, environment type (discrete / continuous).

---

## 1. Primary Performance

| Metric | Source | Why it matters |
|--------|--------|----------------|
| `episode_return/mean` | `ABSOLUTE` | Gold standard — best checkpoint, 10× eval episodes |

---

## 2. Search Quality (per depth)

All from `TRAINER`. Compare final-timestep values across runs — same `search_depth` runs are directly comparable depth-by-depth; different `search_depth` runs compare at their respective last depth.

| Metric | Source | Why it matters |
|--------|--------|----------------|
| `Ess fraction depth:*` | `TRAINER` | Particle diversity — near 1.0 is healthy; collapse = wasted compute |
| `Particles alive depth:*` | `TRAINER` | Survival rate per depth — lower at last depth = shallower effective search |
| `Entropy depth:*` | `TRAINER` | Policy uncertainty per depth — collapse = premature convergence |
| `Mean td weights depth:*` | `TRAINER` | Resampling signal strength — near 0 = no discriminative signal |

> Focus on `Ess fraction` and `Particles alive` at the **last depth** (`depth:search_depth`) — these directly reflect whether more depth / particles help.

---

## 3. Resampling Behaviour

All from `TRAINER`. Periodic and adaptive runs are compared on the same metrics — use `Resample depth:*` pattern to confirm each mode behaved as configured.

| Metric | Source | How to compare |
|--------|--------|----------------|
| `Resample depth:*` | `TRAINER` | Periodic: fixed even/odd pattern; Adaptive: check if it fires at different depths than periodic |
| `Adaptive temperature` | `TRAINER` | Adaptive only — lower = tighter resampling threshold; compare settled value across adaptive runs |
| `Kl nonparametric relative` | `TRAINER` | Both modes — 1.0 = on target; compare magnitude across periodic vs adaptive |

---

## 4. Policy Learning Health

All from `TRAINER`. Compare final-timestep values — meaningful across all experiment axes.

| Metric | Source | Why it matters |
|--------|--------|----------------|
| `Kl nonparametric relative` | `TRAINER` | Policy update quality — 1.0 = on target; compare across resampling modes |
| `Kl div` | `TRAINER` | Parametric policy shift — large = unstable updates |
| `Loss policy` | `TRAINER` | Policy gradient signal — compare across `num_particles` and `search_depth` |
| `Value loss` | `TRAINER` | Critic accuracy — high = noisy search targets |
| `Alpha` | `TRAINER` | KL penalty weight — rising = constraint violated |

---

## 5. Throughput

| Metric | Source | Why it matters |
|--------|--------|----------------|
| `Steps per second` | `ACTOR` | Env interaction speed — drops with more `num_particles` / `search_depth` |
| `Steps per second` | `TRAINER` | Gradient update speed — compare across runs for training cost |

> Higher `num_particles` and `search_depth` hurt ACTOR SPS. Track to understand compute cost vs performance trade-off.

---

## Summary Priority

1. `ABSOLUTE/episode_return/mean` (`ABSOLUTE`) — ranking
2. `Ess fraction depth:search_depth` (`TRAINER`) — search effectiveness at max depth
3. `Particles alive depth:search_depth` (`TRAINER`) — particle survival at max depth
4. `Kl nonparametric relative` (`TRAINER`) — policy update quality
5. `Adaptive temperature` (`TRAINER`, adaptive only) — resampling threshold stability
6. `ACTOR/steps_per_second` (`ACTOR`) — compute cost
