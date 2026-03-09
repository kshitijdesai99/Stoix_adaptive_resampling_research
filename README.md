# Goal

Your final project should look like this:

```
Stoix_snapshot/
│
├── Stoix/                (main repo)
├── external/             (all git dependencies)
│     ├── flashbax/
│     ├── marl-eval/
│     ├── optax/
│     ├── xminigrid/
│     ├── gymnax/
│     ├── popjym/
│     ├── popgym-arcade/
│     └── Stoa/
│
├── pyproject.toml
└── uv.lock
```

Now **everything exists locally**.

---

# Step 1 — Create workspace

```
mkdir stoix_snapshot
cd stoix_snapshot
```

---

# Step 2 — Clone Stoix

```
git clone https://github.com/EdanToledo/Stoix.git
cd Stoix
```

Freeze commit:

```
git rev-parse HEAD > stoix_commit.txt
```

Go back:

```
cd ..
```

---

# Step 3 — Create folder for external repos

```
mkdir external
cd external
```

---

# Step 4 — Clone every git dependency

These come from `[tool.uv.sources]`.

Run:

```
git clone https://github.com/instadeepai/flashbax.git
git clone https://github.com/EdanToledo/marl-eval.git
git clone https://github.com/google-deepmind/optax.git
git clone https://github.com/corl-team/xland-minigrid.git
git clone https://github.com/FLAIROx/popjym.git
git clone https://github.com/bolt-research/popgym-arcade.git
git clone https://github.com/RobertTLange/gymnax.git
git clone https://github.com/EdanToledo/Stoa.git
```

Now your folder becomes:

```
external/
 ├ flashbax
 ├ marl-eval
 ├ optax
 ├ xland-minigrid
 ├ popjym
 ├ popgym-arcade
 ├ gymnax
 └ Stoa
```

---

# Step 5 — Modify `pyproject.toml` to use local repos

Open:

```
Stoix/pyproject.toml
```

Change the dependencies.

Example change:

### BEFORE

```
flashbax @ git+https://github.com/instadeepai/flashbax@main
```

### AFTER

```
flashbax @ file://../external/flashbax
```

Do this for all repos:

```
flashbax @ file://../external/flashbax
id-marl-eval @ file://../external/marl-eval
optax @ file://../external/optax
xminigrid @ file://../external/xland-minigrid
gymnax @ file://../external/gymnax
popjym @ file://../external/popjym
popgym-arcade @ file://../external/popgym-arcade
stoa-env[all] @ file://../external/Stoa
```

---

# Step 6 — Install environment

Go back to Stoix:

```
cd Stoix
```

Run:

```
uv sync
```

Now the environment installs **from local repositories instead of GitHub**.

---

# Step 7 — Save the lock file

`uv sync` generates:

```
uv.lock
```

This freezes versions.

---

# Step 8 — Remove virtual environment

We don't archive `.venv`.

```
rm -rf .venv
```

---

# Step 9 — Zip everything

Go to parent directory:

```
cd ..
```

Create archive:

```
zip -r stoix_full_snapshot.zip Stoix external
```

---

# Your snapshot now contains

```
Stoix/
external/
pyproject.toml
uv.lock
stoix_commit.txt
```

This means:

| Dependency | Source |
| ---------- | ------ |
| Stoix      | local  |
| flashbax   | local  |
| gymnax     | local  |
| optax      | local  |
| marl-eval  | local  |

Even if **GitHub disappears tomorrow**, you can still install.

---

# Restore later

Unzip:

```
unzip stoix_full_snapshot.zip
cd Stoix
```

Install:

```
uv sync
source .venv/bin/activate
```

Environment recreated.

---

# Important note

You also added:

```
jupyter
lab
```

These are **not needed for Stoix training**, but keeping them is fine.

---

# Final mental model

Your experiment depends on:

```
Stoix code
↓
external repos
↓
PyPI packages
```

Now you archived:

```
Stoix code ✔
external repos ✔
dependency versions ✔
```

So reproducibility is **very strong**.
