### Instructions to run experiments in discrete environment

git clone https://github.com/kshitijdesai99/Stoix_adaptive_resampling_research.git
cd Stoix_adaptive_resampling_research
pip install uv
uv sync
source .venv/bin/activate
uv add jax['cuda']

mkdir -p stoix/configs/env/xland_minigrid 

for env in "door_key_8x8:MiniGrid-DoorKey-8x8:minigrid_doorkey_8x8" "door_key_16x16:MiniGrid-DoorKey-16x16:minigrid_doorkey_16x16" "fourrooms:MiniGrid-FourRooms:minigrid_fourrooms" "empty_16x16:MiniGrid-Empty-16x16:minigrid_empty_16x16"; do IFS=: read name scenario task <<< "$env"; cat > "stoix/configs/env/xland_minigrid/${name}.yaml" << EOF
env_name: xland_minigrid
scenario:
  name: ${scenario}
  task_name: ${task}
kwargs: {}
eval_metric: episode_return
wrapper:
  _target_: stoa.FlattenObservationWrapper
EOF
done

python stoix/systems/spo/ff_spo.py \
  env=xland_minigrid/door_key_16x16 \
  arch.total_num_envs=32 \
  arch.total_timesteps=5000000 \
  system.num_particles=16 \
  system.search_depth=64 \
  system.resampling.mode=ess \
  system.resampling.ess_threshold=0.5 \
  system.total_batch_size=64 \
  system.rollout_length=128 \
  system.actor_lr=0.0003 \
  system.critic_lr=0.0003 \
  system.gamma=0.99 \
  system.gae_lambda=0.95

# Instructions to run experiments in continuous environment

python stoix/systems/spo/ff_spo_continuous.py \
  env=brax/halfcheetah \
  arch.total_num_envs=128 \
  arch.total_timesteps=5000000 \
  system.num_particles=64 \
  system.search_depth=32 \
  system.resampling.mode=ess \
  system.resampling.ess_threshold=0.5 \
  system.total_batch_size=64 \
  system.rollout_length=32 \
  system.actor_lr=0.0003 \
  system.critic_lr=0.0003 \
  system.gamma=0.99 \
  system.gae_lambda=0.95