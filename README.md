python stoix/systems/spo/ff_spo.py env=xland_minigrid/door_key_16x16 arch.total_num_envs=32 arch.total_timesteps=5000000 system.num_particles=16 system.search_depth=64 system.resampling.mode=ess system.resampling.ess_threshold=0.5 system.total_batch_size=64 system.rollout_length=128 system.actor_lr=0.0003 system.critic_lr=0.0003 system.gamma=0.99 system.gae_lambda=0.95

cat > stoix/configs/env/xland_minigrid/door_key_8x8.yaml << 'EOF'
env_name: xland_minigrid
scenario:
  name: MiniGrid-DoorKey-8x8
  task_name: minigrid_doorkey_8x8
kwargs: {}
eval_metric: episode_return
wrapper:
  _target_: stoa.FlattenObservationWrapper
EOF

cat > stoix/configs/env/xland_minigrid/door_key_16x16.yaml << 'EOF'
env_name: xland_minigrid
scenario:
  name: MiniGrid-DoorKey-16x16
  task_name: minigrid_doorkey_16x16
kwargs: {}
eval_metric: episode_return
wrapper:
  _target_: stoa.FlattenObservationWrapper
EOF

cat > stoix/configs/env/xland_minigrid/fourrooms.yaml << 'EOF'
env_name: xland_minigrid
scenario:
  name: MiniGrid-FourRooms
  task_name: minigrid_fourrooms
kwargs: {}
eval_metric: episode_return
wrapper:
  _target_: stoa.FlattenObservationWrapper
EOF

cat > stoix/configs/env/xland_minigrid/empty_16x16.yaml << 'EOF'
env_name: xland_minigrid
scenario:
  name: MiniGrid-Empty-16x16
  task_name: minigrid_empty_16x16
kwargs: {}
eval_metric: episode_return
wrapper:
  _target_: stoa.FlattenObservationWrapper
EOF

sed -i 's/^task_name:/  task_name:/' stoix/configs/env/xland_minigrid/door_key_8x8.yaml 

sed -i 's/^task_name:/  task_name:/' stoix/configs/env/xland_minigrid/door_key_16x16.yaml 

sed -i 's/^task_name:/  task_name:/' stoix/configs/env/xland_minigrid/fourrooms.yaml 

sed -i 's/^task_name:/  task_name:/' stoix/configs/env/xland_minigrid/empty_16x16.yaml