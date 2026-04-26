"""Record a rollout video for an SPO policy on continuous-control envs (brax).

Examples
--------
Before training (random init):
    python record_video_continuous.py env=brax/halfcheetah \
        +video.output=videos/halfcheetah_before.mp4

After training (load checkpoint):
    python record_video_continuous.py env=brax/halfcheetah \
        logger.checkpointing.load_model=true \
        logger.checkpointing.load_args.checkpoint_uid=halfcheetah_run \
        +video.output=videos/halfcheetah_after.mp4
"""

import os
import time as _time

import hydra
import imageio
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

from stoix.base_types import OnlineAndTarget
from stoix.networks.base import FeedForwardActor as Actor
from stoix.networks.base import FeedForwardCritic as Critic
from stoix.systems.mpo.mpo_types import DualParams
from stoix.systems.spo.ff_spo_continuous import SPO, make_recurrent_fn, make_root_fn
from stoix.systems.spo.spo_types import SPOParams
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer


@hydra.main(
    config_path="stoix/configs/default/anakin",
    config_name="default_ff_spo_continuous",
    version_base="1.2",
)
def main(config: DictConfig) -> None:
    OmegaConf.set_struct(config, False)

    video_cfg = config.get("video", {})
    output = video_cfg.get("output", "videos/rollout.mp4")
    max_steps = int(video_cfg.get("max_steps", 1000))
    fps = int(video_cfg.get("fps", 30))
    seed = int(video_cfg.get("seed", 0))
    width = int(video_cfg.get("width", 320))
    height = int(video_cfg.get("height", 240))
    use_search = bool(
        video_cfg.get("use_search", config.logger.checkpointing.load_model)
    )

    # Build envs.
    _, eval_env = environments.make(config=config)

    env_suite = config.env.env_name
    if env_suite != "brax":
        raise NotImplementedError(
            f"record_video_continuous.py only supports brax envs (got '{env_suite}')."
        )

    from brax.envs import create as brax_make
    from brax.io import image as brax_image

    raw_env = brax_make(config.env.scenario.name, auto_reset=False, **config.env.kwargs)

    # Action space metadata required by the continuous SPO networks.
    action_space = eval_env.action_space()
    action_dim = int(action_space.shape[-1])
    config.system.action_dim = action_dim
    config.system.action_minimum = float(action_space.minimum)
    config.system.action_maximum = float(action_space.maximum)

    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head,
        action_dim=action_dim,
        minimum=config.system.action_minimum,
        maximum=config.system.action_maximum,
    )
    actor_network = Actor(torso=actor_torso, action_head=actor_action_head)

    critic_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    critic_head = hydra.utils.instantiate(config.network.critic_network.critic_head)
    critic_network = Critic(torso=critic_torso, critic_head=critic_head)

    key = jax.random.PRNGKey(seed)
    key, actor_init_key, critic_init_key, reset_key, act_key = jax.random.split(key, 5)

    init_x = eval_env.observation_space().generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)

    actor_params = actor_network.init(actor_init_key, init_x)
    critic_params = critic_network.init(critic_init_key, init_x)

    if config.system.per_dim_constraining:
        dual_variable_shape = [action_dim]
    else:
        dual_variable_shape = [1]

    dual_params = DualParams(
        log_temperature=jnp.full([1], config.system.init_log_temperature, dtype=jnp.float32),
        log_alpha_mean=jnp.full(
            dual_variable_shape, config.system.init_log_alpha_mean, dtype=jnp.float32
        ),
        log_alpha_stddev=jnp.full(
            dual_variable_shape, config.system.init_log_alpha_stddev, dtype=jnp.float32
        ),
    )
    params = SPOParams(
        OnlineAndTarget(actor_params, actor_params),
        OnlineAndTarget(critic_params, critic_params),
        dual_params,
    )

    if config.logger.checkpointing.load_model:
        loaded = Checkpointer(
            model_name=config.system.system_name,
            **config.logger.checkpointing.load_args,
        )
        params, _ = loaded.restore_params(input_params=params)
        print(f"Loaded checkpoint: {config.logger.checkpointing.load_args.checkpoint_uid}")
    else:
        print("Using randomly initialised actor (before training).")

    online_actor_params = params.actor_params.online

    @jax.jit
    def actor_only_step(actor_params_, obs, k):
        pi = actor_network.apply(actor_params_, obs[None, ...])
        action = pi.sample(seed=k)
        return action[0]

    if use_search:
        print("Building SMC search (matches training-time evaluation)...", flush=True)
        root_fn = make_root_fn(actor_network.apply, critic_network.apply, config)
        model_recurrent_fn = make_recurrent_fn(
            jax.vmap(eval_env.step),
            actor_network.apply,
            critic_network.apply,
            config,
        )
        search_method = SPO(config, recurrent_fn=model_recurrent_fn)
        search_apply_fn = search_method.search

        @jax.jit
        def search_step(params_, obs, env_state_, k):
            root_key, policy_key = jax.random.split(k)
            obs_b, state_b = jax.tree_util.tree_map(
                lambda x: x[jnp.newaxis, ...], (obs, env_state_)
            )
            root = root_fn(params_, obs_b, state_b, root_key)
            search_output = search_apply_fn(params_, policy_key, root)
            return search_output.action[0]

        def select_action(obs, env_state_, k):
            return search_step(params, obs, env_state_, k)
    else:
        def select_action(obs, env_state_, k):
            return actor_only_step(online_actor_params, obs, k)

    def _index_state(s, i):
        return jax.tree_util.tree_map(lambda x: x[i], s)

    # Fast on-device rollout: scan over max_steps.
    def scan_step(carry, _):
        env_state_, timestep_, key_ = carry
        key_, sub_ = jax.random.split(key_)
        action_ = select_action(timestep_.observation, env_state_, sub_)
        new_env_state, new_timestep = eval_env.step(env_state_, action_)
        return (new_env_state, new_timestep, key_), (
            new_env_state,
            new_timestep.reward,
            new_timestep.last(),
        )

    @jax.jit
    def run_rollout(init_env_state, init_timestep, init_key):
        _, (states, rewards, dones) = jax.lax.scan(
            scan_step,
            (init_env_state, init_timestep, init_key),
            None,
            length=max_steps,
        )
        return states, rewards, dones

    print("Resetting env...", flush=True)
    env_state, timestep = eval_env.reset(reset_key)
    print("Compiling on-device rollout (first call may take a few minutes)...", flush=True)
    t0 = _time.time()
    states, rewards, dones = run_rollout(env_state, timestep, act_key)
    jax.block_until_ready(rewards)
    print(f"Rollout done in {_time.time() - t0:.1f}s. Rendering frames...", flush=True)

    rewards_np = jax.device_get(rewards)
    dones_np = jax.device_get(dones)
    end_idx = int(dones_np.argmax()) if bool(dones_np.any()) else int(max_steps - 1)
    n_frames = end_idx + 1
    total_reward = float(rewards_np[: n_frames].sum())
    print(
        f"Episode length: {n_frames} | terminated: {bool(dones_np.any())} | "
        f"return: {total_reward:.3f}",
        flush=True,
    )

    states_host = jax.device_get(states)

    # Brax rendering uses pipeline_state of each timestep.
    pipeline_states = [env_state.pipeline_state]
    for i in range(n_frames):
        pipeline_states.append(_index_state(states_host, i).pipeline_state)

    # Convert to plain numpy structures (mujoco renderer expects host arrays).
    def _to_np(s):
        return jax.tree_util.tree_map(lambda x: np.asarray(x), s)

    pipeline_states_np = [_to_np(s) for s in pipeline_states]

    print(f"Rendering {len(pipeline_states_np)} frames with mujoco...", flush=True)
    frames = brax_image.render_array(
        raw_env.sys, pipeline_states_np, height=height, width=width
    )

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    imageio.mimsave(output, frames, fps=fps)
    print(
        f"Saved {len(frames)} frames -> {output} | episode_return={total_reward:.3f}"
    )


if __name__ == "__main__":
    main()
