"""Record a rollout video for an SPO policy on xland_minigrid environments.

Examples
--------
Before training (random init):
    python record_video.py env=xland_minigrid/door_key_16x16 \
        video.output=videos/doorkey_before.mp4

After training (load checkpoint):
    python record_video.py env=xland_minigrid/door_key_16x16 \
        logger.checkpointing.load_model=true \
        logger.checkpointing.load_args.checkpoint_uid=doorkey_ess03_multinomial \
        video.output=videos/doorkey_after.mp4
"""

import os

import hydra
import imageio
import jax
import jax.numpy as jnp
import xminigrid
from omegaconf import DictConfig, OmegaConf

from stoix.base_types import OnlineAndTarget
from stoix.networks.base import FeedForwardActor as Actor
from stoix.networks.base import FeedForwardCritic as Critic
from stoix.systems.mpo.mpo_types import CategoricalDualParams
from stoix.systems.spo.spo_types import SPOParams
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer


@hydra.main(
    config_path="stoix/configs/default/anakin",
    config_name="default_ff_spo",
    version_base="1.2",
)
def main(config: DictConfig) -> None:
    OmegaConf.set_struct(config, False)

    # Video defaults (override via CLI: video.output=..., video.max_steps=..., video.fps=...).
    video_cfg = config.get("video", {})
    output = video_cfg.get("output", "videos/rollout.mp4")
    max_steps = int(video_cfg.get("max_steps", 600))
    fps = int(video_cfg.get("fps", 10))
    seed = int(video_cfg.get("seed", 0))
    greedy = bool(video_cfg.get("greedy", True))

    # Build envs (eval_env has the same observation pipeline used by the policy).
    _, eval_env = environments.make(config=config)

    # Build a raw xminigrid env purely for RGB rendering.
    raw_env, raw_params = xminigrid.make(config.env.scenario.name, **config.env.kwargs)

    # Instantiate networks (actor used for the rollout; critic only needed to match
    # the SPOParams pytree structure when restoring a checkpoint).
    action_dim = int(eval_env.action_space().num_values)

    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head, action_dim=action_dim
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
    dual_params = CategoricalDualParams(
        log_temperature=jnp.full([1], config.system.init_log_temperature, dtype=jnp.float32),
        log_alpha=jnp.full([1], config.system.init_log_alpha, dtype=jnp.float32),
    )
    params = SPOParams(
        OnlineAndTarget(actor_params, actor_params),
        OnlineAndTarget(critic_params, critic_params),
        dual_params,
    )

    # Optionally restore checkpointed parameters.
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
    def policy_step(actor_params_, obs, k):
        pi = actor_network.apply(actor_params_, obs[None, ...])
        if greedy:
            action = pi.mode()
        else:
            action = pi.sample(seed=k)
        return action[0]

    def _xmg_state(s):
        return getattr(s, "unwrapped_state", s)

    # Roll out one episode and render each step.
    env_state, timestep = eval_env.reset(reset_key)
    frames = [raw_env.render(raw_params, _xmg_state(env_state))]
    total_reward = 0.0
    for step in range(max_steps):
        if bool(timestep.last()):
            break
        act_key, sub = jax.random.split(act_key)
        action = policy_step(online_actor_params, timestep.observation, sub)
        env_state, timestep = eval_env.step(env_state, action)
        frames.append(raw_env.render(raw_params, _xmg_state(env_state)))
        total_reward += float(timestep.reward)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    imageio.mimsave(output, frames, fps=fps)
    print(f"Saved {len(frames)} frames -> {output} | episode_return={total_reward:.3f}")


if __name__ == "__main__":
    main()
