import os
import pickle
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Type

import jax
import numpy as np
from chex import Numeric
from omegaconf import DictConfig, OmegaConf

from stoix.base_types import StoixState

CHECKPOINTER_VERSION = 2.0


class Checkpointer:
    """Model checkpointer for saving and restoring the `learner_state`."""

    def __init__(
        self,
        model_name: str,
        metadata: Optional[Dict] = None,
        rel_dir: str = "checkpoints",
        checkpoint_uid: Optional[str] = None,
        save_interval_steps: int = 1,
        max_to_keep: Optional[int] = 1,
        keep_period: Optional[int] = None,
    ):
        """Initialise the checkpointer tool

        Args:
            model_name (str): Name of the model to be saved.
            metadata (Optional[Dict], optional):
                For storing model metadata. Defaults to None.
            rel_dir (str, optional):
                Relative directory of checkpoints. Defaults to "checkpoints".
            checkpoint_uid (Optional[str], optional):
                Set the uniqiue id of the checkpointer, rel_dir/model_name/checkpoint_uid/...
                If not given, the timestamp is used.
            save_interval_steps (int, optional):
                The interval at which checkpoints should be saved. Defaults to 1.
            max_to_keep (Optional[int], optional):
                Maximum number of checkpoints to keep. Defaults to 1.
            keep_period (Optional[int], optional):
                If set, will not delete any checkpoint where
                checkpoint_step % keep_period == 0. Defaults to None.

        """
        checkpoint_str = (
            checkpoint_uid if checkpoint_uid else datetime.now().strftime("%Y%m%d%H%M%S")
        )
        if os.path.isabs(rel_dir):
            self._directory = os.path.join(rel_dir, model_name, checkpoint_str)
        else:
            self._directory = os.path.join(os.getcwd(), rel_dir, model_name, checkpoint_str)
        os.makedirs(self._directory, exist_ok=True)

        if metadata is not None and isinstance(metadata, DictConfig):
            metadata = OmegaConf.to_container(metadata, resolve=True)
        self._metadata = {
            "checkpointer_version": CHECKPOINTER_VERSION,
            **(metadata if isinstance(metadata, dict) else {}),
        }

        self._max_to_keep = max_to_keep
        self._save_interval_steps = save_interval_steps
        self._keep_period = keep_period
        self._best_metric: Optional[float] = None
        self._best_step: Optional[int] = None
        self._last_save_step: Optional[int] = None

    def save(
        self,
        timestep: int,
        unreplicated_learner_state: StoixState,
        episode_return: Numeric = 0.0,
    ) -> bool:
        """Save the learner state.

        Args:
            timestep (int):
                timestep at which the state is being saved.
            unreplicated_learner_state (StoixState)
                a Stoix LearnerState (must be unreplicated)
            episode_return (Numeric, optional):
                Optional value to determine whether this is the 'best' model to save.
                Defaults to 0.0.

        Returns:
            bool: whether the saving was successful.
        """
        if (
            self._last_save_step is not None
            and (timestep - self._last_save_step) < self._save_interval_steps
        ):
            return False

        episode_return_f = float(episode_return)
        # Always save "latest" so restore works even if no metric improvement.
        latest_path = os.path.join(self._directory, "latest.pkl")
        payload = {
            "step": int(timestep),
            "episode_return": episode_return_f,
            "metadata": self._metadata,
            "learner_state": _to_numpy_pytree(unreplicated_learner_state),
        }
        with open(latest_path, "wb") as f:
            pickle.dump(payload, f)

        # Track best by episode_return (max).
        if self._best_metric is None or episode_return_f > self._best_metric:
            self._best_metric = episode_return_f
            self._best_step = int(timestep)
            best_path = os.path.join(self._directory, "best.pkl")
            with open(best_path, "wb") as f:
                pickle.dump(payload, f)

        self._last_save_step = int(timestep)
        return True

    def restore_params(
        self,
        input_params: Any,
        timestep: Optional[int] = None,
        restore_hstates: bool = False,
        THiddenState: Optional[Type] = None,  # noqa: N803
    ) -> Tuple[Any, Optional[Any]]:
        """Restore the params and the hidden state (in case of RNNs)

        Args:
            timestep (Optional[int], optional):
                Specific timestep for restoration (of course, only if that timestep exists).
                Defaults to None, in which case the latest step will be used.
            restore_hstates (bool, optional): Whether to restore the hidden states.
                Defaults to False.
            TParams (Type[FrozenDict], optional): Type of the params.
                Defaults to ActorCriticParams.
            THiddenState (Type[HiddenStates], optional): Type of the hidden states.
                Defaults to ActorCriticHiddenStates.

        Returns:
            Tuple[ActorCriticParams,Union[HiddenState, None]]: the restored params and
            hidden states.
        """
        # Prefer best.pkl, fall back to latest.pkl.
        best_path = os.path.join(self._directory, "best.pkl")
        latest_path = os.path.join(self._directory, "latest.pkl")
        path = best_path if os.path.exists(best_path) else latest_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found in {self._directory}")

        with open(path, "rb") as f:
            payload = pickle.load(f)

        assert (payload["metadata"]["checkpointer_version"] // 1) == (
            CHECKPOINTER_VERSION // 1
        ), "Loaded checkpoint was created with a different major version of the checkpointer."

        restored_learner_state = _to_jax_pytree(payload["learner_state"])

        # Support both dataclass/NamedTuple attribute access and dict-style access.
        def _get(obj: Any, key: str) -> Any:
            if hasattr(obj, key):
                return getattr(obj, key)
            return obj[key]

        raw_params = _get(restored_learner_state, "params")
        TParams = type(input_params)  # noqa: N806
        if isinstance(raw_params, TParams):
            restored_params = raw_params
        elif isinstance(raw_params, dict):
            restored_params = TParams(**raw_params)
        else:
            # NamedTuple or similar with the same fields.
            restored_params = TParams(*tuple(raw_params))

        restored_hstates = None
        if restore_hstates and THiddenState is not None:
            raw_h = _get(restored_learner_state, "hstates")
            if isinstance(raw_h, THiddenState):
                restored_hstates = raw_h
            elif isinstance(raw_h, dict):
                restored_hstates = THiddenState(**raw_h)
            else:
                restored_hstates = THiddenState(*tuple(raw_h))

        return restored_params, restored_hstates

    def get_cfg(self) -> DictConfig:
        """Return the metadata of the checkpoint."""
        return DictConfig(self._metadata)


def _to_numpy_pytree(tree_in: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree_in)


def _to_jax_pytree(tree_in: Any) -> Any:
    import jax.numpy as jnp

    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), tree_in)
