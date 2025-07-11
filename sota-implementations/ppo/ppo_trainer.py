import pathlib
from typing import Any, Callable, Literal  # Added Any

import hydra
import torch
from tensordict import TensorDictBase  # Added TensorDictBase
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule

from torchrl.collectors.collectors import DataCollectorBase, SyncDataCollector
from torchrl.collectors.distributed.rpc import RPCDataCollector
from torchrl.envs import (
    ClipTransform,
    DoubleToFloat,
    ExplorationType,
    RewardSum,
    StepCounter,
    TransformedEnv,
    VecNorm,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import group_optimizers
from torchrl.objectives.ppo import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
from torchrl.record.loggers import Logger
from torchrl.trainers import (
    BatchSubSampler,
    ClearCudaCache,
    LogScalar,
    LogValidationReward,
    OptimizerHook,
    RewardNormalizer,
    Trainer,
    TrainerHookBase,
)


class GAEHook(TrainerHookBase):
    def __init__(self, adv_module: GAE):
        super().__init__()
        self.adv_module = adv_module

    @torch.no_grad()
    def __call__(self, batch: TensorDictBase) -> TensorDictBase:
        return self.adv_module(batch)

    def state_dict(self) -> dict[str, Any]:
        return {"adv_module": self.adv_module.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.adv_module.load_state_dict(state_dict["adv_module"])

    def register(self, trainer: Trainer, name: str = "gae_hook"):
        trainer.register_op(dest="batch_process", op=self)
        trainer.register_module(name, self)


class PPOAnnealingHook(TrainerHookBase): ...


# Maybe need hooks for:
# - buffer
# - optimization hook that takes care of sampling, LR/Clip annealing?
# - an eval hook?


class PPOTrainer(Trainer):
    def __init__(
        self,
        *,
        total_frames: int,
        frame_skip: int,
        frames_per_batch: int,
        create_env_fn: Callable,
        device: str,
        # Model parameters
        actor_model: torch.nn.Module | None = None,
        critic_model: torch.nn.Module | None = None,
        # Loss module parameters
        loss_module: ClipPPOLoss | None = None,
        clip_epsilon: float = 0.2,  # Defaults if constructing internally
        loss_critic_type: str = "l2",
        entropy_coef: float = 0.01,
        critic_coef: float = 1.0,
        normalize_advantage: bool = True,
        loss_kwargs: dict | None = None,  # Optional: for other ClipPPOLoss params
        # optimizer
        optimizer: torch.optim.Optimizer | None = None,
        lr: float = 3e-4,  # Default if constructing internally
        optimizer_kwargs: dict | None = None,  # Optional: for other optimizer params
        #
        optim_steps_per_batch: int,
        create_env_kwargs: dict | None = None,
        max_frames_per_traj: int = -1,  # -1 means no limit
        collector: DataCollectorBase | None = None,
        collector_type: Literal["sync", "rpc"] = "sync",
        collector_kwargs: dict | None = None,
        # GAE specific parameters
        gae_gamma: float = 0.99,
        gae_lam: float = 0.95,
        gae_kwargs: dict | None = None,
        #
        logger: Logger | None = None,
        clip_grad_norm: bool = True,
        clip_norm: float | None = None,
        progress_bar: bool = True,
        seed: int | None = None,
        save_trainer_interval: int = 10000,
        log_interval: int = 10000,
        save_trainer_file: str | pathlib.Path | None = None,
    ):
        _device = torch.device(device)

        # TODO: abstract this into a helper
        if actor_model is None or critic_model is None:
            # This case should ideally be handled by creating default models
            # or raising an error if they are essential and not provided.
            # For now, we assume they will be provided or handled by a subsequent step/method.
            pass
        else:
            actor_model = actor_model.to(_device)
            critic_model = critic_model.to(_device)

        # TODO: abstract this into a helper
        _collector_kwargs = collector_kwargs or {}
        _create_env_kwargs = create_env_kwargs or {}
        _actual_collector: DataCollectorBase
        if collector is not None:
            _actual_collector = collector
        else:
            common_args_for_collector_construction = {
                "policy": actor_model,
                "frames_per_batch": frames_per_batch,
                "total_frames": total_frames,
                "device": _device,
                "max_frames_per_traj": max_frames_per_traj,
            }

            if collector_type == "sync":
                _actual_collector = SyncDataCollector(
                    create_env_fn=create_env_fn,
                    create_env_kwargs=_create_env_kwargs,
                    **common_args_for_collector_construction,
                    **_collector_kwargs,  # e.g., num_workers for ParallelEnv setup within SyncDataCollector
                )
            elif collector_type == "rpc":
                _actual_collector = RPCDataCollector(
                    create_env_fn=...,
                    create_env_kwargs=...,
                    **common_args_for_collector_construction,
                    **_collector_kwargs,
                )
            else:
                raise ValueError(f"Unsupported collector_type: {collector_type}")

        _actual_loss_module: ClipPPOLoss
        if loss_module is not None:
            _actual_loss_module = loss_module
            # Assuming ClipPPOLoss does not need a .to(_device) call itself,
            # as its internal networks (actor_model, critic_model) are already on the device.
        else:
            if actor_model is None or critic_model is None:
                raise ValueError(
                    "actor_model and critic_model must be provided if loss_module is not."
                )
            _current_loss_kwargs = {
                "clip_epsilon": clip_epsilon,
                "loss_critic_type": loss_critic_type,
                "entropy_coef": entropy_coef,
                "critic_coef": critic_coef,
                "normalize_advantage": normalize_advantage,
            }
            if loss_kwargs:
                _current_loss_kwargs.update(loss_kwargs)

            _actual_loss_module = ClipPPOLoss(
                actor_network=actor_model,
                critic_network=critic_model,
                **_current_loss_kwargs,
            )

        _actual_optimizer: torch.optim.Optimizer
        if optimizer is not None:
            _actual_optimizer = optimizer
            # Optimizers don't have a .to(device) method.
            # Parameters should be on the correct device when the optimizer is created.
        else:
            if actor_model is None or critic_model is None:
                raise ValueError(
                    "actor_model and critic_model must be provided if optimizer is not."
                )

            _opt_kwargs = {"lr": torch.tensor(lr, device=_device)}
            # Default Adam epsilon, can be overridden by optimizer_kwargs
            _default_adam_eps = 1e-5
            if optimizer_kwargs:
                # Ensure lr from optimizer_kwargs doesn't conflict if also passed directly
                # For simplicity, direct lr parameter takes precedence if optimizer_kwargs also has 'lr'
                # User should provide lr via the main parameter or within optimizer_kwargs, not both.
                _processed_opt_kwargs = {
                    k: v for k, v in optimizer_kwargs.items() if k != "lr"
                }
                _opt_kwargs.update(_processed_opt_kwargs)

            actor_params = list(actor_model.parameters())
            critic_params = list(critic_model.parameters())

            if not actor_params:
                # This can happen if actor_model is a functional module without parameters
                # or if it's not properly initialized.
                # Depending on the design, this might be an error or require different handling.
                # For now, we'll assume it's an error if no parameters are found for optimization.
                raise ValueError("actor_model has no parameters to optimize.")
            if not critic_params:
                raise ValueError("critic_model has no parameters to optimize.")

            actor_optim = torch.optim.Adam(
                actor_params,
                lr=_opt_kwargs.pop("lr"),  # lr is taken from _opt_kwargs
                eps=_opt_kwargs.pop(
                    "eps", _default_adam_eps
                ),  # eps from optimizer_kwargs or default
                **_opt_kwargs,  # pass remaining optimizer_kwargs
            )
            # Re-add lr for critic optimizer, or handle separate lr for critic if needed
            _opt_kwargs_critic = {"lr": torch.tensor(lr, device=_device)}
            if optimizer_kwargs:
                _processed_opt_kwargs_critic = {
                    k: v for k, v in optimizer_kwargs.items() if k != "lr"
                }
                _opt_kwargs_critic.update(_processed_opt_kwargs_critic)

            critic_optim = torch.optim.Adam(
                critic_params,
                lr=_opt_kwargs_critic.pop("lr"),
                eps=_opt_kwargs_critic.pop("eps", _default_adam_eps),
                **_opt_kwargs_critic,
            )
            _actual_optimizer = group_optimizers(actor_optim, critic_optim)
            del actor_optim, critic_optim

        super().__init__(
            collector=_actual_collector,
            total_frames=total_frames,
            frame_skip=frame_skip,
            optim_steps_per_batch=optim_steps_per_batch,
            loss_module=_actual_loss_module,
            optimizer=_actual_optimizer,
            logger=logger,
            clip_grad_norm=clip_grad_norm,
            clip_norm=clip_norm,
            progress_bar=progress_bar,
            seed=seed,
            save_trainer_interval=save_trainer_interval,
            log_interval=log_interval,
            save_trainer_file=save_trainer_file,
        )

        # Instantiate and register GAEHook
        # It's possible we first need to add a flattening hook to the collector
        # to ensure the batch is in the right format for GAE.
        # adv_module = GAE(...)  # Use this for the gae hook
        # gae_hook = GAEHook(gamma=gae_gamma, lam=gae_lam, gae_kwargs=gae_kwargs)
        # gae_hook.register(self)
        # batch_subsampler = BatchSubSampler(
        #     batch_size=...,
        #     sub_traj_len=...,
        #     min_sub_traj_len=...,
        # )
        # batch_subsampler.register(self)


# ====================================================================
# TO DELETE LATER
# --------------------------------------------------------------------


def make_env(env_name="HalfCheetah-v4", device="cpu", from_pixels: bool = False):
    env = GymEnv(env_name, device=device, from_pixels=from_pixels, pixels_only=False)
    env = TransformedEnv(env)
    env.append_transform(VecNorm(in_keys=["observation"], decay=0.99999, eps=1e-2))
    env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    env.append_transform(DoubleToFloat(in_keys=["observation"]))
    return env


def make_ppo_models_state(proof_environment, device):
    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape

    # Define policy output distribution class
    num_outputs = proof_environment.action_spec_unbatched.shape[-1]
    distribution_class = TanhNormal
    distribution_kwargs = {
        "low": proof_environment.action_spec_unbatched.space.low.to(device),
        "high": proof_environment.action_spec_unbatched.space.high.to(device),
        "tanh_loc": False,
    }

    # Define policy architecture
    policy_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Tanh,
        out_features=num_outputs,  # predict only loc
        num_cells=[64, 64],
        device=device,
    )

    # Initialize policy weights
    for layer in policy_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()

    # Add state-independent normal scale
    policy_mlp = torch.nn.Sequential(
        policy_mlp,
        AddStateIndependentNormalScale(
            proof_environment.action_spec_unbatched.shape[-1], scale_lb=1e-8
        ).to(device),
    )

    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        TensorDictModule(
            module=policy_mlp,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        ),
        in_keys=["loc", "scale"],
        spec=proof_environment.full_action_spec_unbatched.to(device),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Define value architecture
    value_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Tanh,
        out_features=1,
        num_cells=[64, 64],
        device=device,
    )

    # Initialize value weights
    for layer in value_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()

    # Define value module
    value_module = ValueOperator(
        value_mlp,
        in_keys=["observation"],
    )

    return policy_module, value_module


def make_ppo_models(env_name, device):
    proof_environment = make_env(env_name, device=device)
    actor, critic = make_ppo_models_state(proof_environment, device=device)
    return actor, critic


@hydra.main(config_path="", config_name="config_mujoco", version_base="1.1")
def main(cfg):  # noqa: F821
    device = cfg.optim.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)

    actor, critic = make_ppo_models(cfg.env.env_name, device=device)

    trainer = PPOTrainer(
        total_frames=cfg.collector.total_frames,
        frame_skip=1,
        frames_per_batch=cfg.collector.frames_per_batch,
        device="cpu",
        clip_epsilon=cfg.loss.clip_epsilon,
        optim_steps_per_batch=1,  # TODO this should be ppo specific epoch something
        loss_critic_type=cfg.loss.loss_critic_type,
        entropy_coef=cfg.loss.entropy_coef,
        critic_coef=cfg.loss.critic_coef,
        normalize_advantage=True,
        actor_model=actor,
        critic_model=critic,
        create_env_fn=make_env(cfg.env.env_name, device),
        lr=1e-4,
    )
    print("WAGWAN")


if __name__ == "__main__":
    main()
