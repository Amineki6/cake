"""
runner.py
=========
CAKE experiment orchestrator: teacher loading, sample generation, and student
distillation in a single end-to-end run (or WandB sweep).

Usage
-----
Run directly::

    python folder_second_stage/runner.py            # single run
    python folder_second_stage/runner.py --sweep    # Bayesian sweep (10 trials)
    python folder_second_stage/runner.py --sweep-id <ID> --sweep-count 20
    python folder_second_stage/runner.py --sweep --group my_experiment_name

The entry point reads ``config.yaml`` from the working directory, optionally
patches hyperparameters from a WandB sweep agent, and calls
:func:`run_experiment`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

import torch
import wandb
from coolname import generate_slug
from omegaconf import OmegaConf

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from folder_first_stage.autoencoder import Autoencoder
from folder_second_stage.sampling import generate_samples, filter_empty_samples
from folder_third_stage.train_student import train_student_distillation
from folder_third_stage.evaluate_student import evaluate_student

# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

#: Default path to pre-trained teacher weights.
_DEFAULT_WEIGHTS_PATH: str = "weights/mnist_4class_autoencoder.pth"

#: Standard MNIST pixel shape.
_MNIST_SHAPE = (1, 28, 28)

# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RunConfig:
    """Hyperparameters for a single experiment run."""

    lr: float
    weight_cls: float
    weight_contr: float
    weight_tv: float

    @classmethod
    def from_defaults(cls) -> "_RunConfig":
        """Construct a :class:`_RunConfig` by reading baseline values from config.yaml."""
        cfg = OmegaConf.load("config.yaml")
        return cls(
            lr=float(OmegaConf.select(cfg, "sampling.lr")),
            weight_cls=float(OmegaConf.select(cfg, "sampling.weight.cls")),
            weight_contr=float(OmegaConf.select(cfg, "sampling.weight.contr")),
            weight_tv=float(OmegaConf.select(cfg, "sampling.weight.tv")),
        )

    @classmethod
    def from_wandb(cls, sweep_cfg: Any) -> "_RunConfig":
        """Construct a :class:`_RunConfig` from a WandB sweep config object."""
        defaults = cls.from_defaults()
        return cls(
            lr=float(getattr(sweep_cfg, "sampling_lr", defaults.lr)),
            weight_cls=float(getattr(sweep_cfg, "sampling_weight_cls", defaults.weight_cls)),
            weight_contr=float(getattr(sweep_cfg, "sampling_weight_contr", defaults.weight_contr)),
            weight_tv=float(getattr(sweep_cfg, "sampling_weight_tv", defaults.weight_tv)),
        )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _base_cfg() -> Any:
    """Load ``config.yaml`` and apply mandatory runtime defaults."""
    cfg = OmegaConf.load("config.yaml")
    OmegaConf.update(cfg, "env.data_dir", "./data", merge=True)
    OmegaConf.update(cfg, "env.results_dir", "./results", merge=True)
    OmegaConf.update(cfg, "data.dataset", "MNIST", merge=True)
    OmegaConf.update(cfg, "env.profiler", "simple", merge=True)
    OmegaConf.update(cfg, "env.seed", 42, merge=True)
    OmegaConf.update(cfg, "env.tag", "local_test", merge=True)
    OmegaConf.update(cfg, "env.group_tag", "dev", merge=True)
    OmegaConf.update(cfg, "env.notes", "Testing autoencoder sampling", merge=True)
    return cfg


def _patch_cfg(cfg: Any, rcfg: _RunConfig) -> Any:
    """Overlay experiment hyperparameters onto *cfg* in-place."""
    OmegaConf.update(cfg, "sampling.lr", rcfg.lr, merge=True)
    OmegaConf.update(cfg, "sampling.weight.cls", rcfg.weight_cls, merge=True)
    OmegaConf.update(cfg, "sampling.weight.contr", rcfg.weight_contr, merge=True)
    OmegaConf.update(cfg, "sampling.weight.tv", rcfg.weight_tv, merge=True)
    OmegaConf.update(cfg, "sampling.weight.entropy", 0.0, merge=True)
    return cfg


# ---------------------------------------------------------------------------
# Teacher loading
# ---------------------------------------------------------------------------


def _load_teacher(
    device: torch.device,
    weights_path: str = _DEFAULT_WEIGHTS_PATH,
) -> Autoencoder:
    """Instantiate and optionally load a pre-trained :class:`Autoencoder`."""
    model = Autoencoder().to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        logger.info("Loaded teacher weights from %s", weights_path)
    else:
        logger.warning("No weights found at %s — using random initialisation.", weights_path)
    return model


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_experiment(
    run_id: str,
    rcfg: _RunConfig,
    device: torch.device,
    logger_wandb: Any,
    mode: str = "standard",
    num_iters: int = 5,
    filter_empty: bool = False,
    filter_threshold: float = 1e-6,
) -> float:
    """Execute one full CAKE experiment and return the evaluation metric."""
    cfg = _patch_cfg(_base_cfg(), rcfg)

    model_teacher = _load_teacher(device)

    samples_dir = f"./results/{run_id}/samples_output"
    student_weights_path = f"./weights/student_{run_id}.pth"
    os.makedirs(samples_dir, exist_ok=True)

    logger.info("Generating distillation samples — run_id=%s", run_id)
    dataset = generate_samples(
        model_teacher=model_teacher,
        shape=_MNIST_SHAPE,
        cfg=cfg,
        device=device,
        logger_wandb=logger_wandb,
        samples_dir=samples_dir,
        mode=mode,
        num_iters=num_iters,
    )
    logger.info("Sampling complete.  Archive: %s", dataset.archive)

    if filter_empty:
        kept, removed = filter_empty_samples(dataset.archive, threshold=filter_threshold)
        logger.info(
            "Empty-sample filter: kept=%d  removed=%d  (threshold=%.2e)",
            kept, removed, filter_threshold,
        )

    logger.info("Training student on distilled samples.")
    student_model = train_student_distillation(
        data_path=dataset.archive,
        student_weights_path=student_weights_path,
    )

    logger.info("Evaluating student vs. teacher on real MNIST test set.")
    foreground_miou = evaluate_student(student_model, model_teacher, device)
    logger.info("Foreground mIoU (teacher–student): %.4f", foreground_miou)

    return float(foreground_miou)


# ---------------------------------------------------------------------------
# WandB sweep / single-run wrapper
# ---------------------------------------------------------------------------

_run_counter = [0]  # mutable so the lambda closure can increment it


def _train_sweep(group_name: str, is_sweep: bool = False, mode: str = "standard", num_iters: int = 5, filter_empty: bool = False, filter_threshold: float = 1e-6) -> None:
    """Initialise WandB, resolve hyperparameters, and call :func:`run_experiment`."""
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)

    _run_counter[0] += 1
    run_name = f"{group_name}-{_run_counter[0]}"

    wandb.init(project="cake_distillation", group=group_name, name=run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running on device: %s", device)

    rcfg = _RunConfig.from_wandb(wandb.config) if is_sweep else _RunConfig.from_defaults()

    if mode == "iter_sweep":
        iters_to_try = [1, 2, 5, 10, 20, 50, 100]
        logger.info("Starting iteration sweep: %s", iters_to_try)
        for iters in iters_to_try:
            logger.info("--- Iteration Sweep: iters=%d ---", iters)
            miou = run_experiment(
                run_id=f"{run_name}-it{iters}",
                rcfg=rcfg,
                device=device,
                logger_wandb=wandb,
                mode="iterations",
                num_iters=iters,
                filter_empty=filter_empty,
                filter_threshold=filter_threshold,
            )
            wandb.log({"iterations": iters, "foreground_miou": miou})
    else:
        foreground_miou = run_experiment(
            run_id=run_name,
            rcfg=rcfg,
            device=device,
            logger_wandb=wandb,
            mode=mode,
            num_iters=num_iters,
            filter_empty=filter_empty,
            filter_threshold=filter_threshold,
        )

        wandb.log({"foreground_miou": foreground_miou})
        wandb.run.summary.update({"foreground_miou": foreground_miou})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "foreground_miou", "goal": "maximize"},
    "parameters": {
        "sampling_lr": {
            "distribution": "uniform",
            "min": 0.01,
            "max": 3.0
        },
        "sampling_weight_cls": {
            "distribution": "uniform",
            "min": 1.0,
            "max": 20.0
        },
        "sampling_weight_contr": {
            "distribution": "log_uniform_values",
            "min": 0.00001,
            "max": 0.5
        },
        "sampling_weight_tv": {
            "distribution": "uniform",
            "min": 100.0,
            "max": 2000.0
        },
    },
}


def main() -> None:
    """Parse CLI arguments and launch a single run or a WandB sweep."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run CAKE sample generation")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run hyperparameter optimisation sweep with WandB.",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        help="Provide an existing WandB sweep ID to join an ongoing study.",
    )
    parser.add_argument(
        "--group",
        type=str,
        help="Explicitly set a name for the whole group of study.",
    )
    parser.add_argument(
        "--sweep-count",
        type=int,
        default=10,
        help="Number of sweep iterations to run.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=["standard", "shuffled_targets", "shifted", "iterations", "iter_sweep"],
        help=(
            "Sampling mode: 'standard' uses initialised labels; "
            "'shuffled_targets' derives targets from a batch-shuffled initial teacher prediction; "
            "'shifted' derives targets by rolling the initial teacher prediction 2 pixels right; "
            "'iterations' replaces batch_x by iterating noise through the teacher (no optimisation); "
            "'iter_sweep' tries multiple iteration values for the 'iterations' mode."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of teacher-pass iterations for --mode iterations (default: 50).",
    )
    parser.add_argument(
        "--filter-empty",
        action="store_true",
        help="Remove all-black samples from the archive before student training.",
    )
    parser.add_argument(
        "--filter-threshold",
        type=float,
        default=1e-6,
        help="Pixel-value threshold below which a sample is considered all-black (default: 1e-6).",
    )
    args = parser.parse_args()

    is_sweep_run = args.sweep or args.sweep_id is not None

    # Establish a persistent group name
    study_group = args.group or generate_slug(2)
    if is_sweep_run and not args.group:
        study_group = f"sweep-{study_group}"

    logger.info("Run group mapping: %s | mode: %s", study_group, args.mode)

    if is_sweep_run:
        sweep_id = args.sweep_id
        if not sweep_id:
            sweep_id = wandb.sweep(_SWEEP_CONFIG, project="cake_distillation")
            logger.info("Created new WandB sweep: %s", sweep_id)
        else:
            logger.info("Joining existing WandB sweep: %s", sweep_id)

        wandb.agent(
            sweep_id,
            function=lambda: _train_sweep(study_group, is_sweep=True, mode=args.mode, num_iters=args.iterations, filter_empty=args.filter_empty, filter_threshold=args.filter_threshold),
            count=args.sweep_count,
        )
    else:
        _train_sweep(study_group, is_sweep=False, mode=args.mode, num_iters=args.iterations, filter_empty=args.filter_empty, filter_threshold=args.filter_threshold)


if __name__ == "__main__":
    main()