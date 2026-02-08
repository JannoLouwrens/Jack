"""
Jack Training

Execute real actions in sandbox, learn from results.
Uses action-conditioned state (only relevant features).
"""

from .phase0_digital import (
    DigitalSandbox,
    ExperienceCollector,
    TransitionBuffer,
    Phase0Trainer,
    Transition,
    ActionState,
    ActionEffect,
)

from .train_transformer import (
    JackTrainer,
    TrainingConfig,
    TransitionDataset,
    run_training,
)

__all__ = [
    # Data collection
    "DigitalSandbox",
    "ExperienceCollector",
    "TransitionBuffer",
    "Phase0Trainer",
    "Transition",
    "ActionState",
    "ActionEffect",
    # Training
    "JackTrainer",
    "TrainingConfig",
    "TransitionDataset",
    "run_training",
]
