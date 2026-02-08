"""
Jack Brain Training

Collect experiences → Train to predict actions → Done.

Uses action-conditioned state (only relevant features).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from .phase0_digital import (
    DigitalSandbox,
    ExperienceCollector,
    TransitionBuffer,
    Transition,
    ActionState,
    ActionEffect,
)
from ..core.jack_brain import JackBrain, JackConfig, DEFAULT_CONFIG


@dataclass
class TrainingConfig:
    """Training settings"""
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs_per_round: int = 10
    samples_per_round: int = 200
    collection_rounds: int = 10
    history_length: int = 10
    checkpoint_dir: str = "checkpoints"
    save_every: int = 2


class TransitionDataset(Dataset):
    """Dataset using action-conditioned state"""

    def __init__(self, transitions: List[Transition], history_length: int = 10):
        self.transitions = transitions
        self.history_length = history_length

        # Action type mapping
        self.action_map = {
            'shell_run': 0,
            'file_read': 1,
            'file_write': 2,
            'http_request': 3,
            'get_state': 4,
        }

    def __len__(self):
        return len(self.transitions)

    def _encode_target(self, target: str, max_len: int = 64) -> np.ndarray:
        """Encode action target (path/command) as character features"""
        features = np.zeros(max_len)
        for i, char in enumerate(target[:max_len]):
            features[i] = (ord(char) - 32) / 95  # Normalize ASCII
        return features

    def __getitem__(self, idx):
        t = self.transitions[idx]

        # Current state (action-conditioned - only 7 features!)
        state = np.array(t.state_before.to_vector(), dtype=np.float32)

        # Pad state to match config.state_dim (128)
        # Add target encoding to state
        target_enc = self._encode_target(t.action_target)
        state_full = np.zeros(128, dtype=np.float32)
        state_full[:7] = state
        state_full[7:7+64] = target_enc

        # Build history from previous transitions
        history_types = np.zeros(self.history_length, dtype=np.int64)
        history_params = np.zeros((self.history_length, 64), dtype=np.float32)
        history_outcomes = np.zeros(self.history_length, dtype=np.int64)

        start = max(0, idx - self.history_length)
        for i, j in enumerate(range(start, idx)):
            if i >= self.history_length:
                break
            ht = self.transitions[j]
            history_types[i] = self.action_map.get(ht.action_type, 4)
            history_params[i] = self._encode_target(ht.action_target)
            history_outcomes[i] = 1 if ht.effect.success else 0

        # Target: action type
        target_type = self.action_map.get(t.action_type, 4)

        # Target: action params (target path/command encoded)
        target_params = self._encode_target(t.action_target)

        # Target: success
        target_success = 1 if t.effect.success else 0

        return {
            'state': torch.tensor(state_full, dtype=torch.float32),
            'history_types': torch.tensor(history_types, dtype=torch.long),
            'history_params': torch.tensor(history_params, dtype=torch.float32),
            'history_outcomes': torch.tensor(history_outcomes, dtype=torch.long),
            'target_type': torch.tensor(target_type, dtype=torch.long),
            'target_params': torch.tensor(target_params, dtype=torch.float32),
            'target_success': torch.tensor(target_success, dtype=torch.long),
        }


class JackTrainer:
    """Simple trainer for JackBrain"""

    def __init__(
        self,
        model_config: JackConfig = None,
        train_config: TrainingConfig = None,
    ):
        self.model_config = model_config or DEFAULT_CONFIG
        self.train_config = train_config or TrainingConfig()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = JackBrain(self.model_config).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=0.01,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.train_config.collection_rounds * self.train_config.epochs_per_round,
        )

        self.history = {'total_loss': [], 'type_loss': [], 'param_loss': [], 'success_loss': []}
        Path(self.train_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        type_loss = 0
        param_loss = 0
        success_loss = 0
        n_batches = 0

        for batch in dataloader:
            state = batch['state'].to(self.device)
            history_types = batch['history_types'].to(self.device)
            history_params = batch['history_params'].to(self.device)
            history_outcomes = batch['history_outcomes'].to(self.device)
            target_type = batch['target_type'].to(self.device)
            target_params = batch['target_params'].to(self.device)
            target_success = batch['target_success'].to(self.device)

            self.optimizer.zero_grad()
            losses = self.model.compute_loss(
                state, history_types, history_params, history_outcomes,
                target_type, target_params, target_success,
            )

            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += losses['total_loss'].item()
            type_loss += losses['type_loss'].item()
            param_loss += losses['param_loss'].item()
            success_loss += losses['success_loss'].item()
            n_batches += 1

        return {
            'total_loss': total_loss / max(n_batches, 1),
            'type_loss': type_loss / max(n_batches, 1),
            'param_loss': param_loss / max(n_batches, 1),
            'success_loss': success_loss / max(n_batches, 1),
        }

    def train(self, transitions: List[Transition], epochs: int = None, verbose: bool = True):
        """Train on transitions"""
        epochs = epochs or self.train_config.epochs_per_round

        if len(transitions) < self.train_config.batch_size:
            if verbose:
                print(f"  Not enough transitions ({len(transitions)}), skipping")
            return self.history

        dataset = TransitionDataset(transitions, self.train_config.history_length)
        dataloader = DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        if verbose:
            print(f"  Training on {len(transitions)} transitions")

        for epoch in range(epochs):
            metrics = self.train_epoch(dataloader)
            self.scheduler.step()

            for key, value in metrics.items():
                self.history[key].append(value)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs} | Loss: {metrics['total_loss']:.4f}")

        return self.history

    def save(self, path: str = None):
        path = path or f"{self.train_config.checkpoint_dir}/jack_brain.pt"
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'history': self.history,
        }, path)
        print(f"Saved: {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.history = checkpoint.get('history', self.history)
        print(f"Loaded: {path}")


def run_training(
    model_config: JackConfig = None,
    train_config: TrainingConfig = None,
    verbose: bool = True,
):
    """Run training loop"""
    model_config = model_config or DEFAULT_CONFIG
    train_config = train_config or TrainingConfig()

    print("=" * 50)
    print("JACK BRAIN TRAINING")
    print("=" * 50)
    print(f"Brain: {JackBrain(model_config).count_parameters():,} params")
    print(f"Rounds: {train_config.collection_rounds}")
    print(f"Samples/round: {train_config.samples_per_round}")

    sandbox = DigitalSandbox()
    buffer = TransitionBuffer(save_path=f"{train_config.checkpoint_dir}/transitions.json")
    collector = ExperienceCollector(sandbox, buffer)
    trainer = JackTrainer(model_config, train_config)

    print(f"\nSandbox: {sandbox.sandbox_dir}")

    try:
        for round_num in range(train_config.collection_rounds):
            print(f"\n--- Round {round_num + 1}/{train_config.collection_rounds} ---")

            # Collect
            n = train_config.samples_per_round // 4
            collector.collect_file_read_experiences(n)
            collector.collect_file_write_experiences(n)
            collector.collect_shell_experiences(n)
            collector.collect_sequence_experiences(n)
            buffer.save()

            print(f"  Buffer: {len(buffer)} transitions")

            # Train
            trainer.train(buffer.buffer, verbose=verbose)

            # Save
            if (round_num + 1) % train_config.save_every == 0:
                trainer.save()

            # Reset sandbox periodically
            if round_num % 3 == 2:
                sandbox.reset()

        trainer.save(f"{train_config.checkpoint_dir}/jack_final.pt")

        # Stats
        success = sum(1 for t in buffer.buffer if t.effect.success)
        print(f"\n{'=' * 50}")
        print(f"Done. {len(buffer)} transitions, {100*success/len(buffer):.1f}% success")
        print(f"Final loss: {trainer.history['total_loss'][-1]:.4f}")
        print("=" * 50)

        return trainer

    finally:
        sandbox.cleanup()


if __name__ == "__main__":
    run_training()
