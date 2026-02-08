"""
JACK BRAIN - Simple and Practical

What Jack actually does:
- shell_run: Execute commands
- file_read: Read files
- file_write: Write files
- http_request: Make HTTP calls
- get_state: Observe system

What Jack needs:
- Encode current state
- Remember action history
- Predict next action

That's it. No:
- GUI vision (Jack is CLI-based)
- Flow Matching (that's for robot arms)
- World Model imagination (digital is deterministic)
- Fancy System 1/2 (if stuck, ask LLM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class JackConfig:
    """Minimal config for practical use"""
    # Model
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1

    # State
    state_dim: int = 128
    max_history: int = 10

    # Actions (Jack's 5 primitives)
    n_action_types: int = 5
    action_param_dim: int = 64

    # Training
    learning_rate: float = 1e-4
    batch_size: int = 32


DEFAULT_CONFIG = JackConfig()


class JackBrain(nn.Module):
    """
    Simple transformer brain.

    Input: state features + action history
    Output: next action (type + parameters)
    """

    def __init__(self, config: JackConfig = None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.d_model),
            nn.LayerNorm(self.config.d_model),
            nn.GELU(),
        )

        # History encoder
        self.action_embed = nn.Embedding(
            self.config.n_action_types + 1,  # +1 for padding
            self.config.d_model // 2
        )
        self.param_encoder = nn.Linear(
            self.config.action_param_dim,
            self.config.d_model // 2
        )
        self.outcome_embed = nn.Embedding(3, self.config.d_model // 4)  # success/fail/pending
        self.history_combine = nn.Linear(
            self.config.d_model + self.config.d_model // 4,
            self.config.d_model
        )

        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.config.max_history + 1, self.config.d_model) * 0.02
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.d_ff,
            dropout=self.config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, self.config.n_layers)

        # Output heads
        self.action_type_head = nn.Linear(self.config.d_model, self.config.n_action_types)
        self.action_param_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model),
            nn.GELU(),
            nn.Linear(self.config.d_model, self.config.action_param_dim),
        )

        # Success prediction (simple verifier)
        self.success_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.GELU(),
            nn.Linear(self.config.d_model // 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def encode_history(
        self,
        action_types: torch.Tensor,      # (B, history_len)
        action_params: torch.Tensor,     # (B, history_len, param_dim)
        outcomes: torch.Tensor,          # (B, history_len) 0=fail, 1=success, 2=pending
    ) -> torch.Tensor:
        """Encode action history"""
        action_emb = self.action_embed(action_types)
        param_emb = self.param_encoder(action_params)
        outcome_emb = self.outcome_embed(outcomes)

        combined = torch.cat([action_emb, param_emb, outcome_emb], dim=-1)
        return self.history_combine(combined)

    def forward(
        self,
        state: torch.Tensor,             # (B, state_dim)
        action_types: torch.Tensor,      # (B, history_len)
        action_params: torch.Tensor,     # (B, history_len, param_dim)
        outcomes: torch.Tensor,          # (B, history_len)
    ) -> Dict[str, torch.Tensor]:
        """
        Predict next action from state and history.

        Returns:
            action_type_logits: (B, n_action_types)
            action_params: (B, action_param_dim)
            success_prob: (B,) predicted probability of success
        """
        B = state.size(0)

        # Encode state as first token
        state_token = self.state_encoder(state).unsqueeze(1)  # (B, 1, d_model)

        # Encode history
        history_tokens = self.encode_history(action_types, action_params, outcomes)

        # Combine: [state, history...]
        tokens = torch.cat([state_token, history_tokens], dim=1)

        # Add positional encoding
        seq_len = tokens.size(1)
        tokens = tokens + self.pos_embed[:, :seq_len]

        # Transform
        output = self.transformer(tokens)

        # Use state token (first) for prediction
        context = output[:, 0]

        # Predict action
        action_type_logits = self.action_type_head(context)
        action_params = self.action_param_head(context)
        success_prob = self.success_head(context).squeeze(-1)

        return {
            'action_type_logits': action_type_logits,
            'action_params': action_params,
            'success_prob': success_prob,
        }

    def predict_action(
        self,
        state: torch.Tensor,
        action_types: torch.Tensor,
        action_params: torch.Tensor,
        outcomes: torch.Tensor,
    ) -> Tuple[int, torch.Tensor, float]:
        """Convenience method for inference"""
        with torch.no_grad():
            output = self.forward(state, action_types, action_params, outcomes)
            action_type = output['action_type_logits'].argmax(dim=-1).item()
            params = output['action_params'][0]
            success = output['success_prob'].item()
        return action_type, params, success

    def compute_loss(
        self,
        state: torch.Tensor,
        action_types: torch.Tensor,
        action_params: torch.Tensor,
        outcomes: torch.Tensor,
        target_action_type: torch.Tensor,
        target_params: torch.Tensor,
        target_success: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss"""
        output = self.forward(state, action_types, action_params, outcomes)

        # Action type loss (classification)
        type_loss = F.cross_entropy(output['action_type_logits'], target_action_type)

        # Parameter loss (regression)
        param_loss = F.mse_loss(output['action_params'], target_params)

        # Success prediction loss
        success_loss = F.binary_cross_entropy(
            output['success_prob'],
            target_success.float()
        )

        total_loss = type_loss + param_loss + 0.5 * success_loss

        return {
            'total_loss': total_loss,
            'type_loss': type_loss,
            'param_loss': param_loss,
            'success_loss': success_loss,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Action type mapping
ACTION_TYPES = {
    0: 'shell_run',
    1: 'file_read',
    2: 'file_write',
    3: 'http_request',
    4: 'get_state',
}


if __name__ == "__main__":
    print("=" * 50)
    print("JACK BRAIN - Simple & Practical")
    print("=" * 50)

    config = JackConfig()
    brain = JackBrain(config)

    print(f"\nParameters: {brain.count_parameters():,}")
    print(f"Model dim: {config.d_model}")
    print(f"Layers: {config.n_layers}")

    # Test forward pass
    B = 2
    state = torch.randn(B, config.state_dim)
    action_types = torch.randint(0, 5, (B, config.max_history))
    action_params = torch.randn(B, config.max_history, config.action_param_dim)
    outcomes = torch.randint(0, 2, (B, config.max_history))

    output = brain(state, action_types, action_params, outcomes)
    print(f"\nOutput shapes:")
    print(f"  action_type_logits: {output['action_type_logits'].shape}")
    print(f"  action_params: {output['action_params'].shape}")
    print(f"  success_prob: {output['success_prob'].shape}")

    # Test loss
    target_type = torch.randint(0, 5, (B,))
    target_params = torch.randn(B, config.action_param_dim)
    target_success = torch.randint(0, 2, (B,))

    losses = brain.compute_loss(
        state, action_types, action_params, outcomes,
        target_type, target_params, target_success
    )
    print(f"\nLoss: {losses['total_loss'].item():.4f}")

    print("\n" + "=" * 50)
    print("Simple. Practical. Done.")
    print("=" * 50)
