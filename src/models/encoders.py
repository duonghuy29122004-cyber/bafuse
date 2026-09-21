"""
Modular encoders for discharge curve, EIS spectrum, and physics-informed features.

Each encoder:
- Processes its modality independently
- Outputs fixed-size latent representation
- Can be pre-trained or fine-tuned
"""

import torch
import torch.nn as nn
from typing import Optional


class DischargeEncoder(nn.Module):
    """
    Encode discharge curve dynamics into latent vector using LSTM.

    P4: hidden_size=256, num_layers=3 for more capacity.
    Reverted latent_dim projection to single layer (dataset too small for deep MLP here).
    """

    def __init__(self, input_size: int, hidden_size: int = 256, latent_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=0.2,
            bidirectional=False,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_size)  OR (batch_size, input_size) for aggregate

        Returns:
            (batch_size, latent_dim)
        """
        # Handle both time-series (3-D) and aggregate (2-D) inputs
        if x.dim() == 2:
            # Aggregate features — treat as single time-step
            x = x.unsqueeze(1)  # (B, 1, input_size)

        _, (h_n, _) = self.lstm(x)   # h_n: (num_layers, B, hidden_size)
        # Use the last layer hidden state
        h_last = h_n[-1]             # (B, hidden_size)
        return self.proj(h_last)     # (B, latent_dim)


class EISEncoder(nn.Module):
    """
    Encode EIS spectrum (impedance measurements) into latent vector via 1-D CNN.

    Input: Real/imaginary impedance vs frequency  OR scalar impedance features
    Output: Fixed-size latent representation
    """

    def __init__(self, num_frequencies: int, hidden_size: int = 128, latent_dim: int = 64):
        """
        Args:
            num_frequencies: Number of frequency points in EIS (or feature dimension)
            hidden_size: Hidden layer dimension
            latent_dim: Output latent dimension
        """
        super().__init__()

        self.use_mlp = num_frequencies <= 8  # small feature vector → just MLP

        if self.use_mlp:
            self.net = nn.Sequential(
                nn.Linear(num_frequencies, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.ReLU(),
            )
        else:
            # 1-D CNN on frequency spectrum
            self.cnn = nn.Sequential(
                nn.Conv1d(2, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),   # global average pooling → (B, 64, 1)
            )
            self.proj = nn.Sequential(
                nn.Linear(64, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.ReLU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_frequencies, 2) [real, imag]
               OR (batch_size, num_features) for scalar case

        Returns:
            (batch_size, latent_dim)
        """
        if self.use_mlp:
            if x.dim() == 3:
                # Flatten (B, F, 2) → (B, F*2)? — adapt by flattening last two dims
                x = x.reshape(x.size(0), -1)
            return self.net(x)
        else:
            # x: (B, num_freq, 2) → permute to (B, 2, num_freq) for Conv1d
            if x.dim() == 2:
                # Scalar EIS features, fall back to simple MLP path
                return self._mlp_fallback(x)
            x = x.permute(0, 2, 1)  # (B, 2, num_freq)
            out = self.cnn(x)        # (B, 64, 1)
            out = out.squeeze(-1)    # (B, 64)
            return self.proj(out)

    def _mlp_fallback(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback MLP for scalar EIS features."""
        # Create simple linear projection
        if not hasattr(self, '_fallback_proj'):
            self._fallback_proj = nn.Linear(x.size(1), 64).to(x.device)
        out = torch.relu(self._fallback_proj(x))
        return out


class PhysicsEncoder(nn.Module):
    """
    Encode physics-informed simulation features (degradation trajectory prior) via MLP.

    Input: Simulated SoH trajectory based on cycling model
    Output: Fixed-size latent representation
    """

    def __init__(self, num_physics_features: int = 10, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_physics_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_physics_features)
        Returns:
            (batch_size, latent_dim)
        """
        return self.net(x)


class PhysicsEncoderCNN(nn.Module):
    """
    Task 3: CNN 1D alternative for physics encoder.

    Treats the 4-dim physics vector as a 1-channel sequence of length 4,
    applies 1D convolutions to capture local interactions between adjacent
    features (e.g. cycle_age ↔ empirical_prior, voltage_droop ↔ impedance_rise),
    then projects to latent_dim.

    Note: Physics features do NOT have a natural temporal ordering, so CNN1D
    is an architectural experiment rather than a motivated design choice.
    If MLP performs equally well or better, that result is reported as-is.

    Output shape is identical to PhysicsEncoder → fully compatible with fusion.
    """

    def __init__(self, num_physics_features: int = 4, latent_dim: int = 64):
        super().__init__()
        # (B, 1, num_features) → conv layers
        self.cnn = nn.Sequential(
            # Layer 1: kernel=2 captures pairwise feature interactions
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2, padding=1),
            nn.ReLU(),
            # Layer 2: kernel=2 again, reduce length
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, padding=0),
            nn.ReLU(),
        )
        # After two convolutions on length-4 input with those params:
        # L after conv1(k=2,pad=1): 4+2*1-2+1 = 5
        # L after conv2(k=2,pad=0): 5-2+1 = 4
        # → flatten: 32 * 4 = 128  (but safer to use AdaptiveAvgPool)
        self.pool = nn.AdaptiveAvgPool1d(2)   # → (B, 32, 2)
        self.head = nn.Sequential(
            nn.Flatten(),                      # (B, 64)
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_physics_features)
        Returns:
            (B, latent_dim)
        """
        x = x.unsqueeze(1)          # (B, 1, num_features)
        x = self.cnn(x)             # (B, 32, ?)
        x = self.pool(x)            # (B, 32, 2)
        return self.head(x)         # (B, latent_dim)
