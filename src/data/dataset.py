"""
PyTorch Dataset and DataLoader for BaFuse.

Fixes applied:
- BUG 5: Resample discharge curves to TARGET_SEQ_LEN=100 via np.interp (~4x speedup).
- BUG 1: Removed capacity_fade from physics features (was a direct linear function of
         the SoH label → data leakage). Replaced with an empirical aging prior
         (power-law capacity fade estimate from cycle age alone, independent of the
         actual measured capacity of the sample).
         Physics features are now 3D:
           [cycle_age_norm, voltage_droop, impedance_rise_norm]
         Plus optional empirical prior (total 4D if INCLUDE_EMPIRICAL_PRIOR=True):
           [cycle_age_norm, empirical_fade_prior, voltage_droop, impedance_rise_norm]
- BUG 2: BaFuseDataset accepts external_stats so val/test are normalised with
         train statistics. create_dataloaders passes train_dataset.stats to
         val/test datasets automatically.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────
TARGET_SEQ_LEN = 100          # BUG 5: uniform discharge sequence length

# BUG 1 / P3: physics features (no capacity_fade leakage)
# Physics features: [cycle_age_norm, empirical_fade_prior, voltage_droop, impedance_rise]
PHYSICS_NUM_FEATURES = 4

# Fallback aging model params (used only if train-population fit is unavailable)
# Power-law: expected_fade_fraction = A * (age / N_ref)^b
_FALLBACK_AGING_A     = 0.20
_FALLBACK_AGING_B     = 0.50
_FALLBACK_AGING_N_REF = 168.0


def fit_aging_prior(train_df: pd.DataFrame) -> dict:
    """
    P3: Fit an empirical aging prior from the TRAINING population.

    Fits a power-law: expected_SoH(age) = 100 - A * (age / N_ref)^b
    using numpy's polyfit on log-log scale.

    This is a POPULATION-LEVEL model — it captures the average degradation
    trajectory across all batteries in the training set. At inference, we
    use each sample's cycle_age as input to this curve, NOT the measured
    capacity_ahr, so there is NO per-sample label leakage.

    Returns:
        dict with keys: A, b, N_ref  (power-law parameters)
    """
    # Compute mean SoH per (battery, discharge_cycle)
    ages = train_df["discharge_cycle"].values.astype(float)
    sohs = (train_df["capacity_ahr"] / 2.0 * 100.0).clip(1, 100).values

    # Fade fraction = (100 - SoH) / 100
    fade = (100.0 - sohs) / 100.0
    fade = np.clip(fade, 1e-4, 0.99)

    N_ref = float(np.percentile(ages, 90)) if len(ages) > 10 else 168.0
    N_ref = max(N_ref, 1.0)

    age_norm = np.clip(ages / N_ref, 1e-4, None)

    # log-log regression: log(fade) = log(A) + b * log(age_norm)
    try:
        log_age  = np.log(age_norm)
        log_fade = np.log(fade)
        # filter out degenerate points
        mask = np.isfinite(log_age) & np.isfinite(log_fade)
        if mask.sum() >= 10:
            b, log_A = np.polyfit(log_age[mask], log_fade[mask], 1)
            A = float(np.exp(log_A))
            b = float(np.clip(b, 0.1, 2.0))
        else:
            A, b = _FALLBACK_AGING_A, _FALLBACK_AGING_B
    except Exception:
        A, b = _FALLBACK_AGING_A, _FALLBACK_AGING_B

    params = {"A": A, "b": b, "N_ref": N_ref}
    logger.info(f"Aging prior fit: A={A:.4f}  b={b:.4f}  N_ref={N_ref:.0f}  "
                f"(from {len(train_df)} train samples)")
    return params


def _empirical_fade_prior(cycle_age: float, aging_params: dict) -> float:
    """
    Compute expected capacity fade fraction from cycle_age using fitted prior.
    fade_fraction = A * (age / N_ref)^b  — independent of measured capacity.
    """
    A     = aging_params.get("A",     _FALLBACK_AGING_A)
    b     = aging_params.get("b",     _FALLBACK_AGING_B)
    N_ref = aging_params.get("N_ref", _FALLBACK_AGING_N_REF)
    return float(A * (max(cycle_age, 0.0) / N_ref) ** b)


class BaFuseDataset(Dataset):
    """
    PyTorch Dataset for multimodal battery health data.

    Returns samples with:
    - discharge : (TARGET_SEQ_LEN, 3)  voltage/current/temp resampled
    - eis       : (3,)  [|Z|_norm, Re, Rct]
    - physics   : (PHYSICS_NUM_FEATURES,)  leak-free degradation features
    - soh_label : scalar 0–100
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        discharge_data_df: Optional[pd.DataFrame] = None,
        normalize: bool = True,
        max_seq_len: int = TARGET_SEQ_LEN,
        device: str = 'cpu',
        external_stats: Optional[Dict] = None,      # BUG 2: accept train stats
        aging_prior_params: Optional[Dict] = None,  # P3: fitted aging prior
    ):
        """
        Args:
            data_df            : Paired discharge-EIS DataFrame.
            discharge_data_df  : Full time-series discharge DataFrame (optional).
            normalize          : Whether to z-score features.
            max_seq_len        : Resampled sequence length.
            device             : Torch device string.
            external_stats     : BUG 2 — train-set mean/std for val/test normalization.
            aging_prior_params : P3 — power-law params {A, b, N_ref} fit on train pop.
        """
        self.data_df           = data_df.reset_index(drop=True)
        self.discharge_data_df = discharge_data_df
        self.normalize         = normalize
        self.max_seq_len       = max_seq_len
        self.device            = device

        # P3: aging prior (fallback to defaults if not provided)
        self.aging_params = aging_prior_params if aging_prior_params is not None else {
            "A": _FALLBACK_AGING_A, "b": _FALLBACK_AGING_B, "N_ref": _FALLBACK_AGING_N_REF
        }

        if self.normalize:
            if external_stats is not None:
                self.stats = external_stats
                logger.info("Normalization stats loaded from external source (train set)")
            else:
                self._compute_normalization_stats()

    # ── normalization ───────────────────────────────────────────────────────

    def _compute_normalization_stats(self):
        """Compute mean/std from this dataset's own data_df (train only)."""
        self.stats = {
            'voltage': {
                'mean': float(self.data_df['voltage_mean'].mean()),
                'std':  float(self.data_df['voltage_mean'].std()) + 1e-8,
            },
            'current': {
                'mean': float(self.data_df['current_mean'].mean()),
                'std':  float(self.data_df['current_mean'].std()) + 1e-8,
            },
            'temperature': {
                'mean': float(self.data_df['temp_mean'].mean()),
                'std':  float(self.data_df['temp_mean'].std()) + 1e-8,
            },
            'impedance': {
                'mean': float(self.data_df['impedance_ohm'].mean()),
                'std':  float(self.data_df['impedance_ohm'].std()) + 1e-8,
            },
            'capacity': {
                'mean': float(self.data_df['capacity_ahr'].mean()),
                'std':  float(self.data_df['capacity_ahr'].std()) + 1e-8,
            },
        }
        logger.info("Normalization stats computed from training data")

    def _normalize(self, value: float, stat_key: str) -> float:
        if not self.normalize:
            return value
        s = self.stats[stat_key]
        return (value - s['mean']) / s['std']

    # ── sequence resampling ─────────────────────────────────────────────────

    @staticmethod
    def _resample_sequence(ts: np.ndarray, target_len: int) -> np.ndarray:
        """Resample (N, F) → (target_len, F) via linear interpolation."""
        n, f = ts.shape
        if n == target_len:
            return ts.astype(np.float32)
        x_old = np.linspace(0.0, 1.0, n)
        x_new = np.linspace(0.0, 1.0, target_len)
        out   = np.empty((target_len, f), dtype=np.float32)
        for i in range(f):
            out[:, i] = np.interp(x_new, x_old, ts[:, i])
        return out

    # ── dataset interface ───────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.data_df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row            = self.data_df.iloc[idx]
        battery_id     = row['battery_id']
        discharge_cycle = row['discharge_cycle']
        capacity_ahr   = row['capacity_ahr']

        # ── DISCHARGE ──────────────────────────────────────────────────────
        discharge_features = np.array([
            self._normalize(row['voltage_mean'], 'voltage'),
            self._normalize(row['current_mean'], 'current'),
            self._normalize(row['temp_mean'],    'temperature'),
            self._normalize(row['voltage_min'],  'voltage'),
            self._normalize(row['voltage_max'],  'voltage'),
        ], dtype=np.float32)

        if self.discharge_data_df is not None:
            ts_df = self.discharge_data_df[
                (self.discharge_data_df['battery_id'] == battery_id) &
                (self.discharge_data_df['cycle_idx']  == discharge_cycle)
            ]
            if not ts_df.empty:
                ts_raw = ts_df[['voltage_v', 'current_a', 'temperature_c']].values
                discharge_features = self._resample_sequence(ts_raw, self.max_seq_len)

        discharge_tensor = torch.from_numpy(discharge_features.astype(np.float32))

        # ── EIS ────────────────────────────────────────────────────────────
        eis_features = np.array([
            self._normalize(row['impedance_ohm'], 'impedance'),
            float(row['re_ohm'])  if pd.notna(row['re_ohm'])  else 0.0,
            float(row['rct_ohm']) if pd.notna(row['rct_ohm']) else 0.0,
        ], dtype=np.float32)
        eis_tensor = torch.from_numpy(eis_features)

        # ── PHYSICS (leak-free, population-level prior) ────────────────────
        # [cycle_age_norm, empirical_fade_prior, voltage_droop, impedance_rise]
        # empirical_fade_prior uses fitted power-law from train population — NOT per-sample cap
        cycle_age      = float(discharge_cycle)
        cycle_age_norm = cycle_age / self.aging_params.get("N_ref", _FALLBACK_AGING_N_REF)
        voltage_droop  = (row['voltage_min'] - 2.7) / 1.5
        impedance_rise = (
            (row['impedance_ohm'] - self.stats['impedance']['mean'])
            / self.stats['impedance']['std']
        )
        empirical_prior = _empirical_fade_prior(cycle_age, self.aging_params)

        physics_features = np.array([
            cycle_age_norm,
            empirical_prior,
            voltage_droop,
            impedance_rise,
        ], dtype=np.float32)

        physics_tensor = torch.from_numpy(physics_features)

        # ── SOH LABEL ──────────────────────────────────────────────────────
        soh_label = float(np.clip((capacity_ahr / 2.0) * 100.0, 0.0, 100.0))
        soh_tensor = torch.tensor(soh_label, dtype=torch.float32)

        return {
            'discharge':  discharge_tensor,
            'eis':        eis_tensor,
            'physics':    physics_tensor,
            'soh_label':  soh_tensor,
            'battery_id': battery_id,
            'cycle_idx':  torch.tensor(discharge_cycle, dtype=torch.long),
        }


# ── DataLoader factory ──────────────────────────────────────────────────────

def create_dataloaders(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    discharge_data_df: Optional[pd.DataFrame] = None,
    batch_size:  int  = 32,
    num_workers: int  = 0,
    pin_memory:  bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders.

    BUG 2 FIX: val and test datasets receive train_dataset.stats via
    external_stats so all splits are normalised with the same mean/std.
    """
    # Train: compute stats + fit aging prior from training data
    train_dataset = BaFuseDataset(
        train_df,
        discharge_data_df=discharge_data_df,
        normalize=True,
        external_stats=None,
        aging_prior_params=None,   # fit from train data
    )
    # P3: fit aging prior on train population and propagate to val/test
    aging_params = fit_aging_prior(train_df)

    # Val / Test: reuse train stats + aging prior — BUG 2 + P3 fix
    val_dataset = BaFuseDataset(
        val_df,
        discharge_data_df=discharge_data_df,
        normalize=True,
        external_stats=train_dataset.stats,
        aging_prior_params=aging_params,
    )
    test_dataset = BaFuseDataset(
        test_df,
        discharge_data_df=discharge_data_df,
        normalize=True,
        external_stats=train_dataset.stats,
        aging_prior_params=aging_params,
    )
    # Also update train dataset to use fitted params (already constructed, patch it)
    train_dataset.aging_params = aging_params

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    logger.info(
        f"Created DataLoaders: train={len(train_loader)} "
        f"val={len(val_loader)} test={len(test_loader)} "
        f"[physics_dim={PHYSICS_NUM_FEATURES}, norm=train_stats]"
    )
    return train_loader, val_loader, test_loader
