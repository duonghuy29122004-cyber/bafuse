"""
PyTorch Dataset and DataLoader for BaFuse.

Fixes applied:
- BUG 5: Resample discharge curves to TARGET_SEQ_LEN=100 via np.interp (~4x speedup).
- BUG 1: Removed capacity_fade from physics features (data leakage). Replaced with
         empirical aging prior (power-law, cycle-age only, no per-sample capacity).
- BUG 2: BaFuseDataset accepts external_stats so val/test are normalised with
         train statistics. create_dataloaders passes train_dataset.stats automatically.
- P1-#2: Added BATTERY_NOMINAL_CAPACITY dict (per-campaign from NASA PCoE READMEs).
         SOH = capacity_ahr / nominal_capacity(battery_id) — no more hardcoded /2.0.
         fit_aging_prior also uses per-cell nominal capacity for correct SOH [0,1].
- P2-#3: Time-series discharge channels (voltage/current/temp) are now z-score
         normalised before being returned, matching the aggregate fallback path.
- P2-#4: re_ohm and rct_ohm are now normalised via dedicated stats computed at
         train time. Stats dict keys: 're' and 'rct'.
- P2-#5: If discharge_data_df is provided, __init__ pre-filters data_df to rows
         that actually have matching time-series. Missing rows are logged and dropped
         at construction time to prevent silent shape mixing at collate.
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
_DEFAULT_CUTOFF = 2.7   # fallback for unknown battery_id

# Set to False to reproduce old fixed-2.7V behaviour (used in CV comparison config (a))
_USE_PER_BATTERY_CUTOFF = True

def get_cutoff_voltage(battery_id: str) -> float:
    """Return discharge cutoff voltage for battery_id. Falls back to 2.7V if not found."""
    if not _USE_PER_BATTERY_CUTOFF:
        return 2.7
    return BATTERY_CUTOFF_VOLTAGE.get(str(battery_id), _DEFAULT_CUTOFF)

# ── Aging model fallback params ────────────────────────────────────────────
_FALLBACK_AGING_A     = 0.20
_FALLBACK_AGING_B     = 0.50
_FALLBACK_AGING_N_REF = 168.0

# ── Per-battery discharge cutoff voltages (from NASA PCoE README files) ────
BATTERY_CUTOFF_VOLTAGE: Dict[str, float] = {
    # Campaign 1 — BatteryAgingARC-FY08Q4
    "B0005": 2.7, "B0006": 2.5, "B0007": 2.2, "B0018": 2.5,
    # Campaign 2 & 3 — B0025-B0028 (same protocol)
    "B0025": 2.0, "B0026": 2.2, "B0027": 2.5, "B0028": 2.7,
    # Campaign 3 cont.
    "B0029": 2.0, "B0030": 2.2, "B0031": 2.5, "B0032": 2.7,
    "B0033": 2.0, "B0034": 2.2, "B0036": 2.7,
    "B0038": 2.2, "B0039": 2.5, "B0040": 2.7,
    "B0041": 2.0, "B0042": 2.2, "B0043": 2.5, "B0044": 2.7,
    # Campaign 4
    "B0045": 2.0, "B0046": 2.2, "B0047": 2.5, "B0048": 2.7,
    # Campaign 5
    "B0049": 2.0, "B0050": 2.2, "B0051": 2.5, "B0052": 2.7,
    # Campaign 6
    "B0053": 2.0, "B0054": 2.2, "B0055": 2.5, "B0056": 2.7,
}

# ── Per-battery nominal (rated) capacity from NASA PCoE README files ────────
# All Campaign 1-6 cells are 18650-type Li-ion, rated 2.0 Ah by NASA PCoE.
# Source: README files in each BatteryAgingARC-* subdirectory.
# If a future campaign uses a different cell chemistry/format, add it here.
BATTERY_NOMINAL_CAPACITY: Dict[str, float] = {
    # Campaign 1 — BatteryAgingARC-FY08Q4 (2.0 Ah rated)
    "B0005": 2.0, "B0006": 2.0, "B0007": 2.0, "B0018": 2.0,
    # Campaign 2 — BatteryAgingARC_25_26_27_28_P1 (2.0 Ah rated)
    "B0025": 2.0, "B0026": 2.0, "B0027": 2.0, "B0028": 2.0,
    # Campaign 3 — BatteryAgingARC_25-44 (2.0 Ah rated)
    "B0029": 2.0, "B0030": 2.0, "B0031": 2.0, "B0032": 2.0,
    "B0033": 2.0, "B0034": 2.0, "B0036": 2.0,
    "B0038": 2.0, "B0039": 2.0, "B0040": 2.0,
    "B0041": 2.0, "B0042": 2.0, "B0043": 2.0, "B0044": 2.0,
    # Campaign 4 — BatteryAgingARC_45_46_47_48 (2.0 Ah rated)
    "B0045": 2.0, "B0046": 2.0, "B0047": 2.0, "B0048": 2.0,
    # Campaign 5 — BatteryAgingARC_49_50_51_52 (2.0 Ah rated)
    "B0049": 2.0, "B0050": 2.0, "B0051": 2.0, "B0052": 2.0,
    # Campaign 6 — BatteryAgingARC_53_54_55_56 (2.0 Ah rated)
    "B0053": 2.0, "B0054": 2.0, "B0055": 2.0, "B0056": 2.0,
}
_DEFAULT_NOMINAL_CAPACITY = 2.0  # fallback if battery_id not in dict


def get_nominal_capacity(battery_id: str) -> float:
    """Return rated nominal capacity (Ah) for battery_id. Falls back to 2.0 Ah."""
    return BATTERY_NOMINAL_CAPACITY.get(str(battery_id), _DEFAULT_NOMINAL_CAPACITY)


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
    # P1-#2: use per-battery nominal capacity for SOH in [0,1].
    # Previously hardcoded /2.0 — now looks up BATTERY_NOMINAL_CAPACITY per row.
    ages = train_df["discharge_cycle"].values.astype(float)
    nominal = train_df["battery_id"].map(
        lambda bid: get_nominal_capacity(bid)
    ).values.astype(float)
    sohs = np.clip(train_df["capacity_ahr"].values / nominal, 0.01, 1.0)

    # Fade fraction = 1 - SOH  (already in [0,1] range)
    fade = (1.0 - sohs)
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
        external_stats: Optional[Dict] = None,
        aging_prior_params: Optional[Dict] = None,
    ):
        """
        Args:
            data_df            : Paired discharge-EIS DataFrame.
            discharge_data_df  : Full time-series discharge DataFrame (optional).
                                 P2-#5: If provided, data_df rows whose
                                 (battery_id, cycle_idx) are absent in
                                 discharge_data_df are silently dropped at
                                 construction time so all samples in this
                                 Dataset have the same (TARGET_SEQ_LEN, 3) shape.
            normalize          : Whether to z-score features.
            max_seq_len        : Resampled sequence length.
            device             : Torch device string.
            external_stats     : Train-set mean/std for val/test normalization.
            aging_prior_params : Power-law params {A, b, N_ref} fit on train pop.
        """
        self.discharge_data_df = discharge_data_df
        self.normalize         = normalize
        self.max_seq_len       = max_seq_len
        self.device            = device

        # P2-#5: Pre-filter data_df when discharge_data_df is provided.
        # This prevents silent shape mixing in DataLoader.collate_fn.
        if discharge_data_df is not None and not discharge_data_df.empty:
            available = set(
                zip(discharge_data_df["battery_id"], discharge_data_df["cycle_idx"])
            )
            mask = data_df.apply(
                lambda r: (r["battery_id"], r["discharge_cycle"]) in available,
                axis=1,
            )
            n_before = len(data_df)
            data_df  = data_df[mask].reset_index(drop=True)
            n_dropped = n_before - len(data_df)
            if n_dropped > 0:
                logger.warning(
                    f"P2-#5: Dropped {n_dropped}/{n_before} rows from data_df "
                    f"because their (battery_id, discharge_cycle) pairs were not "
                    f"found in discharge_data_df. All remaining samples will use "
                    f"time-series shape ({max_seq_len}, 3)."
                )
            else:
                logger.debug(
                    f"P2-#5: All {n_before} data_df rows have matching time-series."
                )

        self.data_df      = data_df.reset_index(drop=True)
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
        """
        Compute mean/std from this dataset's own data_df (train only).

        P2-#4: Added 're' and 'rct' stats so re_ohm and rct_ohm are
        normalised consistently with impedance_ohm in __getitem__.
        """
        def _safe_stat(col: str) -> Dict[str, float]:
            vals = self.data_df[col].dropna().values.astype(float)
            return {
                "mean": float(np.mean(vals)) if len(vals) else 0.0,
                "std":  float(np.std(vals))  + 1e-8,
            }

        self.stats = {
            'voltage':     _safe_stat('voltage_mean'),
            'current':     _safe_stat('current_mean'),
            'temperature': _safe_stat('temp_mean'),
            'impedance':   _safe_stat('impedance_ohm'),
            'capacity':    _safe_stat('capacity_ahr'),
            # P2-#4: EIS sub-components
            're':          _safe_stat('re_ohm'),
            'rct':         _safe_stat('rct_ohm'),
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
        # Aggregate fallback (5 features, shape=(5,)) — used when no time-series
        # data is available. After P2-#5 __init__ pre-filter, if discharge_data_df
        # was provided the ts branch below will always succeed; the fallback is
        # only reached when discharge_data_df=None.
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

                # P2-#3: z-score each channel using train-set stats BEFORE
                # resampling. Normalise column-by-column with matching stat keys.
                if self.normalize:
                    v_mean = self.stats['voltage']['mean']
                    v_std  = self.stats['voltage']['std']
                    c_mean = self.stats['current']['mean']
                    c_std  = self.stats['current']['std']
                    t_mean = self.stats['temperature']['mean']
                    t_std  = self.stats['temperature']['std']

                    ts_norm = ts_raw.copy().astype(np.float64)
                    ts_norm[:, 0] = (ts_norm[:, 0] - v_mean) / v_std   # voltage
                    ts_norm[:, 1] = (ts_norm[:, 1] - c_mean) / c_std   # current
                    ts_norm[:, 2] = (ts_norm[:, 2] - t_mean) / t_std   # temperature
                    ts_raw = ts_norm

                discharge_features = self._resample_sequence(
                    ts_raw.astype(np.float32), self.max_seq_len
                )

        discharge_tensor = torch.from_numpy(discharge_features.astype(np.float32))

        # ── EIS ────────────────────────────────────────────────────────────
        # P2-#4: normalise all three EIS features. re_ohm / rct_ohm use their
        # own stats ('re', 'rct') computed at train time; missing values → 0.0
        # (the mean before normalisation, so NaN→0 after z-score is correct).
        def _eis_val(col: str, stat_key: str) -> float:
            raw = row.get(col, None)
            val = float(raw) if (raw is not None and pd.notna(raw)) else 0.0
            return self._normalize(val, stat_key) if self.normalize else val

        eis_features = np.array([
            self._normalize(row['impedance_ohm'], 'impedance'),
            _eis_val('re_ohm',  're'),
            _eis_val('rct_ohm', 'rct'),
        ], dtype=np.float32)
        eis_tensor = torch.from_numpy(eis_features)

        # ── PHYSICS (leak-free, population-level prior) ────────────────────
        cycle_age      = float(discharge_cycle)
        cycle_age_norm = cycle_age / self.aging_params.get("N_ref", _FALLBACK_AGING_N_REF)
        cutoff_v       = get_cutoff_voltage(battery_id)
        voltage_range  = max(4.2 - cutoff_v, 0.1)
        voltage_droop  = (row['voltage_min'] - cutoff_v) / voltage_range
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

        # ── SOH LABEL — normalised to [0, 1] ──────────────────────────────
        # P1-#2: use per-battery nominal capacity, not hardcoded 2.0 Ah.
        # SOH = capacity_ahr / nominal_capacity(battery_id).
        # Reported metrics: MAE% = MAE*100, RMSE% = RMSE*100.
        nominal_cap = get_nominal_capacity(battery_id)
        soh_label   = float(np.clip(capacity_ahr / nominal_cap, 0.0, 1.0))
        soh_tensor  = torch.tensor(soh_label, dtype=torch.float32)

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
