"""
Mendeley Dataset Parser and PyTorch Dataset for BaFuse v2.

Dataset: "Li-ion cells EIS dataset with fitting and degradation modes"
Cells  : Samsung INR18650-30Q (8 cells, 2.95 Ah nominal)
Source : Mendeley Data (https://data.mendeley.com)

File layout expected:
    <mendeley_root>/
        ExperimentalDATA/
            EISexpCell01.xlsx  ...  EISexpCell08.xlsx
        OutputDATA/
            Circuit_parameter_Cell01.xlsx  ...  Circuit_parameter_Cell08.xlsx
            Z_predicted_Cell01.xlsx        ...  (not used by default)

Confirmed column schemas (read from actual files):

EISexpCell*.xlsx  (sheet='CH1'):
    Aging cycle | SOH(%) | SOC(%) | R_int(%) | Frequency(Hz) |
    Zmod(Ohm)   | Zphz(deg) | Zreal(Ohm) | Zimg(Ohm) | OCV(V)

Circuit_parameter_Cell*.xlsx  (sheet='Sheet1'):
    SOC | aging_cycle | L | R | R_ct1 | Q1 | alpha1 | R_ct2 | Q2 | alpha2 |
    Zw  | residual | chi2 | R² Real | R² Imag | RMSE_Real | RMSE_imag |
    R2  | RMSE | LAM\n% | LLI | CL

IMPORTANT:
  LAM / LLI / CL are model-derived degradation-mode labels obtained by fitting
  an equivalent-circuit model (ECM) to the EIS data.  They are NOT absolute
  physical ground truth.  All downstream code must refer to them as
  "model-derived degradation-mode estimates".

Design decisions:
  - SOH is taken from EISexpCell files (SOH(%) column) and normalised by 100.
  - LAM, LLI, CL are taken from Circuit_parameter files (continuous, may be %).
  - Features are normalised using TRAIN-set statistics only; val/test receive
    external_stats from the train dataset.
  - Only rows with SOC == 100 are used by default (full-charge EIS, more
    reproducible) — configurable via soc_filter argument.
  - Each (cell_id, aging_cycle) forms one sample.
  - The parser is robust to column-name variations via COLUMN_MAP.
  - Missing LAM/LLI/CL are reported and the rows are dropped (never fabricated).
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ── Nominal capacity ──────────────────────────────────────────────────────────
MENDELEY_NOMINAL_CAPACITY_AH = 2.95   # Samsung INR18650-30Q

# ── Column normalisation map ──────────────────────────────────────────────────
# Maps various possible raw names → canonical internal names.
# Handles the 'LAM\n%' newline artefact and Unicode variants.
COLUMN_MAP: Dict[str, str] = {
    # EISexp columns
    "aging cycle":    "aging_cycle",
    "aging_cycle":    "aging_cycle",
    "soh(%)":         "soh_pct",
    "soh":            "soh_pct",
    "soc(%)":         "soc_pct",
    "soc":            "soc_pct",
    "r_int(%)":       "r_int_pct",
    "frequency(hz)":  "freq_hz",
    "zmod(ohm)":      "zmod_ohm",
    "zphz(deg)":      "zphz_deg",
    "zreal(ohm)":     "zreal_ohm",
    "zimg(ohm)":      "zimg_ohm",
    "ocv(v)":         "ocv_v",
    # Circuit_parameter columns (note: "soc" → "soc_pct" already covered above)
    "l":              "L_inductance",
    "r":              "R_electrolyte",
    "r_ct1":          "R_ct1",
    "q1":             "Q1",
    "alpha1":         "alpha1",
    "r_ct2":          "R_ct2",
    "q2":             "Q2",
    "alpha2":         "alpha2",
    "zw":             "Zw",
    "residual":       "residual",
    "chi2":           "chi2",
    "r² real":        "r2_real",   # Unicode superscript variant
    "r2 real":        "r2_real",   # ASCII variant
    "r² imag":        "r2_imag",
    "r2 imag":        "r2_imag",
    "rmse_real":      "rmse_real",  # P4-#13: removed duplicate entry
    "rmse_imag":      "rmse_imag",  # P4-#13: removed duplicate entry
    "r2":             "r2_overall",
    "rmse":           "rmse_overall",
    # Degradation modes — handle newline and spacing artefacts
    "lam\n%":         "lam_pct",
    "lam%":           "lam_pct",
    "lam":            "lam_pct",
    "lli":            "lli_pct",
    "cl":             "cl_pct",
}

# ── EIS features extracted per (cell, aging_cycle) ───────────────────────────
# These are the MEDIAN values across frequencies for the cycle.
EIS_FEATURE_COLS = ["zmod_ohm", "zphz_deg", "zreal_ohm", "zimg_ohm"]

# ── Circuit-parameter features (subset used as EIS features in the model) ────
CIRCUIT_FEATURE_COLS = [
    "R_electrolyte",   # electrolyte / ohmic resistance proxy
    "R_ct1",           # charge-transfer resistance (1st arc)
    "R_ct2",           # charge-transfer resistance (2nd arc)
    "Zw",              # Warburg impedance (diffusion)
]


# ═══════════════════════════════════════════════════════════════════════════════
# Low-level parsers
# ═══════════════════════════════════════════════════════════════════════════════

def _normalise_col(name: str) -> str:
    """Lower-case + strip newlines/spaces, then look up in COLUMN_MAP."""
    key = re.sub(r"\s+", " ", str(name).lower().strip().replace("\n", " "))
    # also try with newline preserved for LAM\n% variant
    key_nl = str(name).lower().strip()
    canonical = COLUMN_MAP.get(key_nl, COLUMN_MAP.get(key, key))
    return canonical


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply COLUMN_MAP to all column names."""
    df = df.copy()
    df.columns = [_normalise_col(c) for c in df.columns]
    return df


def _load_sheet(path: Path, sheet_name: str = None) -> pd.DataFrame:
    """Load first matching sheet, try sheet_name then index 0."""
    xl = pd.ExcelFile(str(path))
    available = xl.sheet_names
    if sheet_name and sheet_name in available:
        df = xl.parse(sheet_name)
    else:
        df = xl.parse(available[0])
    return _rename_columns(df)


def parse_eis_exp_file(path: Path, cell_id: int) -> pd.DataFrame:
    """
    Parse ExperimentalDATA/EISexpCell{NN}.xlsx.

    Returns a DataFrame with one row per (cell_id, aging_cycle, frequency),
    columns: cell_id, aging_cycle, soh_pct, soc_pct, freq_hz,
             zmod_ohm, zphz_deg, zreal_ohm, zimg_ohm, ocv_v.
    """
    df = _load_sheet(path, sheet_name="CH1")
    required = {"aging_cycle", "soh_pct", "soc_pct"}
    missing = required - set(df.columns)
    if missing:
        logger.warning(f"Cell {cell_id}: EISexp missing columns {missing} — skipping")
        return pd.DataFrame()

    df["cell_id"] = cell_id
    df = df.sort_values(["aging_cycle", "freq_hz"], ascending=[True, False])
    df = df.reset_index(drop=True)

    logger.info(
        f"Cell {cell_id}: EISexp loaded — "
        f"{df['aging_cycle'].nunique()} aging cycles, "
        f"{len(df)} frequency-point rows"
    )
    return df


def parse_circuit_param_file(path: Path, cell_id: int) -> pd.DataFrame:
    """
    Parse OutputDATA/Circuit_parameter_Cell{NN}.xlsx.

    Returns a DataFrame with one row per (cell_id, aging_cycle),
    columns include degradation-mode estimates: lam_pct, lli_pct, cl_pct.

    IMPORTANT: lam_pct / lli_pct / cl_pct are model-derived from ECM fitting.
    They must NOT be treated as physical ground truth.
    """
    df = _load_sheet(path, sheet_name="Sheet1")
    required = {"aging_cycle", "lam_pct", "lli_pct", "cl_pct"}
    missing = required - set(df.columns)
    if missing:
        logger.warning(
            f"Cell {cell_id}: Circuit_parameter missing columns {missing}. "
            f"Available: {list(df.columns)}"
        )
        return pd.DataFrame()

    df["cell_id"] = cell_id
    df = df.sort_values("aging_cycle").reset_index(drop=True)

    # Report but do NOT fabricate missing degradation values
    n_missing = df[["lam_pct", "lli_pct", "cl_pct"]].isna().any(axis=1).sum()
    if n_missing:
        logger.warning(
            f"Cell {cell_id}: {n_missing} rows have NaN in degradation labels — "
            f"these will be dropped."
        )
    df = df.dropna(subset=["lam_pct", "lli_pct", "cl_pct"])

    logger.info(
        f"Cell {cell_id}: Circuit_parameter loaded — "
        f"{len(df)} valid aging cycles"
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset builder
# ═══════════════════════════════════════════════════════════════════════════════

def _aggregate_eis_per_cycle(eis_df: pd.DataFrame, soc_filter: int = 100) -> pd.DataFrame:
    """
    Aggregate per-frequency EIS rows into one row per (cell_id, aging_cycle).

    Steps:
    1. Filter to SOC == soc_filter (default 100 % = full charge).
    2. Take the median across frequency points for each EIS feature.
    3. Keep the first SOH value per cycle (same within a cycle).
    """
    df = eis_df.copy()
    if soc_filter is not None and "soc_pct" in df.columns:
        df = df[df["soc_pct"] == soc_filter].copy()
        if df.empty:
            logger.warning(
                f"No rows with SOC={soc_filter}% — returning empty. "
                f"Check soc_filter or raw data."
            )
            return pd.DataFrame()

    agg_cols = {}
    for col in EIS_FEATURE_COLS:
        if col in df.columns:
            agg_cols[col] = "median"

    if "soh_pct" in df.columns:
        agg_cols["soh_pct"] = "first"
    if "ocv_v" in df.columns:
        agg_cols["ocv_v"] = "first"

    if not agg_cols:
        logger.warning("No EIS feature columns found after column mapping.")
        return pd.DataFrame()

    agg = (
        df.groupby(["cell_id", "aging_cycle"])
        .agg(agg_cols)
        .reset_index()
    )
    return agg


def build_mendeley_dataframe(
    mendeley_root: str,
    soc_filter: int = 100,
    num_cells: int = 8,
) -> pd.DataFrame:
    """
    Build a clean merged DataFrame from all Mendeley cells.

    For each cell:
      1. Parse EISexpCell{NN}.xlsx → aggregate to one row per aging cycle.
      2. Parse Circuit_parameter_Cell{NN}.xlsx → degradation-mode labels.
      3. Inner-join on (cell_id, aging_cycle).

    Returns a DataFrame with columns:
        cell_id, aging_cycle, soh_norm,
        zmod_ohm, zphz_deg, zreal_ohm, zimg_ohm, ocv_v,
        R_electrolyte, R_ct1, R_ct2, Zw,
        lam_pct, lli_pct, cl_pct

    soh_norm = soh_pct / 100  (0–1 scale)
    """
    root = Path(mendeley_root)
    exp_dir = root / "ExperimentalDATA"
    out_dir = root / "OutputDATA"

    if not exp_dir.exists():
        raise FileNotFoundError(f"ExperimentalDATA not found at: {exp_dir}")
    if not out_dir.exists():
        raise FileNotFoundError(f"OutputDATA not found at: {out_dir}")

    all_frames: List[pd.DataFrame] = []

    for cell_num in range(1, num_cells + 1):
        cell_id   = cell_num
        eis_path  = exp_dir / f"EISexpCell{cell_num:02d}.xlsx"
        circ_path = out_dir / f"Circuit_parameter_Cell{cell_num:02d}.xlsx"

        if not eis_path.exists():
            logger.warning(f"Missing {eis_path}, skipping cell {cell_id}")
            continue
        if not circ_path.exists():
            logger.warning(f"Missing {circ_path}, skipping cell {cell_id}")
            continue

        try:
            eis_raw  = parse_eis_exp_file(eis_path, cell_id)
            circ_raw = parse_circuit_param_file(circ_path, cell_id)
        except Exception as exc:
            logger.error(f"Cell {cell_id}: parse error — {exc}")
            continue

        if eis_raw.empty or circ_raw.empty:
            continue

        eis_agg = _aggregate_eis_per_cycle(eis_raw, soc_filter=soc_filter)
        if eis_agg.empty:
            continue

        # Inner join on (cell_id, aging_cycle)
        merged = pd.merge(
            eis_agg,
            circ_raw[["cell_id", "aging_cycle", "lam_pct", "lli_pct", "cl_pct"]
                     + [c for c in CIRCUIT_FEATURE_COLS if c in circ_raw.columns]],
            on=["cell_id", "aging_cycle"],
            how="inner",
        )

        if merged.empty:
            logger.warning(
                f"Cell {cell_id}: inner join produced 0 rows — "
                f"check aging_cycle alignment between EIS and Circuit_parameter files."
            )
            continue

        # P4-#11: log match rate to detect silent partial-join data loss
        n_eis  = len(eis_agg)
        n_circ = len(circ_raw)
        n_join = len(merged)
        match_rate = n_join / max(min(n_eis, n_circ), 1)
        if match_rate < 0.9:
            logger.warning(
                f"Cell {cell_id}: inner join matched {n_join}/{min(n_eis,n_circ)} cycles "
                f"({match_rate*100:.0f}%) — possible aging_cycle numbering mismatch "
                f"between EISexp ({n_eis} cycles) and Circuit_parameter ({n_circ} cycles)."
            )
        else:
            logger.info(
                f"Cell {cell_id}: inner join matched {n_join}/{min(n_eis,n_circ)} cycles "
                f"({match_rate*100:.0f}%)"
            )

        all_frames.append(merged)

    if not all_frames:
        raise RuntimeError(
            "No valid Mendeley data loaded. Check path and file format."
        )

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["cell_id", "aging_cycle"]).reset_index(drop=True)

    # SOH normalised to [0, 1]
    if "soh_pct" in combined.columns:
        combined["soh_norm"] = combined["soh_pct"] / 100.0
    else:
        logger.warning("soh_pct not found — soh_norm will be NaN")
        combined["soh_norm"] = np.nan

    logger.info(
        f"Mendeley dataset: {combined['cell_id'].nunique()} cells, "
        f"{len(combined)} samples, "
        f"aging cycles {combined['aging_cycle'].min()}–{combined['aging_cycle'].max()}"
    )
    return combined


def split_mendeley_by_cell(
    df: pd.DataFrame,
    train_cells: Optional[List[int]] = None,
    val_cells:   Optional[List[int]] = None,
    test_cells:  Optional[List[int]] = None,
    val_ratio:   float = 0.15,
    test_ratio:  float = 0.15,
    random_state: int  = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cell-level train/val/test split for Mendeley data.

    If explicit cell lists are not provided, splits automatically.
    Cell-level split prevents leakage: a cell does not appear in both
    train and val/test sets.
    """
    all_cells = sorted(df["cell_id"].unique())
    n = len(all_cells)

    if train_cells is None:
        rng = np.random.RandomState(random_state)
        perm = rng.permutation(all_cells)
        n_test  = max(1, int(n * test_ratio))
        n_val   = max(1, int(n * val_ratio))
        test_cells  = list(perm[:n_test])
        val_cells   = list(perm[n_test:n_test + n_val])
        train_cells = list(perm[n_test + n_val:])

    train_df = df[df["cell_id"].isin(train_cells)].reset_index(drop=True)
    val_df   = df[df["cell_id"].isin(val_cells)].reset_index(drop=True)
    test_df  = df[df["cell_id"].isin(test_cells)].reset_index(drop=True)

    logger.info(
        f"Mendeley split: train={len(train_df)} ({train_cells}), "
        f"val={len(val_df)} ({val_cells}), "
        f"test={len(test_df)} ({test_cells})"
    )
    return train_df, val_df, test_df


def leave_one_cell_out_splits(
    df: pd.DataFrame,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    P4-#12: Leave-One-Cell-Out (LOCO) cross-validation splits for Mendeley data.

    With only 8 cells a single fixed split (1 test cell) is too thin for
    reliable evaluation. LOCO runs 8 folds, each holding out one cell as
    test and using the remaining 7 as training, giving a more robust estimate
    of generalisation across cells.

    Returns:
        List of (train_df, test_df) tuples, one per fold.
        val_df is not provided separately; callers may take a fraction of
        train_df for validation within each fold.

    Usage:
        for fold_i, (train_df, test_df) in enumerate(leave_one_cell_out_splits(df)):
            # train on train_df, evaluate on test_df
            ...
    """
    all_cells = sorted(df["cell_id"].unique())
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []

    for held_out in all_cells:
        train_part = df[df["cell_id"] != held_out].reset_index(drop=True)
        test_part  = df[df["cell_id"] == held_out].reset_index(drop=True)
        folds.append((train_part, test_part))
        logger.debug(
            f"LOCO fold: test=cell_{held_out} "
            f"({len(test_part)} samples), "
            f"train={[c for c in all_cells if c != held_out]} "
            f"({len(train_part)} samples)"
        )

    logger.info(
        f"leave_one_cell_out_splits: {len(folds)} folds over cells {all_cells}"
    )
    return folds


# ═══════════════════════════════════════════════════════════════════════════════
# Normalisation stats
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mendeley_stats(train_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute mean/std normalisation statistics from the Mendeley training set.

    Stats are computed only from TRAIN rows — must be propagated to val/test.
    """
    feature_cols = EIS_FEATURE_COLS + [
        c for c in CIRCUIT_FEATURE_COLS if c in train_df.columns
    ]
    stats: Dict[str, Dict[str, float]] = {}
    for col in feature_cols:
        if col in train_df.columns:
            vals = train_df[col].dropna().values.astype(float)
            stats[col] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)) + 1e-8,
            }

    # Degradation labels — normalise to unit scale
    for deg_col in ["lam_pct", "lli_pct", "cl_pct"]:
        if deg_col in train_df.columns:
            vals = train_df[deg_col].dropna().values.astype(float)
            stats[deg_col] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)) + 1e-8,
            }

    logger.info(f"Mendeley normalisation stats computed for {len(stats)} features")
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# PyTorch Dataset
# ═══════════════════════════════════════════════════════════════════════════════

# EIS features in the order they will be returned by __getitem__
_EIS_FEATURES_ORDERED = [
    "zmod_ohm", "zreal_ohm", "zimg_ohm", "zphz_deg",
    "R_electrolyte", "R_ct1", "R_ct2", "Zw",
]


class MendeleyDataset(Dataset):
    """
    PyTorch Dataset for the Mendeley EIS degradation dataset.

    Each sample contains:
        eis          : (num_eis_features,)   — normalised EIS/circuit features
        soh_label    : scalar (0–1)          — normalised SOH from EISexp
        lli_label    : scalar                — model-derived LLI estimate
        lam_label    : scalar                — model-derived LAM estimate
        cl_label     : scalar                — model-derived CL estimate
        cell_id      : int
        aging_cycle  : int

    IMPORTANT: lli_label / lam_label / cl_label are model-derived from ECM
    fitting.  They are NOT physical ground truth.

    Args:
        df             : DataFrame from build_mendeley_dataframe() (one split).
        normalize      : Whether to apply z-score normalisation.
        external_stats : If provided, use these stats (from training set).
    """

    def __init__(
        self,
        df:             pd.DataFrame,
        normalize:      bool                         = True,
        external_stats: Optional[Dict[str, Dict]]   = None,
    ):
        self.df        = df.reset_index(drop=True)
        self.normalize = normalize

        # Determine which EIS feature columns are actually present
        self.eis_cols = [c for c in _EIS_FEATURES_ORDERED if c in self.df.columns]
        if not self.eis_cols:
            raise ValueError(
                "No EIS feature columns found in DataFrame. "
                f"Expected one of: {_EIS_FEATURES_ORDERED}. "
                f"Got: {list(self.df.columns)}"
            )

        if normalize:
            if external_stats is not None:
                self.stats = external_stats
            else:
                self.stats = compute_mendeley_stats(self.df)

        logger.info(
            f"MendeleyDataset: {len(self.df)} samples, "
            f"eis_features={self.eis_cols} ({len(self.eis_cols)}D)"
        )

    # -----------------------------------------------------------------------

    @property
    def num_eis_features(self) -> int:
        return len(self.eis_cols)

    # -----------------------------------------------------------------------

    def _norm(self, value: float, col: str) -> float:
        if not self.normalize or not hasattr(self, "stats"):
            return value
        s = self.stats.get(col)
        if s is None:
            return value
        return (value - s["mean"]) / s["std"]

    # -----------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # ── EIS features ────────────────────────────────────────────────────
        eis_vals = []
        for col in self.eis_cols:
            raw = float(row[col]) if pd.notna(row.get(col, np.nan)) else 0.0
            eis_vals.append(self._norm(raw, col))
        eis_tensor = torch.tensor(eis_vals, dtype=torch.float32)

        # ── SOH label (0–1) ─────────────────────────────────────────────────
        soh_raw = float(row.get("soh_norm", row.get("soh_pct", 100.0)))
        if soh_raw > 1.5:            # legacy: still in percent form
            soh_raw = soh_raw / 100.0
        soh_tensor = torch.tensor(np.clip(soh_raw, 0.0, 1.0), dtype=torch.float32)

        # ── Model-derived degradation-mode labels ────────────────────────────
        lli_raw = float(row.get("lli_pct", 0.0)) if pd.notna(row.get("lli_pct")) else 0.0
        lam_raw = float(row.get("lam_pct", 0.0)) if pd.notna(row.get("lam_pct")) else 0.0
        cl_raw  = float(row.get("cl_pct",  0.0)) if pd.notna(row.get("cl_pct"))  else 0.0

        lli_norm = self._norm(lli_raw, "lli_pct")
        lam_norm = self._norm(lam_raw, "lam_pct")
        cl_norm  = self._norm(cl_raw,  "cl_pct")

        return {
            "eis":          eis_tensor,
            "soh_label":    soh_tensor,
            "lli_label":    torch.tensor(lli_norm, dtype=torch.float32),
            "lam_label":    torch.tensor(lam_norm, dtype=torch.float32),
            "cl_label":     torch.tensor(cl_norm,  dtype=torch.float32),
            # raw (un-normalised) for logging / visualisation
            "lli_raw":      torch.tensor(lli_raw,  dtype=torch.float32),
            "lam_raw":      torch.tensor(lam_raw,  dtype=torch.float32),
            "cl_raw":       torch.tensor(cl_raw,   dtype=torch.float32),
            "cell_id":      int(row["cell_id"]),
            "aging_cycle":  torch.tensor(int(row["aging_cycle"]), dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DataLoader factory
# ═══════════════════════════════════════════════════════════════════════════════

def create_mendeley_dataloaders(
    train_df:    pd.DataFrame,
    val_df:      pd.DataFrame,
    test_df:     pd.DataFrame,
    batch_size:  int  = 16,
    num_workers: int  = 0,
    pin_memory:  bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create Mendeley train/val/test DataLoaders.

    Val and test sets use train normalisation statistics (no leakage).

    Returns:
        (train_loader, val_loader, test_loader, train_stats)
    """
    train_ds = MendeleyDataset(train_df, normalize=True, external_stats=None)
    train_stats = train_ds.stats if hasattr(train_ds, "stats") else {}

    val_ds  = MendeleyDataset(val_df,  normalize=True, external_stats=train_stats)
    test_ds = MendeleyDataset(test_df, normalize=True, external_stats=train_stats)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    logger.info(
        f"Mendeley DataLoaders: train={len(train_loader)} "
        f"val={len(val_loader)} test={len(test_loader)} batches  "
        f"[eis_dim={train_ds.num_eis_features}]"
    )
    return train_loader, val_loader, test_loader, train_stats
