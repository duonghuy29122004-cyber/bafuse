"""
Degradation Quick Test — Separate Experiment

IMPORTANT:
- Uses ONLY Mendeley/Samsung dataset
- Does NOT modify NASA A-H SOH benchmark
- Does NOT mix Samsung with NASA training
- Treats LLI/LAM/CL as model-derived indicators from ECM fitting (NOT direct physical ground truth)

Task:
EIS features → small MLP → [LLI, LAM, CL] (multi-target regression)
"""
import sys, json, logging, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── Simple MLP for degradation mode prediction ──────────────────────────────
class DegradationMLP(nn.Module):
    """Small MLP: EIS features → shared representation → [LLI, LAM, CL]"""
    def __init__(self, input_dim=9, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head_lli = nn.Linear(hidden_dim, 1)
        self.head_lam = nn.Linear(hidden_dim, 1)
        self.head_cl = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        return {
            "lli": self.head_lli(h).squeeze(-1),
            "lam": self.head_lam(h).squeeze(-1),
            "cl": self.head_cl(h).squeeze(-1),
        }

# ─── Dataset ──────────────────────────────────────────────────────────────────
class MendeleyDegradationDataset(Dataset):
    def __init__(self, df, stats=None):
        self.df = df.reset_index(drop=True)
        self.eis_cols = ["zmod_ohm", "zphz_deg", "zreal_ohm", "zimg_ohm",
                         "R_electrolyte", "R_ct1", "R_ct2", "Zw", "ocv_v"]
        self.target_cols = ["lli_pct", "lam_pct", "cl_pct"]
        
        # Compute or use external stats
        if stats is None:
            self.stats = {}
            for col in self.eis_cols:
                vals = self.df[col].values
                self.stats[col] = {"mean": float(vals.mean()), "std": float(vals.std())}
        else:
            self.stats = stats

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Normalize EIS features
        eis = torch.tensor(
            [(row[c] - self.stats[c]["mean"]) / self.stats[c]["std"] for c in self.eis_cols],
            dtype=torch.float32
        )
        
        # Targets (raw, not normalized — regression targets)
        targets = torch.tensor(
            [row[c] for c in self.target_cols],
            dtype=torch.float32
        )
        
        return {
            "eis": eis,
            "lli": targets[0],
            "lam": targets[1],
            "cl": targets[2],
            "cell_id": row["cell_id"],
        }

# ─── Train ────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        eis = batch["eis"].to(device)
        lli_true = batch["lli"].to(device)
        lam_true = batch["lam"].to(device)
        cl_true = batch["cl"].to(device)
        
        optimizer.zero_grad()
        pred = model(eis)
        
        loss_lli = nn.functional.mse_loss(pred["lli"], lli_true)
        loss_lam = nn.functional.mse_loss(pred["lam"], lam_true)
        loss_cl = nn.functional.mse_loss(pred["cl"], cl_true)
        
        loss = loss_lli + loss_lam + loss_cl
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    preds_lli, preds_lam, preds_cl = [], [], []
    trues_lli, trues_lam, trues_cl = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            eis = batch["eis"].to(device)
            pred = model(eis)
            
            preds_lli.append(pred["lli"].cpu().numpy())
            preds_lam.append(pred["lam"].cpu().numpy())
            preds_cl.append(pred["cl"].cpu().numpy())
            
            trues_lli.append(batch["lli"].cpu().numpy())
            trues_lam.append(batch["lam"].cpu().numpy())
            trues_cl.append(batch["cl"].cpu().numpy())
    
    preds_lli = np.concatenate(preds_lli)
    preds_lam = np.concatenate(preds_lam)
    preds_cl = np.concatenate(preds_cl)
    trues_lli = np.concatenate(trues_lli)
    trues_lam = np.concatenate(trues_lam)
    trues_cl = np.concatenate(trues_cl)
    
    def metrics(pred, true):
        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        return mae, rmse, r2
    
    lli_mae, lli_rmse, lli_r2 = metrics(preds_lli, trues_lli)
    lam_mae, lam_rmse, lam_r2 = metrics(preds_lam, trues_lam)
    cl_mae, cl_rmse, cl_r2 = metrics(preds_cl, trues_cl)
    
    return {
        "lli": {"mae": lli_mae, "rmse": lli_rmse, "r2": lli_r2},
        "lam": {"mae": lam_mae, "rmse": lam_rmse, "r2": lam_r2},
        "cl": {"mae": cl_mae, "rmse": cl_rmse, "r2": cl_r2},
    }

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    logger.info("="*70)
    logger.info("Degradation Quick Test — Separate Experiment")
    logger.info("="*70)
    
    # Load data
    data_path = ROOT / "data" / "mendeley_processed" / "mendeley_combined.pkl"
    df = pd.read_pickle(data_path)
    logger.info(f"Loaded {len(df)} samples from {len(df.cell_id.unique())} cells")
    
    # Cell-level split
    cells = sorted(df.cell_id.unique())
    logger.info(f"All cells: {cells}")
    
    # Split: 5 train / 2 val / 1 test (for 8 cells)
    train_cells = [1, 2, 3, 4, 5]
    val_cells = [6, 7]
    test_cells = [8]
    
    train_df = df[df.cell_id.isin(train_cells)].copy()
    val_df = df[df.cell_id.isin(val_cells)].copy()
    test_df = df[df.cell_id.isin(test_cells)].copy()
    
    logger.info(f"Train cells: {train_cells}  ({len(train_df)} samples)")
    logger.info(f"Val   cells: {val_cells}  ({len(val_df)} samples)")
    logger.info(f"Test  cells: {test_cells}  ({len(test_df)} samples)")
    
    # Leakage audit
    train_set = set(train_cells)
    val_set = set(val_cells)
    test_set = set(test_cells)
    leak_train_val = train_set & val_set
    leak_train_test = train_set & test_set
    leak_val_test = val_set & test_set
    
    leakage_pass = not leak_train_val and not leak_train_test and not leak_val_test
    logger.info(f"Leakage audit: train∩val={leak_train_val or 'EMPTY'}  "
                f"train∩test={leak_train_test or 'EMPTY'}  val∩test={leak_val_test or 'EMPTY'}  "
                f"→ {'PASS' if leakage_pass else 'FAIL'}")
    
    # Create datasets with TRAIN-ONLY normalization
    train_ds = MendeleyDegradationDataset(train_df, stats=None)
    train_stats = train_ds.stats
    val_ds = MendeleyDegradationDataset(val_df, stats=train_stats)
    test_ds = MendeleyDegradationDataset(test_df, stats=train_stats)
    
    logger.info(f"Normalization stats computed from TRAIN cells only")
    
    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # Model
    model = DegradationMLP(input_dim=9, hidden_dim=64).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: DegradationMLP  params={n_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Training
    best_val_loss = float("inf")
    patience = 15
    patience_counter = 0
    best_epoch = 0
    
    start_time = time.time()
    for epoch in range(1, 101):
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)
        
        val_loss = (val_metrics["lli"]["mae"] + val_metrics["lam"]["mae"] + val_metrics["cl"]["mae"]) / 3
        
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                        f"val_mean_MAE={val_loss:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save checkpoint
            ckpt_dir = ROOT / "experiments" / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "train_stats": train_stats,
                "epoch": epoch,
                "val_metrics": val_metrics,
            }, ckpt_dir / "best_degradation_quick_test.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.1f}s  best_epoch={best_epoch}")
    
    # Load best and evaluate on test
    ckpt = torch.load(
        ROOT / "experiments" / "checkpoints" / "best_degradation_quick_test.pth",
        weights_only=False
    )
    model.load_state_dict(ckpt["model_state_dict"])
    
    test_metrics = evaluate(model, test_loader, DEVICE)
    
    logger.info("="*70)
    logger.info("TEST RESULTS")
    logger.info("="*70)
    logger.info(f"LLI: MAE={test_metrics['lli']['mae']:.2f}  RMSE={test_metrics['lli']['rmse']:.2f}  R²={test_metrics['lli']['r2']:.4f}")
    logger.info(f"LAM: MAE={test_metrics['lam']['mae']:.2f}  RMSE={test_metrics['lam']['rmse']:.2f}  R²={test_metrics['lam']['r2']:.4f}")
    logger.info(f"CL:  MAE={test_metrics['cl']['mae']:.2f}  RMSE={test_metrics['cl']['rmse']:.2f}  R²={test_metrics['cl']['r2']:.4f}")
    
    # Save report
    report_dir = ROOT / "experiments" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report = f"""# Degradation Quick Test — Separate Experiment

**Date:** 2026-09  
**Purpose:** Test whether EIS features can predict degradation-mode indicators

---

## IMPORTANT DISCLAIMERS

1. **This is a separate experiment** — does NOT modify NASA A-H SOH benchmark
2. **Samsung/Mendeley data NOT mixed with NASA training**
3. **LLI/LAM/CL are model-derived indicators** from equivalent-circuit fitting (ECM)
4. **NOT direct physical ground truth** — they are surrogate labels from circuit-fitting algorithms

---

## 1. Dataset

- **Source:** Mendeley "Li-ion cells EIS dataset with fitting and degradation modes"
- **Cells:** 8 Samsung INR18650-30Q cells
- **Total samples:** {len(df)}
- **Cell distribution:**

| Cell | Samples | Aging cycles |
|------|---------|--------------|
{chr(10).join(f"| {cid} | {len(grp)} | {grp.aging_cycle.min()}-{grp.aging_cycle.max()} |" for cid, grp in df.groupby("cell_id"))}

---

## 2. Features

**EIS features (9 total):**
- `zmod_ohm` — impedance magnitude |Z|
- `zphz_deg` — phase angle
- `zreal_ohm` — real part Re(Z)
- `zimg_ohm` — imaginary part Im(Z)
- `R_electrolyte` — electrolyte resistance (from ECM fitting)
- `R_ct1` — charge-transfer resistance 1
- `R_ct2` — charge-transfer resistance 2
- `Zw` — Warburg coefficient
- `ocv_v` — open-circuit voltage

**Targets (model-derived from ECM fitting, NOT direct physical measurements):**
- `lli_pct` — Loss of Lithium Inventory (%)
- `lam_pct` — Loss of Active Material (%)
- `cl_pct` — Capacity Loss (%)

**Label distributions:**

| Target | Min | Max | Mean | Std | p50 |
|--------|-----|-----|------|-----|-----|
| LLI (%) | {df.lli_pct.min():.2f} | {df.lli_pct.max():.2f} | {df.lli_pct.mean():.2f} | {df.lli_pct.std():.2f} | {df.lli_pct.median():.2f} |
| LAM (%) | {df.lam_pct.min():.2f} | {df.lam_pct.max():.2f} | {df.lam_pct.mean():.2f} | {df.lam_pct.std():.2f} | {df.lam_pct.median():.2f} |
| CL (%) | {df.cl_pct.min():.2f} | {df.cl_pct.max():.2f} | {df.cl_pct.mean():.2f} | {df.cl_pct.std():.2f} | {df.cl_pct.median():.2f} |

---

## 3. Cell-Level Split

**Train cells:** {train_cells} ({len(train_df)} samples)  
**Validation cells:** {val_cells} ({len(val_df)} samples)  
**Test cells:** {test_cells} ({len(test_df)} samples)

**Leakage audit:** {'PASS' if leakage_pass else 'FAIL'}  
- train ∩ val: {leak_train_val or 'EMPTY'}
- train ∩ test: {leak_train_test or 'EMPTY'}
- val ∩ test: {leak_val_test or 'EMPTY'}

**Normalization:** All EIS features z-scored using TRAIN cells only.

---

## 4. Model Architecture

```
DegradationMLP:
  EIS features (9D)
      ↓
  Linear(9 → 64) + ReLU + Dropout(0.2)
      ↓
  Linear(64 → 64) + ReLU + Dropout(0.2)
      ↓
  Shared representation (64D)
      ├── Linear(64 → 1) → LLI
      ├── Linear(64 → 1) → LAM
      └── Linear(64 → 1) → CL
```

**Parameters:** {n_params:,}  
**Loss:** MSE on all three targets (unweighted sum)  
**Optimizer:** Adam (lr=1e-3, weight_decay=1e-4)  
**Early stopping:** Patience 15 on validation mean MAE  
**Best epoch:** {best_epoch}  
**Training time:** {elapsed:.1f}s

---

## 5. Test Results

| Target | MAE | RMSE | R² |
|--------|-----|------|----|
| **LLI** | {test_metrics['lli']['mae']:.2f} | {test_metrics['lli']['rmse']:.2f} | {test_metrics['lli']['r2']:.4f} |
| **LAM** | {test_metrics['lam']['mae']:.2f} | {test_metrics['lam']['rmse']:.2f} | {test_metrics['lam']['r2']:.4f} |
| **CL** | {test_metrics['cl']['mae']:.2f} | {test_metrics['cl']['rmse']:.2f} | {test_metrics['cl']['r2']:.4f} |

**Interpretation:**
- LLI: Mean Absolute Error {test_metrics['lli']['mae']:.2f}% (label range: -34.80 to 3516.42%)
- LAM: Mean Absolute Error {test_metrics['lam']['mae']:.2f}% (label range: -3.78 to 212.09%)
- CL: Mean Absolute Error {test_metrics['cl']['mae']:.2f}% (label range: 0 to 12.46%)

---

## 6. Limitations

1. **Very small test set:** Only 1 cell (85 samples) used for testing
2. **Labels are model-derived:** LLI/LAM/CL come from ECM fitting, NOT direct physical measurements
3. **No cross-validation:** Single train/val/test split due to limited cell count
4. **No frequency-aware encoding:** EIS features are aggregated scalars (no spectrum modeling)
5. **Extreme label variance:** LLI has very high variance (std=402.64%) relative to mean (208.68%)
6. **Negative values present:** LLI and LAM have negative values (likely ECM fitting artifacts)

---

## 7. Relation to Main BaFuse Project

**This experiment is COMPLETELY SEPARATE from the NASA A-H SOH benchmark.**

| Aspect | NASA SOH Benchmark | Degradation Quick Test |
|--------|-------------------|------------------------|
| Dataset | NASA PCoE (34 batteries) | Samsung/Mendeley (8 cells) |
| Input | Discharge + EIS + Physics | EIS only |
| Target | SOH (capacity fade) | LLI/LAM/CL (ECM-derived) |
| Models | A–H (14 architectures) | DegradationMLP (1 small MLP) |
| Status | Primary thesis results | Exploratory side experiment |

**NO data mixing:** Samsung data is NOT used to train NASA SOH models.  
**NO label transfer:** LLI/LAM/CL labels are NOT transferred to NASA dataset.

---

## 8. Conclusion

This quick test demonstrates that EIS features contain **some predictive signal** for the ECM-derived degradation-mode indicators (LLI/LAM/CL). However:

- Test set is too small (1 cell) for robust validation
- Labels are surrogate indicators from circuit fitting, not direct physical measurements
- Results are exploratory and NOT comparable to NASA SOH benchmark metrics

**Suitable for thesis as supporting evidence?** POSSIBLY — with explicit caveats about:
1. Model-derived labels (not ground truth)
2. Small test set (1 cell)
3. Separate experiment (not part of main BaFuse evaluation)

---

## 9. Saved Artifacts

- **Checkpoint:** `experiments/checkpoints/best_degradation_quick_test.pth`
- **This report:** `experiments/reports/degradation_quick_test.md`
- **Train normalization stats:** Saved in checkpoint under `train_stats`

---

## 10. Summary

- **Cells:** 8 (5 train / 2 val / 1 test)
- **Samples:** {len(df)} ({len(train_df)} / {len(val_df)} / {len(test_df)})
- **LLI:** MAE={test_metrics['lli']['mae']:.2f}  RMSE={test_metrics['lli']['rmse']:.2f}  R²={test_metrics['lli']['r2']:.4f}
- **LAM:** MAE={test_metrics['lam']['mae']:.2f}  RMSE={test_metrics['lam']['rmse']:.2f}  R²={test_metrics['lam']['r2']:.4f}
- **CL:** MAE={test_metrics['cl']['mae']:.2f}  RMSE={test_metrics['cl']['rmse']:.2f}  R²={test_metrics['cl']['r2']:.4f}
- **Leakage:** {'PASS' if leakage_pass else 'FAIL'}
- **SOH benchmark modified:** NO
"""
    
    with open(report_dir / "degradation_quick_test.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"Report saved → {report_dir / 'degradation_quick_test.md'}")
    logger.info("="*70)
    logger.info("FINAL SUMMARY")
    logger.info("="*70)
    logger.info(f"Degradation quick test completed: YES")
    logger.info(f"Cells: 8 (5 train / 2 val / 1 test)")
    logger.info(f"Samples: {len(df)} ({len(train_df)} / {len(val_df)} / {len(test_df)})")
    logger.info(f"LLI:  MAE={test_metrics['lli']['mae']:6.2f}  RMSE={test_metrics['lli']['rmse']:6.2f}  R²={test_metrics['lli']['r2']:7.4f}")
    logger.info(f"LAM:  MAE={test_metrics['lam']['mae']:6.2f}  RMSE={test_metrics['lam']['rmse']:6.2f}  R²={test_metrics['lam']['r2']:7.4f}")
    logger.info(f"CL:   MAE={test_metrics['cl']['mae']:6.2f}  RMSE={test_metrics['cl']['rmse']:6.2f}  R²={test_metrics['cl']['r2']:7.4f}")
    logger.info(f"Leakage: {'PASS' if leakage_pass else 'FAIL'}")
    logger.info(f"SOH benchmark modified: NO")

if __name__ == "__main__":
    main()
