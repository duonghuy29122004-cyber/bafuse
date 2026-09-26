"""
Preprocess Mendeley EIS dataset for BaFuse v2.

Reads all EISexpCell*.xlsx and Circuit_parameter_Cell*.xlsx files,
merges them into a clean DataFrame, applies normalisation, and saves
processed artefacts for downstream training.

Output:
    data/mendeley_processed/
        mendeley_combined.pkl    — full merged DataFrame
        mendeley_train.pkl       — training split
        mendeley_val.pkl         — validation split
        mendeley_test.pkl        — test split
        mendeley_stats.json      — normalisation statistics (train-set only)
        mendeley_summary.txt     — human-readable dataset summary

Usage:
    python scripts/preprocess_mendeley.py
    python scripts/preprocess_mendeley.py --mendeley_dir "path/to/dataset"
    python scripts/preprocess_mendeley.py --soc_filter 100 --out_dir data/mendeley_processed
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(args):
    from src.data.mendeley_dataset import (
        build_mendeley_dataframe,
        split_mendeley_by_cell,
        compute_mendeley_stats,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Parse all cells ───────────────────────────────────────────────────
    logger.info(f"Loading Mendeley data from: {args.mendeley_dir}")
    df = build_mendeley_dataframe(
        mendeley_root=args.mendeley_dir,
        soc_filter=args.soc_filter,
        num_cells=args.num_cells,
    )
    logger.info(f"Combined dataset: {len(df)} samples, {df['cell_id'].nunique()} cells")
    logger.info(f"Aging cycle range: {df['aging_cycle'].min()} – {df['aging_cycle'].max()}")
    logger.info(f"SOH range: {df['soh_norm'].min():.3f} – {df['soh_norm'].max():.3f}")

    # ── 2. Split by cell ─────────────────────────────────────────────────────
    train_cells = args.train_cells if args.train_cells else None
    val_cells   = args.val_cells   if args.val_cells   else None
    test_cells  = args.test_cells  if args.test_cells  else None

    train_df, val_df, test_df = split_mendeley_by_cell(
        df,
        train_cells=train_cells,
        val_cells=val_cells,
        test_cells=test_cells,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    # ── 3. Compute normalisation stats (train only) ──────────────────────────
    stats = compute_mendeley_stats(train_df)

    # ── 4. Save artefacts ───────────────────────────────────────────────────
    df.to_pickle(str(out_dir / "mendeley_combined.pkl"))
    train_df.to_pickle(str(out_dir / "mendeley_train.pkl"))
    val_df.to_pickle(str(out_dir / "mendeley_val.pkl"))
    test_df.to_pickle(str(out_dir / "mendeley_test.pkl"))

    with open(out_dir / "mendeley_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ── 5. Summary ──────────────────────────────────────────────────────────
    SEP = "=" * 60
    summary_lines = [
        SEP,
        "Mendeley EIS Dataset - Preprocessing Summary",
        SEP,
        f"  Source          : {args.mendeley_dir}",
        f"  SOC filter      : {args.soc_filter}%",
        f"  Total samples   : {len(df)}",
        f"  Cells           : {sorted(df['cell_id'].unique().tolist())}",
        f"  Train samples   : {len(train_df)}  cells={sorted(train_df['cell_id'].unique().tolist())}",
        f"  Val samples     : {len(val_df)}  cells={sorted(val_df['cell_id'].unique().tolist())}",
        f"  Test samples    : {len(test_df)}  cells={sorted(test_df['cell_id'].unique().tolist())}",
        "",
        "  Columns         : " + ", ".join(df.columns.tolist()),
        "",
        "  SOH norm range  : {:.3f} - {:.3f}".format(
            float(df["soh_norm"].min()), float(df["soh_norm"].max())),
        ("  LLI range (%)   : {:.2f} - {:.2f}".format(
            float(df["lli_pct"].min()),  float(df["lli_pct"].max()))
            if "lli_pct" in df.columns else "  LLI : not available"),
        ("  LAM range (%)   : {:.2f} - {:.2f}".format(
            float(df["lam_pct"].min()),  float(df["lam_pct"].max()))
            if "lam_pct" in df.columns else "  LAM : not available"),
        ("  CL  range (%)   : {:.2f} - {:.2f}".format(
            float(df["cl_pct"].min()),   float(df["cl_pct"].max()))
            if "cl_pct"  in df.columns else "  CL  : not available"),
        "",
        "IMPORTANT: LLI/LAM/CL values are model-derived degradation-mode",
        "estimates from ECM fitting. They are NOT physical ground truth.",
        SEP,
    ]
    summary = "\n".join(summary_lines)
    # Print safely — replace unmappable chars on Windows cp1252 terminals
    print(summary.encode("ascii", errors="replace").decode("ascii"))
    with open(out_dir / "mendeley_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    logger.info(f"Mendeley preprocessing complete. Output → {out_dir}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Preprocess Mendeley EIS dataset")
    p.add_argument(
        "--mendeley_dir",
        default=(
            "Lithium-ion cells EIS dataset with fitting and deg/"
            "Lithium-ion cells EIS dataset with fitting and deg"
        ),
    )
    p.add_argument("--out_dir",    default="data/mendeley_processed")
    p.add_argument("--soc_filter", type=int, default=100,
                   help="Only use rows with this SOC value (default 100)")
    p.add_argument("--num_cells",  type=int, default=8)
    p.add_argument("--train_cells", type=int, nargs="+",
                   default=None, help="Cell IDs for training (default: auto)")
    p.add_argument("--val_cells",   type=int, nargs="+", default=None)
    p.add_argument("--test_cells",  type=int, nargs="+", default=None)
    sys.exit(main(p.parse_args()))
