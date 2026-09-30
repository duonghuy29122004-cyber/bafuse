"""
BaFuse — Physical Degradation Analysis Layer.

NEW DIRECTION (2026-09):
  Battery degradation analysis is now performed by observing PHYSICAL SIGNAL
  CHANGES from the NASA-trained SOH model. It does NOT use supervised
  LLI/LAM/CL labels from Samsung/Mendeley.

  Samsung/Mendeley is used ONLY as an independent external test dataset
  (SOH generalization check). It is never used for training, normalization,
  model selection, or degradation labeling of NASA cells.

Architecture of analysis:

    NASA battery  ─→  BaFuse SOH model  ─→  soh_pred (trajectory)
                                                   │
                                                   ▼
                              Physical Signal Extractor
                         (from model internal representations)
                                   │         │         │
                              V_droop    Imp_rise   EIS_trend
                                   │         │         │
                                   └─────────┴─────────┘
                                             │
                                             ▼
                              Literature RAG Retriever
                                (rag/retriever.py)
                                             │
                                             ▼
                              Degradation Mechanism Report
                             (plausible mechanisms, indicators,
                              conditions, references)

IMPORTANT:
  - All degradation analysis is INTERPRETIVE, not supervised.
  - Physical signals (impedance rise, voltage droop, EIS trends) are
    OBSERVABLE measurements from NASA data.
  - Retrieved mechanisms are plausible candidates from scientific literature.
  - Do NOT claim mechanisms are definitively confirmed from model outputs alone.
  - NASA data does NOT have ground-truth LLI/LAM/CL labels.
  - Do NOT fabricate LLI/LAM/CL for NASA.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PhysicalSignalExtractor:
    """
    Extracts physical degradation signals from battery cycle data.

    Uses only OBSERVABLE measurements from NASA paired_df:
      - impedance_ohm  : |Z| at each cycle
      - voltage_min    : minimum discharge voltage
      - voltage_mean   : mean discharge voltage
      - capacity_ahr   : measured discharge capacity

    Computes normalised changes vs. beginning-of-life (BOL):
      - impedance_rise  : (Z_cycle - Z_bol) / Z_bol
      - voltage_droop   : (Vmin_cycle - Vmin_bol) / (4.2 - Vcutoff)
      - capacity_fade   : 1 - (Q_cycle / Q_bol)
      - re_change       : (Re_cycle - Re_bol) / Re_bol  (if re_ohm available)
      - rct_change      : (Rct_cycle - Rct_bol) / Rct_bol  (if rct_ohm available)

    All signals are dimensionless ratios — comparable across batteries
    with different nominal specs.
    """

    def __init__(self, bol_cycles: int = 3):
        """
        Args:
            bol_cycles : Number of early cycles used to estimate BOL baseline.
                         Mean of the first `bol_cycles` valid cycles is used.
        """
        self.bol_cycles = bol_cycles

    def extract(self, battery_df: pd.DataFrame,
                cutoff_v: float = 2.7,
                nominal_capacity_ah: float = 2.0) -> pd.DataFrame:
        """
        Compute normalised physical signals for one battery's paired_df subset.

        Args:
            battery_df          : Rows from paired_df for one battery_id,
                                  sorted by discharge_cycle.
            cutoff_v            : Discharge cutoff voltage for this battery.
            nominal_capacity_ah : Rated nominal capacity (Ah).

        Returns:
            DataFrame with columns:
                discharge_cycle, soh,
                impedance_rise, voltage_droop, capacity_fade,
                re_change (if available), rct_change (if available)
        """
        df = battery_df.sort_values("discharge_cycle").copy()
        if df.empty:
            return pd.DataFrame()

        # BOL baseline (first bol_cycles cycles)
        bol = df.head(self.bol_cycles)

        z_bol  = float(bol["impedance_ohm"].mean()) if "impedance_ohm" in df.columns else None
        vd_bol = float(bol["voltage_min"].mean())   if "voltage_min"    in df.columns else None
        q_bol  = float(bol["capacity_ahr"].mean())  if "capacity_ahr"   in df.columns else None
        re_bol = float(bol["re_ohm"].mean())        if ("re_ohm" in df.columns and
                                                         bol["re_ohm"].notna().any()) else None
        rct_bol= float(bol["rct_ohm"].mean())       if ("rct_ohm" in df.columns and
                                                         bol["rct_ohm"].notna().any()) else None

        v_range = max(4.2 - cutoff_v, 0.1)

        result_rows = []
        for _, row in df.iterrows():
            signals = {
                "discharge_cycle": int(row["discharge_cycle"]),
                "soh": float(np.clip(row["capacity_ahr"] / nominal_capacity_ah, 0, 1.1))
                       if "capacity_ahr" in row else np.nan,
            }

            # Impedance rise
            if z_bol and z_bol > 0 and "impedance_ohm" in row:
                signals["impedance_rise"] = float(
                    (row["impedance_ohm"] - z_bol) / z_bol
                )
            else:
                signals["impedance_rise"] = np.nan

            # Voltage droop (normalised)
            if vd_bol is not None and "voltage_min" in row:
                signals["voltage_droop"] = float(
                    (row["voltage_min"] - vd_bol) / v_range
                )
            else:
                signals["voltage_droop"] = np.nan

            # Capacity fade
            if q_bol and q_bol > 0 and "capacity_ahr" in row:
                signals["capacity_fade"] = float(
                    1.0 - row["capacity_ahr"] / q_bol
                )
            else:
                signals["capacity_fade"] = np.nan

            # Re change
            if re_bol and re_bol > 0 and "re_ohm" in row and pd.notna(row.get("re_ohm")):
                signals["re_change"] = float(
                    (row["re_ohm"] - re_bol) / re_bol
                )
            else:
                signals["re_change"] = np.nan

            # Rct change
            if rct_bol and rct_bol > 0 and "rct_ohm" in row and pd.notna(row.get("rct_ohm")):
                signals["rct_change"] = float(
                    (row["rct_ohm"] - rct_bol) / rct_bol
                )
            else:
                signals["rct_change"] = np.nan

            result_rows.append(signals)

        return pd.DataFrame(result_rows)

    def extract_final_state(
        self, battery_df: pd.DataFrame,
        cutoff_v: float = 2.7,
        nominal_capacity_ah: float = 2.0,
        last_cycles: int = 5,
    ) -> Dict[str, float]:
        """
        Summarise the END-OF-LIFE physical signals (mean of last `last_cycles` cycles).
        Returns a flat dict suitable for passing to PhysicalDegradationExplainer.
        """
        signals_df = self.extract(battery_df, cutoff_v, nominal_capacity_ah)
        if signals_df.empty:
            return {}

        eol = signals_df.tail(last_cycles)
        return {
            "impedance_rise": float(eol["impedance_rise"].mean()),
            "voltage_droop":  float(eol["voltage_droop"].mean()),
            "capacity_fade":  float(eol["capacity_fade"].mean()),
            "eis_re_change":  float(eol["re_change"].mean())  if "re_change"  in eol else 0.0,
            "eis_rct_change": float(eol["rct_change"].mean()) if "rct_change" in eol else 0.0,
        }


class BatteryDegradationAnalyzer:
    """
    Full pipeline: NASA paired_df → physical signal extraction →
    literature-based degradation mechanism retrieval.

    This is the top-level analysis object for the new research direction.

    Usage:
        from src.degradation_analysis import BatteryDegradationAnalyzer
        from src.data.dataset import BATTERY_CUTOFF_VOLTAGE, get_nominal_capacity

        analyzer = BatteryDegradationAnalyzer()

        # Analyse one battery
        report = analyzer.analyze_battery(
            battery_df=paired_df[paired_df.battery_id == "B0005"],
            battery_id="B0005",
        )
        print(analyzer.format_report(report))

        # Analyse all batteries
        reports = analyzer.analyze_dataset(paired_df)
    """

    def __init__(self, bol_cycles: int = 3, last_cycles: int = 5, top_k: int = 2):
        from rag.retriever import DegradationExplainer
        self.extractor = PhysicalSignalExtractor(bol_cycles=bol_cycles)
        self.explainer = DegradationExplainer(top_k=top_k)
        self.last_cycles = last_cycles

    def analyze_battery(
        self,
        battery_df: pd.DataFrame,
        battery_id: str,
        cell_info:  Optional[Dict] = None,
    ) -> Dict:
        """
        Analyse one battery's degradation trajectory.

        Returns a report dict with:
            battery_id, signal_summary, mechanisms, disclaimer
        """
        from src.data.dataset import BATTERY_CUTOFF_VOLTAGE, get_nominal_capacity

        cutoff_v  = BATTERY_CUTOFF_VOLTAGE.get(str(battery_id), 2.7)
        nominal   = get_nominal_capacity(str(battery_id))

        # Extract signals
        signals_df = self.extractor.extract(battery_df, cutoff_v, nominal)
        final_signals = self.extractor.extract_final_state(
            battery_df, cutoff_v, nominal, self.last_cycles
        )

        if not final_signals:
            return {"battery_id": battery_id, "error": "insufficient data"}

        # Retrieve plausible mechanisms via RAG
        rag_result = self.explainer.explain_from_signals(
            impedance_rise = final_signals.get("impedance_rise", 0.0),
            voltage_droop  = final_signals.get("voltage_droop",  0.0),
            capacity_fade  = final_signals.get("capacity_fade",  0.0),
            eis_re_change  = final_signals.get("eis_re_change",  0.0),
            eis_rct_change = final_signals.get("eis_rct_change", 0.0),
            cell_info      = cell_info,
        )

        return {
            "battery_id":      battery_id,
            "final_signals":   final_signals,
            "n_cycles":        len(signals_df),
            "final_soh":       float(signals_df["soh"].iloc[-1]) if not signals_df.empty else np.nan,
            "rag_result":      rag_result,
            "disclaimer": (
                "Physical degradation analysis based on observable signal changes "
                "(impedance rise, voltage droop, capacity fade) from NASA battery data. "
                "Degradation mechanisms are plausible literature-based candidates. "
                "No LLI/LAM/CL labels from Samsung/Mendeley are used."
            ),
        }

    def analyze_dataset(
        self,
        paired_df: pd.DataFrame,
        cell_info_map: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Dict]:
        """
        Analyse all batteries in paired_df.

        Returns:
            Dict mapping battery_id → report dict.
        """
        reports = {}
        for bid, grp in paired_df.groupby("battery_id"):
            ci = (cell_info_map or {}).get(str(bid))
            try:
                reports[str(bid)] = self.analyze_battery(grp, str(bid), cell_info=ci)
                logger.info(
                    f"  {bid}: SOH={reports[bid].get('final_soh', '?'):.3f}  "
                    f"imp_rise={reports[bid]['final_signals'].get('impedance_rise', 0):.3f}  "
                    f"dominant={reports[bid]['rag_result'].get('dominant_mode', '?')}"
                )
            except Exception as e:
                logger.warning(f"  {bid}: analysis failed — {e}")
                reports[str(bid)] = {"battery_id": bid, "error": str(e)}
        return reports

    def format_report(self, report: Dict) -> str:
        """Format one battery's degradation report as a readable string."""
        lines = ["=" * 60,
                 f"Battery Degradation Analysis — {report.get('battery_id', '?')}",
                 "=" * 60]

        if "error" in report:
            lines.append(f"  ERROR: {report['error']}")
            return "\n".join(lines)

        sig = report.get("final_signals", {})
        lines += [
            f"  Cycles analysed : {report.get('n_cycles', '?')}",
            f"  Final SOH       : {report.get('final_soh', float('nan')):.3f}",
            f"  Impedance rise  : {sig.get('impedance_rise', float('nan')):.3f}",
            f"  Voltage droop   : {sig.get('voltage_droop',  float('nan')):.3f}",
            f"  Capacity fade   : {sig.get('capacity_fade',  float('nan')):.3f}",
            f"  Re(Z) change    : {sig.get('eis_re_change',  float('nan')):.3f}",
            "",
        ]

        rag = report.get("rag_result", {})
        if rag:
            from rag.retriever import DegradationExplainer
            explainer = DegradationExplainer()
            lines.append(explainer.format_explanation(rag, show_references=False))

        lines += ["", report.get("disclaimer", ""), "=" * 60]
        return "\n".join(lines)
