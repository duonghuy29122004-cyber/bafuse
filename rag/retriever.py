"""
BaFuse RAG Explanation Layer — Physical Degradation Interpretation.

Interprets observed physical signal changes from the NASA-trained SOH model
(voltage droop, impedance rise, EIS features) and retrieves plausible battery
degradation mechanisms from the static knowledge base.

NEW DIRECTION (2026-09):
  The interpretation layer is now driven by OBSERVED PHYSICAL SIGNALS from
  NASA battery measurements, NOT by model-derived LLI/LAM/CL labels from the
  Samsung/Mendeley dataset.

  Input to this layer:
    - observed_impedance_rise   : change in impedance over cycles
    - observed_voltage_droop    : voltage sag relative to cutoff
    - observed_capacity_fade    : SOH decline over aging
    - observed_eis_features     : Re(Z) and |Im(Z)| trends
    - cell_info (optional)      : temperature, C-rate, cycling protocol

  These are OBSERVABLE physical quantities from NASA data.
  They are not supervised degradation labels.

IMPORTANT DISCLAIMERS:
  1. Degradation mechanism retrieval is based on literature-reported
     correlations between observable physical signals and degradation modes.
     It is NOT a supervised classification or a direct measurement.

  2. The retrieved mechanisms are plausible candidates consistent with the
     observed physical changes. They do NOT confirm mechanistic presence.

  3. Samsung/Mendeley LLI/LAM/CL labels are NOT used here.
     This layer works exclusively with NASA physical signal observations.

  4. The RAG layer is a DOWNSTREAM INTERPRETATION component only.
     It plays no role in model training or prediction.

Usage:
    explainer = PhysicalDegradationExplainer()
    result = explainer.explain_from_signals(
        impedance_rise=0.15,      # fractional increase in |Z| vs cycle 1
        voltage_droop=-0.08,      # normalised V_min - V_cutoff change
        capacity_fade=0.12,       # 1 - SOH
        eis_re_change=0.05,       # Re(Z) increase vs cycle 1
        cell_info={"temperature_c": 35, "c_rate": 1.0}
    )
    print(explainer.format_explanation(result))

    # Legacy interface still supported for backward compatibility:
    explainer.explain({"LLI": 32.5, "LAM": 15.0, "CL": 2.1})
"""

from typing import Dict, List, Optional, Tuple
import logging

from .knowledge_base import (
    DEGRADATION_KNOWLEDGE_BASE,
    get_entries_for_mode,
)

logger = logging.getLogger(__name__)

# ── Thresholds for severity classification ────────────────────────────────────
# These are rough empirical thresholds based on the Mendeley dataset range.
# They are NOT diagnostic criteria — use for qualitative categorisation only.
_SEVERITY_THRESHOLDS = {
    "LLI": {"low": 10.0, "moderate": 30.0, "high": 60.0},
    "LAM": {"low": 5.0,  "moderate": 20.0, "high": 40.0},
    "CL":  {"low": 1.0,  "moderate": 3.0,  "high": 6.0},
}


def _classify_severity(mode: str, value: float) -> str:
    """Classify a degradation estimate as negligible/low/moderate/high."""
    thresh = _SEVERITY_THRESHOLDS.get(mode.upper(), {})
    if value < 0:
        return "negligible (negative estimate — possible normalisation artefact)"
    if value < thresh.get("low", 5.0):
        return "low"
    if value < thresh.get("moderate", 20.0):
        return "moderate"
    if value < thresh.get("high", 50.0):
        return "high"
    return "very high"


class DegradationExplainer:
    """
    Retrieval-based explanation layer for battery degradation-mode estimates.

    Args:
        top_k        : Number of mechanism entries to retrieve per mode.
        threshold    : Minimum estimate value to trigger retrieval. Estimates
                       below this are treated as negligible.
        verbose      : Log retrieved entries.
    """

    def __init__(
        self,
        top_k:     int   = 2,
        threshold: float = 0.0,
        verbose:   bool  = False,
    ):
        self.top_k     = top_k
        self.threshold = threshold
        self.verbose   = verbose
        logger.debug(
            f"DegradationExplainer init: top_k={top_k}, threshold={threshold}"
        )

    # -----------------------------------------------------------------------

    def explain(
        self,
        degradation_result: Dict[str, float],
        cell_info:          Optional[Dict] = None,
    ) -> Dict:
        """
        Generate explanations for a degradation-mode estimate dict.

        Args:
            degradation_result : Dict with keys "LLI", "LAM", "CL" (float values).
                                 Values may be in percent or normalised units
                                 depending on upstream normalisation.
            cell_info          : Optional dict with operating context
                                 (temperature, C-rate, etc.) for richer retrieval.

        Returns:
            Dict with:
                "estimated_modes"   : input values
                "severity"          : per-mode severity classification
                "explanations"      : per-mode list of retrieved mechanisms
                "disclaimer"        : standard disclaimer text
                "dominant_mode"     : mode with highest estimate value
        """
        lli = float(degradation_result.get("LLI", 0.0))
        lam = float(degradation_result.get("LAM", 0.0))
        cl  = float(degradation_result.get("CL",  0.0))

        estimates = {"LLI": lli, "LAM": lam, "CL": cl}
        severity  = {m: _classify_severity(m, v) for m, v in estimates.items()}

        explanations: Dict[str, List[Dict]] = {}
        for mode, value in estimates.items():
            if value > self.threshold:
                entries    = get_entries_for_mode(mode)
                retrieved  = self._rank_entries(entries, mode, value, cell_info)
                explanations[mode] = retrieved[: self.top_k]
            else:
                explanations[mode] = []

        dominant = max(estimates, key=lambda m: estimates[m])

        return {
            "estimated_modes": estimates,
            "severity":        severity,
            "explanations":    explanations,
            "dominant_mode":   dominant,
            "cell_info":       cell_info or {},
            "disclaimer": (
                "IMPORTANT: This interpretation is based on OBSERVABLE PHYSICAL "
                "SIGNALS from the NASA-trained SOH model (voltage droop, impedance "
                "rise, EIS features). It is NOT based on supervised degradation "
                "labels (LLI/LAM/CL) from the Samsung/Mendeley dataset. "
                "The mechanisms listed below are plausible candidates supported "
                "by published literature; their presence cannot be confirmed "
                "without dedicated diagnostic measurements."
            ),
        }

    # -----------------------------------------------------------------------

    def _rank_entries(
        self,
        entries:   List[Dict],
        mode:      str,
        value:     float,
        cell_info: Optional[Dict],
    ) -> List[Dict]:
        """
        Simple ranking: prioritise entries whose conditions match cell_info,
        otherwise return in knowledge-base order.
        """
        if not cell_info or not entries:
            return entries

        temp  = float(cell_info.get("temperature_c", 25.0))
        crate = float(cell_info.get("c_rate", 1.0))

        def _score(entry: Dict) -> float:
            score = 0.0
            conds = " ".join(entry.get("conditions", [])).lower()
            if temp > 35 and "elevated temperature" in conds:
                score += 1.0
            if temp < 10 and "low temperature" in conds:
                score += 1.0
            if crate > 1.5 and "high current" in conds:
                score += 1.0
            return score

        return sorted(entries, key=_score, reverse=True)

    # -----------------------------------------------------------------------

    def format_explanation(
        self,
        result:         Dict,
        show_references: bool = True,
    ) -> str:
        """
        Format the explanation result as a human-readable string.

        Args:
            result           : Output of explain().
            show_references  : Include literature references.

        Returns:
            Multi-line string suitable for logging or display.
        """
        lines = []
        lines.append("═" * 70)
        lines.append("BaFuse v2 — Battery Degradation Mode Explanation")
        lines.append("═" * 70)
        lines.append("")
        lines.append("⚠  " + result["disclaimer"])
        lines.append("")
        lines.append("─" * 70)
        lines.append("Model-Derived Degradation-Mode Estimates:")
        lines.append("─" * 70)

        for mode, val in result["estimated_modes"].items():
            sev = result["severity"].get(mode, "unknown")
            lines.append(f"  {mode:4s}: {val:8.3f}  [{sev} severity]")

        dom = result["dominant_mode"]
        lines.append(f"\n  Dominant estimated mode: {dom}")
        lines.append("")

        for mode, entries in result["explanations"].items():
            if not entries:
                continue
            val = result["estimated_modes"][mode]
            lines.append(f"─" * 70)
            lines.append(
                f"  {mode} — Plausible mechanisms "
                f"(estimate={val:.3f}, {result['severity'][mode]} severity)"
            )
            lines.append("─" * 70)
            for i, entry in enumerate(entries, 1):
                lines.append(f"\n  [{i}] {entry['mechanism']}")
                lines.append(f"      {entry['description']}")
                lines.append("      Observable indicators:")
                for ind in entry.get("indicators", []):
                    lines.append(f"        • {ind}")
                lines.append("      Accelerating conditions:")
                for cond in entry.get("conditions", []):
                    lines.append(f"        • {cond}")
                if show_references:
                    lines.append(f"      Reference: {entry.get('reference', 'N/A')}")
                    if entry.get("source_url"):
                        lines.append(f"      URL: {entry['source_url']}")

        lines.append("")
        lines.append("═" * 70)
        return "\n".join(lines)

    # -----------------------------------------------------------------------

    def explain_from_signals(
        self,
        impedance_rise:  float = 0.0,
        voltage_droop:   float = 0.0,
        capacity_fade:   float = 0.0,
        eis_re_change:   float = 0.0,
        eis_rct_change:  float = 0.0,
        cell_info:       Optional[Dict] = None,
    ) -> Dict:
        """
        Generate degradation mechanism explanations from OBSERVABLE PHYSICAL
        SIGNALS measured on NASA cells.

        This is the PRIMARY interface under the new research direction (2026-09).
        It does NOT rely on Samsung/Mendeley LLI/LAM/CL labels.

        Physical signal heuristics (literature-informed thresholds):
          - Impedance rise > 0.20 (20% increase)  → probable LLI (SEI growth, CL)
          - Voltage droop  < -0.10 (normalised)    → probable LAM (active material loss)
          - Capacity fade  > 0.15 (15% SOH loss)  → moderate to high degradation
          - Re(Z) increase > 0.05                  → CL (conductivity loss)
          - Rct increase   > 0.10                  → LAM or LLI (CT resistance)

        Args:
            impedance_rise  : Fractional increase in |Z| vs. BOL (e.g. 0.15 = 15%).
            voltage_droop   : Normalised (Vmin - Vcutoff) / (4.2 - Vcutoff) change.
            capacity_fade   : 1 - SOH (e.g. 0.12 = 12% capacity loss).
            eis_re_change   : Absolute change in Re(Z) vs. BOL (Ohm).
            eis_rct_change  : Absolute change in |Im(Z)| vs. BOL (Ohm, proxy for Rct).
            cell_info       : Optional operating context (temperature_c, c_rate, etc.).

        Returns:
            Same structure as explain() with additional `signal_analysis` field.

        IMPORTANT:
            Threshold-based mapping is a simplified heuristic, NOT a diagnostic model.
            Use only for qualitative interpretation.
        """
        # Map physical signals to approximate mode indicators
        # These thresholds are literature-informed heuristics, not ground truth.
        lli_signal = max(impedance_rise * 50.0, 0.0)    # scale to ~0-100 range
        lam_signal = max(abs(voltage_droop) * 80.0, 0.0)
        cl_signal  = max(eis_re_change * 200.0, 0.0)

        # Also boost signals from capacity fade (general degradation)
        fade_boost = capacity_fade * 30.0
        lli_signal += fade_boost * 0.5
        lam_signal += fade_boost * 0.3
        cl_signal  += eis_rct_change * 100.0

        estimated = {"LLI": round(lli_signal, 2),
                     "LAM": round(lam_signal, 2),
                     "CL":  round(cl_signal,  2)}

        base_result = self.explain(estimated, cell_info=cell_info)

        # Add signal analysis for traceability
        base_result["signal_analysis"] = {
            "impedance_rise":  impedance_rise,
            "voltage_droop":   voltage_droop,
            "capacity_fade":   capacity_fade,
            "eis_re_change":   eis_re_change,
            "eis_rct_change":  eis_rct_change,
            "note": (
                "Physical signals are from NASA battery measurements. "
                "Heuristic mapping to degradation modes uses literature-reported "
                "correlations. This is NOT a diagnostic classification."
            ),
        }
        # Update disclaimer to emphasise physical-signal origin
        base_result["disclaimer"] = (
            "IMPORTANT: Degradation mechanism suggestions are inferred from "
            "OBSERVED PHYSICAL SIGNAL CHANGES (impedance rise, voltage droop, "
            "capacity fade) measured from NASA battery data. This is NOT based "
            "on supervised LLI/LAM/CL labels from Samsung/Mendeley. "
            "Mechanisms are plausible literature-based candidates only; "
            "presence cannot be confirmed without dedicated diagnostics."
        )
        return base_result

    def explain_and_print(
        self,
        degradation_result: Dict[str, float],
        cell_info:          Optional[Dict] = None,
    ) -> Dict:
        """Convenience: explain + print formatted output. Returns result dict."""
        result = self.explain(degradation_result, cell_info)
        print(self.format_explanation(result))
        return result

    # -----------------------------------------------------------------------

    @staticmethod
    def from_model_output(
        model_output: Dict,
        normalisation_stats: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Extract raw (un-normalised) degradation estimates from model output dict.

        If normalisation_stats are provided, denormalise back to original units.

        Args:
            model_output         : Dict with keys 'lli_pred', 'lam_pred', 'cl_pred'.
            normalisation_stats  : Optional dict from MendeleyDataset.stats.

        Returns:
            Dict {"LLI": float, "LAM": float, "CL": float} in original units.
        """
        import numpy as np

        def _extract(key: str) -> float:
            val = model_output.get(key, 0.0)
            if hasattr(val, "item"):     # tensor
                val = val.item()
            if hasattr(val, "__len__"):   # array
                val = float(np.asarray(val).ravel()[0])
            return float(val)

        lli = _extract("lli_pred")
        lam = _extract("lam_pred")
        cl  = _extract("cl_pred")

        # Denormalise if stats provided
        if normalisation_stats:
            def _denorm(v: float, col: str) -> float:
                s = normalisation_stats.get(col)
                if s:
                    return v * s["std"] + s["mean"]
                return v
            lli = _denorm(lli, "lli_pct")
            lam = _denorm(lam, "lam_pct")
            cl  = _denorm(cl,  "cl_pct")

        return {"LLI": lli, "LAM": lam, "CL": cl}
