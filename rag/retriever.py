"""
BaFuse v2 RAG Explanation Layer.

Takes model-derived degradation-mode estimates (LLI, LAM, CL) and retrieves
plausible battery degradation mechanisms from the static knowledge base.

IMPORTANT DISCLAIMERS:
  1. LLI/LAM/CL values are model-derived from ECM fitting — NOT physical
     ground truth. A high LLI estimate suggests lithium inventory loss is a
     plausible degradation pathway, but does NOT confirm it diagnostically.

  2. The RAG layer is a DOWNSTREAM EXPLANATION component only. It plays no
     role in model training or prediction. It cannot improve or correct
     the model's numerical outputs.

  3. Retrieved mechanisms are plausible candidates based on the literature.
     The output clearly distinguishes:
       (a) the estimated degradation-mode value
       (b) plausible mechanisms (from knowledge base)
       (c) supporting literature

  4. Do NOT claim a specific mechanism is definitively present based solely
     on the degradation-mode estimate.

Usage:
    explainer = DegradationExplainer()
    result = explainer.explain({"LLI": 32.5, "LAM": 15.0, "CL": 2.1})
    print(explainer.format_explanation(result))
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
                "IMPORTANT: LLI/LAM/CL values are model-derived estimates "
                "obtained from equivalent-circuit model (ECM) fitting of EIS "
                "data (Mendeley dataset). They are NOT absolute physical ground "
                "truth. The mechanisms listed below are plausible candidates "
                "supported by published literature; their presence cannot be "
                "confirmed solely from these estimates."
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
