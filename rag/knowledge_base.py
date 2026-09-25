"""
BaFuse v2 RAG Knowledge Base — Battery Degradation Mechanisms.

This is a STATIC, curated knowledge base.  It is NOT trained or updated
automatically.  It is used only as the retrieval corpus for the RAG
explanation layer that interprets model-derived degradation-mode estimates.

IMPORTANT:
  - LLI, LAM, CL are model-derived estimates from ECM fitting (Mendeley dataset).
  - A degradation mechanism is NOT definitively confirmed just because a
    degradation mode was predicted.
  - The RAG layer provides plausible mechanisms and supporting literature;
    it does NOT assert causality.

Structure of each entry:
    {
        "mode"       : "LLI" | "LAM" | "CL",
        "mechanism"  : short name,
        "description": plain-language description,
        "indicators" : list of observable indicators,
        "conditions" : list of operating conditions that accelerate it,
        "reference"  : APA-style citation,
        "source_url" : URL for the source (if available),
    }

Sources:
  - Birkl et al. (2017). Degradation diagnostics for lithium ion cells.
    Journal of Power Sources, 341, 373–386.
  - Dubarry et al. (2012). Identifying battery aging mechanisms in large format
    Li ion cells. Journal of Power Sources, 196(7), 3420-3425.
  - Vetter et al. (2005). Ageing mechanisms in lithium-ion batteries.
    Journal of Power Sources, 147(1-2), 269-281.
  - Wang et al. (2011). Cycle-life model for graphite-LiFePO4 cells.
    Journal of Power Sources, 196(8), 3942-3948.
  - Han et al. (2019). A comparative study of commercial lithium ion battery
    cycle life in electrical vehicle: Capacity loss estimation.
    Journal of Power Sources, 268, 658-669.
"""

from typing import Dict, List

# ── Knowledge base entries ────────────────────────────────────────────────────

DEGRADATION_KNOWLEDGE_BASE: List[Dict] = [

    # ─────────────────────────── LLI entries ─────────────────────────────────

    {
        "mode":        "LLI",
        "mechanism":   "SEI growth",
        "description": (
            "Continuous growth of the Solid Electrolyte Interphase (SEI) on "
            "the anode consumes lithium ions irreversibly, reducing the "
            "cycleable lithium inventory. SEI thickening also increases "
            "impedance and internal resistance."
        ),
        "indicators":  [
            "Increasing internal resistance (R_electrolyte)",
            "Capacity fade at constant coulombic efficiency",
            "Loss of low-voltage capacity in differential voltage analysis",
        ],
        "conditions": [
            "Elevated temperature (>35 °C)",
            "High state-of-charge storage",
            "Frequent shallow cycling",
        ],
        "reference": (
            "Vetter, J. et al. (2005). Ageing mechanisms in lithium-ion "
            "batteries. Journal of Power Sources, 147(1-2), 269-281."
        ),
        "source_url": "https://doi.org/10.1016/j.jpowsour.2005.01.006",
    },
    {
        "mode":        "LLI",
        "mechanism":   "Lithium plating",
        "description": (
            "At low temperatures or high charge rates, lithium ions may "
            "deposit as metallic lithium on the anode surface instead of "
            "intercalating. Plated lithium can become electrically isolated "
            "('dead lithium'), permanently removing it from the inventory."
        ),
        "indicators":  [
            "Voltage plateau around 0 V vs Li/Li⁺ during relaxation",
            "Accelerated capacity fade at low temperatures",
            "Increased internal resistance after fast charging",
        ],
        "conditions": [
            "Low temperature (<10 °C)",
            "High charge current (>1C)",
            "Full or near-full state-of-charge charging",
        ],
        "reference": (
            "Dubarry, M. et al. (2012). Identifying battery aging mechanisms "
            "in large format Li ion cells. Journal of Power Sources, 196(7), "
            "3420-3425."
        ),
        "source_url": "https://doi.org/10.1016/j.jpowsour.2011.10.111",
    },
    {
        "mode":        "LLI",
        "mechanism":   "Electrolyte decomposition",
        "description": (
            "Oxidative decomposition of the electrolyte at the cathode and "
            "reductive decomposition at the anode both consume lithium and "
            "generate resistive by-products. Gas evolution may also occur "
            "at high voltages."
        ),
        "indicators":  [
            "Increased R_ct (charge-transfer resistance) in EIS",
            "Gas evolution / swelling",
            "Capacity loss without significant voltage hysteresis increase",
        ],
        "conditions": [
            "High voltage (>4.2 V cutoff)",
            "Elevated temperature",
            "Long storage periods",
        ],
        "reference": (
            "Birkl, C. R. et al. (2017). Degradation diagnostics for lithium "
            "ion cells. Journal of Power Sources, 341, 373-386."
        ),
        "source_url": "https://doi.org/10.1016/j.jpowsour.2016.12.011",
    },

    # ─────────────────────────── LAM entries ─────────────────────────────────

    {
        "mode":        "LAM",
        "mechanism":   "Particle cracking (cathode)",
        "description": (
            "Repeated volume change of cathode particles during Li "
            "intercalation/de-intercalation induces mechanical stress, "
            "causing particle fracture. Cracks create new surfaces that "
            "react with electrolyte and electrically isolate fragments, "
            "reducing active material."
        ),
        "indicators":  [
            "Capacity fade at high C-rates",
            "Increasing charge-transfer resistance (R_ct) in EIS",
            "Voltage hysteresis increase",
        ],
        "conditions": [
            "Deep cycling (full depth-of-discharge)",
            "High charge/discharge rates",
            "Materials with large volume change (e.g. NCA, NMC at high V)",
        ],
        "reference": (
            "Vetter, J. et al. (2005). Ageing mechanisms in lithium-ion "
            "batteries. Journal of Power Sources, 147(1-2), 269-281."
        ),
        "source_url": "https://doi.org/10.1016/j.jpowsour.2005.01.006",
    },
    {
        "mode":        "LAM",
        "mechanism":   "Anode graphite exfoliation",
        "description": (
            "Solvent co-intercalation or mechanical stress can cause graphene "
            "layer separation in graphite anodes. Exfoliated particles lose "
            "electrical contact and are no longer electrochemically accessible."
        ),
        "indicators":  [
            "Capacity fade predominantly at low SOC",
            "Increase in low-frequency Warburg impedance (Zw) in EIS",
        ],
        "conditions": [
            "Incompatible electrolyte solvents",
            "Overdischarge (below 2.5 V)",
            "Mechanical vibration / stress",
        ],
        "reference": (
            "Dubarry, M. et al. (2012). Identifying battery aging mechanisms "
            "in large format Li ion cells. Journal of Power Sources, 196(7), "
            "3420-3425."
        ),
        "source_url": "https://doi.org/10.1016/j.jpowsour.2011.10.111",
    },
    {
        "mode":        "LAM",
        "mechanism":   "Transition metal dissolution (cathode)",
        "description": (
            "At elevated temperatures or high voltages, transition metals "
            "(e.g. Mn, Co, Ni) can dissolve from the cathode into the "
            "electrolyte, migrate to the anode, and deposit there. This "
            "reduces active cathode material and catalyses anode SEI growth."
        ),
        "indicators":  [
            "Capacity loss more pronounced at elevated temperature",
            "Increase in anode impedance",
            "Mn/Co/Ni detected in anode post-mortem analysis",
        ],
        "conditions": [
            "Temperature > 45 °C",
            "Voltage > 4.3 V",
            "Acidic electrolyte conditions (HF formation from LiPF₆ hydrolysis)",
        ],
        "reference": (
            "Han, X. et al. (2019). A comparative study of commercial lithium "
            "ion battery cycle life. Journal of Power Sources, 268, 658-669."
        ),
        "source_url": "https://doi.org/10.1016/j.jpowsour.2014.06.010",
    },

    # ─────────────────────────── CL entries ──────────────────────────────────

    {
        "mode":        "CL",
        "mechanism":   "Loss of electrical contact (binder degradation)",
        "description": (
            "The polymeric binder (e.g. PVDF) that holds active material "
            "particles to the current collector can soften, crack, or "
            "delaminate over cycling. Loss of binder integrity increases "
            "contact resistance and reduces electronic conductivity of the "
            "electrode."
        ),
        "indicators":  [
            "Increasing R (ohmic resistance) in EIS",
            "Capacity fade at high discharge rates (power fade)",
            "Non-uniform current distribution in electrode",
        ],
        "conditions": [
            "Elevated temperature",
            "Mechanical vibration",
            "Highly swelling active materials",
        ],
        "reference": (
            "Birkl, C. R. et al. (2017). Degradation diagnostics for lithium "
            "ion cells. Journal of Power Sources, 341, 373-386."
        ),
        "source_url": "https://doi.org/10.1016/j.jpowsour.2016.12.011",
    },
    {
        "mode":        "CL",
        "mechanism":   "Corrosion of current collector",
        "description": (
            "Aluminium current collectors can corrode in the presence of HF "
            "(from LiPF₆ hydrolysis) or at low potentials, increasing "
            "contact resistance between the current collector and the active "
            "material layer."
        ),
        "indicators":  [
            "Rise in high-frequency impedance (L, R in EIS)",
            "Capacity fade at low discharge rates",
            "Discolouration of current collector",
        ],
        "conditions": [
            "Moisture contamination (HF formation)",
            "Overdischarge (Al dissolution at <1 V)",
            "Acidic electrolyte",
        ],
        "reference": (
            "Vetter, J. et al. (2005). Ageing mechanisms in lithium-ion "
            "batteries. Journal of Power Sources, 147(1-2), 269-281."
        ),
        "source_url": "https://doi.org/10.1016/j.jpowsour.2005.01.006",
    },
    {
        "mode":        "CL",
        "mechanism":   "SEI impedance growth (conduction pathway blocking)",
        "description": (
            "Thick or resistive SEI layers block ionic conduction pathways "
            "through the porous electrode, increasing effective ionic "
            "resistance. This manifests as an overall rise in ohmic and "
            "charge-transfer resistance components in EIS."
        ),
        "indicators":  [
            "Increasing R_electrolyte in EIS over cycles",
            "Power fade more pronounced than energy fade",
            "Higher internal heat generation during fast charge",
        ],
        "conditions": [
            "High current operation",
            "Elevated temperature",
            "High cycle count",
        ],
        "reference": (
            "Wang, J. et al. (2011). Cycle-life model for graphite-LiFePO4 "
            "cells. Journal of Power Sources, 196(8), 3942-3948."
        ),
        "source_url": "https://doi.org/10.1016/j.jpowsour.2010.11.134",
    },
]

# ── Lookup helpers ─────────────────────────────────────────────────────────────

def get_entries_for_mode(mode: str) -> List[Dict]:
    """Return all knowledge base entries for a degradation mode (LLI/LAM/CL)."""
    return [e for e in DEGRADATION_KNOWLEDGE_BASE if e["mode"] == mode.upper()]


def get_all_modes() -> List[str]:
    return sorted({e["mode"] for e in DEGRADATION_KNOWLEDGE_BASE})
