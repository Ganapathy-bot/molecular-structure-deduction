"""Fixed-length features shared by the rule engine and the saved network."""

from __future__ import annotations

import numpy as np

from deduction.formula import Formula

LABELS = [
    "carbonyl",
    "aldehyde",
    "ketone",
    "carboxylic_acid",
    "ester",
    "primary_amine",
    "aromatic",
    "monosubstituted_benzene",
    "ethyl_group",
    "methyl_ketone",
]

IR_FLAGS = [
    "alcohol_oh",
    "carboxylic_acid",
    "nh",
    "sp2_ch",
    "sp3_ch",
    "aldehyde_ch",
    "triple_bond",
    "carbonyl",
    "aromatic",
    "monosubstituted_benzene",
    "c_o_stretch",
]


def featurize(formula: Formula, ihd: float, ir: dict, nmr: dict, ms: dict) -> np.ndarray:
    values = [
        formula.C / 10.0,
        formula.H / 24.0,
        float(formula.N),
        formula.O / 4.0,
        float(formula.X),
        ihd / 8.0,
    ]
    present = set(ir.get("present", []))
    values.extend(1.0 if name in present else 0.0 for name in IR_FLAGS)
    signals = nmr.get("signals", [])
    patterns = set(nmr.get("patterns", []))
    values.extend(
        [
            min(len(signals), 8) / 8.0,
            1.0 if any(signal["delta"] >= 9.5 for signal in signals) else 0.0,
            1.0 if any(signal["delta"] >= 11 for signal in signals) else 0.0,
            1.0 if any(6.5 <= signal["delta"] <= 8.5 for signal in signals) else 0.0,
            1.0 if any(3.4 <= signal["delta"] <= 4.8 for signal in signals) else 0.0,
            1.0 if any(2.0 <= signal["delta"] <= 3.2 for signal in signals) else 0.0,
            1.0 if any(0.6 <= signal["delta"] <= 1.8 for signal in signals) else 0.0,
            1.0 if "all_protons_equivalent" in patterns else 0.0,
        ]
    )
    fragment_mz = {fragment["mz"] for fragment in ms.get("fragments", [])}
    peak_mz = {peak["mz"] for peak in ms.get("peaks", [])}
    seen = fragment_mz | peak_mz
    base = ms.get("base_peak")
    values.extend(
        [
            1.0 if ms.get("molecular_ion") else 0.0,
            1.0 if base == 30 else 0.0,
            1.0 if 15 in seen else 0.0,
            1.0 if 29 in seen else 0.0,
            1.0 if 43 in seen else 0.0,
            1.0 if 45 in seen else 0.0,
            1.0 if 77 in seen else 0.0,
            1.0 if 91 in seen else 0.0,
            1.0 if 105 in seen else 0.0,
            1.0 if base is not None and ms.get("molecular_ion") and base == ms["molecular_ion"] - 15 else 0.0,
        ]
    )
    return np.asarray(values, dtype=np.float32)


def feature_names() -> list[str]:
    names = ["C", "H", "N", "O", "halogen", "IHD"]
    names.extend(f"ir_{name}" for name in IR_FLAGS)
    names.extend(
        [
            "nmr_signal_count",
            "nmr_above_9_5",
            "nmr_above_11",
            "nmr_aromatic",
            "nmr_deshielded_alkyl",
            "nmr_alpha",
            "nmr_alkyl",
            "nmr_single_signal",
        ]
    )
    names.extend(
        ["ms_molecular_ion", "ms_base_30", "ms_15", "ms_29", "ms_43", "ms_45", "ms_77", "ms_91", "ms_105", "ms_base_m_minus_15"]
    )
    return names
