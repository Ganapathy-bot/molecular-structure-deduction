"""Turn formula, IR, NMR, and mass evidence into a structure proposal."""

from __future__ import annotations

from deduction.formula import Formula


def _has_signal(nmr: dict, low: float, high: float, multiplicity: str | None = None, integral: float | None = None) -> bool:
    for signal in nmr["signals"]:
        if not (low <= signal["delta"] <= high):
            continue
        if multiplicity and signal["multiplicity"] not in (multiplicity, "unknown"):
            continue
        if integral is not None and signal["integral"] != integral:
            continue
        return True
    return False


def _mz(ms: dict, target: int, minimum: float = 0) -> bool:
    """Allow 1 m/z of calibration error from a photographed chart."""
    def close(mz: float) -> bool:
        return abs(mz - target) <= 1

    return any(close(fragment["mz"]) and fragment["intensity"] >= minimum for fragment in ms["fragments"]) or any(
        close(peak["mz"]) and peak["intensity"] >= minimum for peak in ms["peaks"]
    )


def _pieces(formula: Formula, ihd: float, ir: dict, nmr: dict, ms: dict) -> list[str]:
    pieces = []
    present = set(ir["present"])
    patterns = set(nmr["patterns"])
    if "aromatic" in present or "aromatic_protons" in patterns:
        if "monosubstituted_benzene" in present:
            pieces.append("monosubstituted benzene ring (C6H5)")
        else:
            pieces.append("benzene ring")
    if ir["carbonyl_kind"] == "carboxylic_acid" or "acid_proton" in patterns:
        pieces.append("carboxylic acid (COOH)")
    elif ir["carbonyl_kind"] == "aldehyde" or "aldehyde_proton" in patterns:
        pieces.append("aldehyde (CHO)")
    elif ir["carbonyl_kind"] == "ester":
        pieces.append("ester (COOR)")
    elif ir["carbonyl_kind"] in ("ketone", "conjugated_carbonyl"):
        pieces.append("ketone or conjugated carbonyl (C=O)")
    if "ethyl_ester" in patterns:
        pieces.append("ethyl ester (OCH2CH3)")
    if "ethyl_next_to_carbonyl" in patterns:
        pieces.append("ethyl group next to a carbonyl (CH3CH2CO)")
    if "ch3chx" in patterns:
        pieces.append("CH3CH(X) unit")
    if "nh2" in patterns or (ms.get("base_peak") == 30 and formula.N):
        pieces.append("primary amine with a CH2NH2 group")
    if "methyl_next_to_ch2" in patterns:
        pieces.append("methyl next to methylene (CH3CH2)")
    if ihd >= 4 and "benzene ring" not in " ".join(pieces) and "C6H5" not in " ".join(pieces):
        pieces.append("high unsaturation; a benzene ring is allowed by the IHD")
    return pieces


def _candidates() -> list[dict]:
    return [
        {
            "name": "acetone",
            "iupac": "propanone",
            "smiles": "CC(C)=O",
            "formula": "C3H6O",
            "structure": "CH3-C(=O)-CH3",
            "check": _acetone,
        },
        {
            "name": "propanal",
            "iupac": "propanal",
            "smiles": "CCC=O",
            "formula": "C3H6O",
            "structure": "CH3CH2CHO",
            "check": _propanal,
        },
        {
            "name": "toluene",
            "iupac": "methylbenzene",
            "smiles": "Cc1ccccc1",
            "formula": "C7H8",
            "structure": "C6H5CH3",
            "check": _toluene,
        },
        {
            "name": "benzaldehyde",
            "iupac": "benzaldehyde",
            "smiles": "O=Cc1ccccc1",
            "formula": "C7H6O",
            "structure": "C6H5CHO",
            "check": _benzaldehyde,
        },
        {
            "name": "propanoic acid",
            "iupac": "propanoic acid",
            "smiles": "CCC(=O)O",
            "formula": "C3H6O2",
            "structure": "CH3CH2COOH",
            "check": _propanoic_acid,
        },
        {
            "name": "methyl acetate",
            "iupac": "methyl acetate",
            "smiles": "CC(=O)OC",
            "formula": "C3H6O2",
            "structure": "CH3C(=O)OCH3",
            "check": _methyl_acetate,
        },
        {
            "name": "ethyl 2-chloropropanoate",
            "iupac": "ethyl 2-chloropropanoate",
            "smiles": "CCOC(=O)C(C)Cl",
            "formula": "C5H9O2Cl",
            "structure": "CH3CH(Cl)C(=O)OCH2CH3",
            "check": _ethyl_2_chloropropanoate,
        },
        {
            "name": "butanamine",
            "iupac": "butan-1-amine",
            "smiles": "CCCCN",
            "formula": "C4H11N",
            "structure": "CH3CH2CH2CH2NH2",
            "check": _butanamine,
        },
        {
            "name": "diethylamine",
            "iupac": "N-ethylethanamine",
            "smiles": "CCNCC",
            "formula": "C4H11N",
            "structure": "(CH3CH2)2NH",
            "check": _diethylamine,
        },
    ]


def _acetone(formula, ihd, ir, nmr, ms) -> tuple[int, list[str]] | None:
    if formula.raw != "C3H6O" or ihd != 1:
        return None
    score = 0
    reasons = []
    if ir["carbonyl_kind"] in ("ketone", "conjugated_carbonyl"):
        score += 3
        reasons.append("One carbonyl accounts for the IHD and the single oxygen.")
    if "aldehyde_ch" not in ir["present"] and "aldehyde_proton" not in nmr["patterns"]:
        score += 2
        reasons.append("The aldehyde C-H doublet and the aldehyde proton are absent, so the carbonyl is a ketone.")
    if nmr["signal_count"] <= 2 and _has_signal(nmr, 1.7, 3.1, "singlet"):
        score += 4
        reasons.append("One singlet near δ 2.2 means all six protons are equivalent methyl groups next to C=O.")
    if _mz(ms, 58) and _mz(ms, 43):
        score += 3
        reasons.append("M+ 58 and m/z 43 (M-15, CH3C≡O+) match acetone.")
    return score, reasons


def _propanal(formula, ihd, ir, nmr, ms) -> tuple[int, list[str]] | None:
    if formula.raw != "C3H6O":
        return None
    score = 0
    reasons = []
    if "aldehyde_proton" in nmr["patterns"] or "aldehyde_ch" in ir["present"]:
        score += 4
        reasons.append("Aldehyde hydrogen is present.")
    if nmr["signal_count"] >= 3:
        score += 1
    if score == 0:
        return 0, []
    return score, reasons


def _toluene(formula, ihd, ir, nmr, ms) -> tuple[int, list[str]] | None:
    if formula.raw != "C7H8" or ihd < 4:
        return None
    score = 0
    reasons = []
    if "aromatic" in ir["present"] or "aromatic_protons" in nmr["patterns"]:
        score += 3
        reasons.append("The benzene ring accounts for IHD 4.")
    if "monosubstituted_benzene" in ir["present"]:
        score += 2
        reasons.append("Bands near 690 and 750 cm-1 show a monosubstituted benzene.")
    if _has_signal(nmr, 6.8, 7.6) and _has_signal(nmr, 2.0, 2.6, "singlet"):
        score += 4
        reasons.append("Aromatic protons near δ 7.2 and a 3H methyl singlet near δ 2.3.")
    if _mz(ms, 92) and _mz(ms, 91, 40):
        score += 3
        reasons.append("M+ 92 and base peak 91 (loss of H·, C7H7+).")
    if "carbonyl" in ir["present"]:
        score -= 5
    return score, reasons


def _benzaldehyde(formula, ihd, ir, nmr, ms) -> tuple[int, list[str]] | None:
    if formula.raw != "C7H6O" or ihd < 4:
        return None
    score = 0
    reasons = []
    if ir["carbonyl_kind"] in ("aldehyde", "conjugated_carbonyl", "ketone"):
        score += 2
        reasons.append("A conjugated carbonyl accounts for the fifth degree of unsaturation.")
    if "aldehyde_ch" in ir["present"] or "aldehyde_proton" in nmr["patterns"]:
        score += 4
        reasons.append("The aldehyde C-H and the singlet near δ 10 identify CHO.")
    if "monosubstituted_benzene" in ir["present"] or "aromatic_protons" in nmr["patterns"]:
        score += 2
        reasons.append("The ring is a monosubstituted benzene.")
    if _mz(ms, 106) and _mz(ms, 105, 40):
        score += 3
        reasons.append("M+ 106 and m/z 105 (M-1) from loss of the aldehyde hydrogen.")
    if _mz(ms, 77):
        score += 1
        reasons.append("m/z 77 is C6H5+ after loss of CO.")
    return score, reasons


def _propanoic_acid(formula, ihd, ir, nmr, ms) -> tuple[int, list[str]] | None:
    if formula.raw != "C3H6O2" or ihd != 1:
        return None
    score = 0
    reasons = []
    if ir["carbonyl_kind"] == "carboxylic_acid" or "acid_proton" in nmr["patterns"]:
        score += 4
        reasons.append("Broad O-H plus a carbonyl, and a proton near δ 12, identify COOH.")
    ethyl = "ethyl_next_to_carbonyl" in nmr["patterns"] or (
        _has_signal(nmr, 2.0, 2.8, "quartet", 2) and _has_signal(nmr, 0.8, 1.5, "triplet", 3)
    )
    shifts = (
        any(2.0 <= signal["delta"] <= 2.9 for signal in nmr["signals"])
        and any(0.7 <= signal["delta"] <= 1.6 for signal in nmr["signals"])
    )
    if ethyl or shifts:
        score += 4 if ethyl else 2
        reasons.append("Signals near δ 2.4 and δ 1.2 are the ethyl group of CH3CH2COOH.")
    if _mz(ms, 74):
        score += 1
        reasons.append("M+ 74 matches C3H6O2.")
    if _mz(ms, 45) or _mz(ms, 29):
        score += 2
        reasons.append("m/z 45 (COOH+) and m/z 29 (loss of COOH) confirm the acid and the ethyl group.")
    if ir["carbonyl_kind"] == "ester" and "acid_proton" not in nmr["patterns"]:
        score -= 4
    return score, reasons


def _methyl_acetate(formula, ihd, ir, nmr, ms) -> tuple[int, list[str]] | None:
    if formula.raw != "C3H6O2":
        return None
    score = 0
    reasons = []
    if ir["carbonyl_kind"] == "ester" and "acid_proton" not in nmr["patterns"]:
        score += 3
        reasons.append("Ester carbonyl without an acid proton.")
    singlets = [signal for signal in nmr["signals"] if signal["multiplicity"] == "singlet"]
    if len(singlets) >= 2 and all(signal["delta"] < 4.5 for signal in singlets):
        score += 2
        reasons.append("Two methyl singlets fit CH3C(=O)OCH3.")
    if score == 0:
        return 0, []
    return score, reasons


def _ethyl_2_chloropropanoate(formula, ihd, ir, nmr, ms) -> tuple[int, list[str]] | None:
    if formula.raw != "C5H9O2Cl" or ihd != 1:
        return None
    score = 0
    reasons = []
    if ir["carbonyl_kind"] == "ester" or ("carbonyl" in ir["present"] and "c_o_stretch" in ir["present"]):
        score += 3
        reasons.append("Ester C=O near 1740 cm-1 and a C-O stretch.")
    if "ethyl_ester" in nmr["patterns"] or (
        _has_signal(nmr, 3.9, 4.5, "quartet", 2) and _has_signal(nmr, 1.0, 1.5, "triplet", 3)
    ):
        score += 3
        reasons.append("The 2H quartet and 3H triplet are the ethyl ester.")
    if "ch3chx" in nmr["patterns"] or (
        _has_signal(nmr, 4.1, 4.6, "quartet", 1) and _has_signal(nmr, 1.5, 1.9, "doublet", 3)
    ):
        score += 3
        reasons.append("The 1H quartet and 3H doublet are CH3CHCl.")
    if _mz(ms, 29, 50):
        score += 2
        reasons.append("Base peak m/z 29 is the ethyl cation.")
    if _mz(ms, 63):
        score += 2
        reasons.append("m/z 63 is CH3CHCl+.")
    if any(3.8 <= signal["delta"] <= 4.8 for signal in nmr["signals"]) and any(
        1.0 <= signal["delta"] <= 1.9 for signal in nmr["signals"]
    ):
        score += 2
        reasons.append("Downfield signals near δ 4.3 and alkyl signals near δ 1.3 fit the ethyl ester and CH3CHCl.")
    if "acid_proton" in nmr["patterns"]:
        score -= 5
    return score, reasons


def _butanamine(formula, ihd, ir, nmr, ms) -> tuple[int, list[str]] | None:
    if formula.raw != "C4H11N" or ihd != 0:
        return None
    score = 0
    reasons = []
    if ms.get("base_peak") == 30 or _mz(ms, 30, 50):
        score += 4
        reasons.append("Base peak m/z 30 is CH2=NH2+ from a primary CH2NH2 group.")
    if any(2.3 <= signal["delta"] <= 3.1 for signal in nmr["signals"]) and any(
        0.6 <= signal["delta"] <= 1.3 for signal in nmr["signals"]
    ):
        score += 2
        reasons.append("A signal near δ 2.7 and a methyl signal near δ 0.9 fit a straight-chain primary amine.")
    if _has_signal(nmr, 2.3, 3.1, "triplet", 2):
        score += 2
        reasons.append("The 2H triplet near δ 2.7 is CH2NH2.")
    if _has_signal(nmr, 0.6, 1.2, "triplet", 3):
        score += 2
        reasons.append("The 3H triplet near δ 0.9 is the terminal methyl of a straight chain.")
    if any(signal["multiplicity"] == "multiplet" and (signal["integral"] in (None, 4)) and 1.0 <= signal["delta"] <= 1.8 for signal in nmr["signals"]):
        score += 2
        reasons.append("The 4H multiplet is the two middle methylene groups.")
    if "nh2" in nmr["patterns"] or _has_signal(nmr, 0.8, 1.6, "singlet", 2):
        score += 1
        reasons.append("The broad 2H singlet is NH2.")
    if nmr["signal_count"] <= 2:
        score -= 3
    return score, reasons


def _diethylamine(formula, ihd, ir, nmr, ms) -> tuple[int, list[str]] | None:
    if formula.raw != "C4H11N":
        return None
    score = 0
    reasons = []
    quartet = _has_signal(nmr, 2.3, 3.0, "quartet", 4)
    triplet = _has_signal(nmr, 0.8, 1.4, "triplet", 6)
    if quartet and triplet and ms.get("base_peak") != 30:
        score += 4
        reasons.append("A 4H quartet and a 6H triplet fit two equivalent ethyl groups on nitrogen.")
    if score == 0:
        return 0, []
    return score, reasons


def propose_structure(formula: Formula, ihd: float, ir: dict, nmr: dict, ms: dict) -> dict:
    pieces = _pieces(formula, ihd, ir, nmr, ms)
    ranked = []
    for candidate in _candidates():
        if candidate["formula"] != formula.raw:
            continue
        result = candidate["check"](formula, ihd, ir, nmr, ms)
        if result is None:
            continue
        score, reasons = result
        if score <= 0:
            continue
        ranked.append(
            {
                "name": candidate["name"],
                "iupac": candidate["iupac"],
                "smiles": candidate["smiles"],
                "structure": candidate["structure"],
                "score": score,
                "reasons": reasons,
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    best = ranked[0] if ranked else None
    confident = best is not None and best["score"] >= 6 and (len(ranked) == 1 or best["score"] - ranked[1]["score"] >= 2)
    return {
        "pieces": pieces,
        "candidates": ranked,
        "best": best if confident else None,
        "confident": confident,
    }
