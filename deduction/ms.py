"""Assign the molecular ion and the main fragments in an EI mass spectrum."""

from __future__ import annotations

from dataclasses import dataclass

from deduction.formula import Formula


@dataclass
class MsPeak:
    mz: float
    intensity: float  # percent of the base peak


def _closest(peaks: list[MsPeak], target: float, tolerance: float = 1.0) -> MsPeak | None:
    found = [peak for peak in peaks if abs(peak.mz - target) <= tolerance]
    if not found:
        return None
    return max(found, key=lambda peak: peak.intensity)


def interpret_ms(peaks: list[MsPeak], formula: Formula) -> dict:
    if not peaks:
        return {"peaks": [], "molecular_ion": None, "base_peak": None, "fragments": [], "notes": []}

    base = max(peaks, key=lambda peak: peak.intensity)
    nominal = formula.mass
    molecular = _closest(peaks, nominal, 0.6)
    isotope = _closest(peaks, nominal + 2, 0.6) if formula.counts.get("Cl") or formula.counts.get("Br") else None

    fragments = []

    def add(label: str, mz: float, detail: str) -> None:
        peak = _closest(peaks, mz, 0.8)
        if peak is None:
            return
        fragments.append(
            {
                "mz": round(peak.mz),
                "intensity": round(peak.intensity),
                "label": label,
                "detail": detail,
            }
        )

    if molecular is not None:
        mass = molecular.mz
        add("M-1", mass - 1, "Loss of H·.")
        add("M-15", mass - 15, "Loss of ·CH3.")
        add("M-17", mass - 17, "Loss of ·OH.")
        add("M-29", mass - 29, "Loss of ·CH2CH3 or ·CHO.")
        add("M-43", mass - 43, "Loss of ·COCH3 or ·C3H7.")
        add("M-45", mass - 45, "Loss of ·COOH or ·OCH2CH3.")
    add("m/z 15", 15, "CH3+.")
    add("m/z 29", 29, "CH3CH2+.")
    add("m/z 30", 30, "CH2=NH2+, the alpha-cleavage ion of a primary amine RCH2NH2.")
    add("m/z 43", 43, "CH3C≡O+, typical of a methyl ketone.")
    add("m/z 45", 45, "COOH+.")
    add("m/z 63", 63, "C2H4Cl+, for example CH3CHCl+.")
    add("m/z 77", 77, "C6H5+, the phenyl cation, often after loss of CO from an aroyl ion.")
    add("m/z 91", 91, "C7H7+ from an alkylbenzene, or an acylium ion of mass 91.")
    add("m/z 105", 105, "C6H5C≡O+, from loss of H· from a benzoyl compound.")

    notes = []
    if molecular is not None:
        notes.append(f"M+ at m/z {round(molecular.mz)} matches {formula.raw} (mass {nominal}).")
    else:
        notes.append(f"No clear M+ at m/z {nominal}. The formula mass is still {nominal}.")
    if isotope is not None:
        notes.append(f"M+2 at m/z {round(isotope.mz)} fits one Cl or Br atom.")
    if base is not None:
        notes.append(f"Base peak at m/z {round(base.mz)} ({round(base.intensity)}%).")
        if abs(base.mz - 30) <= 0.8:
            notes.append("A base peak at m/z 30 is the usual marker of a CH2NH2 group.")
        if molecular is not None and abs(base.mz - (molecular.mz - 15)) <= 0.8:
            notes.append("The base peak is M-15, loss of a methyl radical.")
        if molecular is not None and abs(base.mz - (molecular.mz - 1)) <= 0.8:
            notes.append("The base peak is M-1, loss of a hydrogen atom.")

    # Keep each m/z once, preferring the more specific label.
    unique = []
    seen = set()
    for fragment in fragments:
        if fragment["mz"] in seen:
            continue
        seen.add(fragment["mz"])
        unique.append(fragment)

    return {
        "peaks": [{"mz": round(peak.mz), "intensity": round(peak.intensity)} for peak in sorted(peaks, key=lambda peak: peak.mz)],
        "molecular_ion": None if molecular is None else round(molecular.mz),
        "base_peak": None if base is None else round(base.mz),
        "fragments": unique,
        "notes": notes,
    }
