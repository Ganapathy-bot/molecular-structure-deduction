"""Assign infrared peaks with the functional-group correlation table."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IrPeak:
    wavenumber: float
    intensity: str = "medium"  # weak, medium, strong
    shape: str = "sharp"  # sharp, broad


def _near(peaks: list[IrPeak], low: float, high: float, intensity: str | None = None) -> list[IrPeak]:
    found = [peak for peak in peaks if low <= peak.wavenumber <= high]
    if intensity == "strong":
        found = [peak for peak in found if peak.intensity == "strong"]
    return found


def interpret_ir(peaks: list[IrPeak]) -> dict:
    """Return functional-group calls and the peaks that support them."""
    groups: list[dict] = []

    def add(name: str, present: bool, detail: str, evidence: list[IrPeak]) -> None:
        groups.append(
            {
                "group": name,
                "present": present,
                "detail": detail,
                "peaks": [round(peak.wavenumber) for peak in evidence],
            }
        )

    oh = [peak for peak in _near(peaks, 3200, 3600) if peak.shape == "broad"]
    acid_oh = [
        peak
        for peak in peaks
        if 2500 <= peak.wavenumber <= 3300 and peak.shape == "broad" and peak.intensity in ("strong", "medium")
    ]
    nh = [peak for peak in _near(peaks, 3200, 3500) if peak.shape == "sharp"]
    sp2 = _near(peaks, 3000, 3100)
    sp3 = _near(peaks, 2850, 2995)
    aldehyde = _near(peaks, 2700, 2860)
    triple = _near(peaks, 2100, 2300)
    carbonyl = _near(peaks, 1650, 1880)
    strong_carbonyl = [peak for peak in carbonyl if peak.intensity == "strong"] or carbonyl
    alkene = _near(peaks, 1600, 1680)
    aromatic = _near(peaks, 1450, 1610)
    monosub = _near(peaks, 680, 710) and _near(peaks, 730, 770)
    co_stretch = [peak for peak in _near(peaks, 1000, 1300) if peak.intensity == "strong"]

    carbonyl_peak = None
    if strong_carbonyl:
        carbonyl_peak = max(strong_carbonyl, key=lambda peak: (peak.intensity == "strong", -abs(peak.wavenumber - 1715)))

    add(
        "alcohol_oh",
        bool(oh) and not acid_oh,
        "Broad O-H stretch, 3200-3600 cm-1.",
        oh,
    )
    add(
        "carboxylic_acid",
        bool(acid_oh) and bool(strong_carbonyl),
        "Very broad O-H from about 3300 to 2500 cm-1 together with a carbonyl.",
        acid_oh + strong_carbonyl,
    )
    if len(nh) >= 2:
        nh_detail = "Two sharp N-H spikes: primary amine or amide."
    elif len(nh) == 1:
        nh_detail = "One sharp N-H spike: secondary amine, or an unresolved primary pair."
    else:
        nh_detail = "No sharp N-H stretch."
    add("nh", bool(nh), nh_detail, nh)
    add("sp2_ch", bool(sp2), "C-H stretch above 3000 cm-1: alkene or aromatic hydrogen.", sp2)
    add("sp3_ch", bool(sp3), "C-H stretch below 3000 cm-1: saturated carbon.", sp3)
    add(
        "aldehyde_ch",
        len(aldehyde) >= 1 and bool(strong_carbonyl),
        "Aldehyde C-H doublet near 2720 and 2820 cm-1.",
        aldehyde,
    )
    add("triple_bond", bool(triple), "Sharp band at 2200-2300 cm-1: C≡C or C≡N.", triple)

    carbonyl_kind = "none"
    if strong_carbonyl and carbonyl_peak is not None:
        position = carbonyl_peak.wavenumber
        if acid_oh:
            carbonyl_kind = "carboxylic_acid"
        elif aldehyde:
            carbonyl_kind = "aldehyde"
        elif position >= 1735 and co_stretch:
            carbonyl_kind = "ester"
        elif position >= 1730 and co_stretch and not acid_oh:
            carbonyl_kind = "ester"
        elif position <= 1710 and (sp2 or aromatic):
            carbonyl_kind = "conjugated_carbonyl"
        else:
            carbonyl_kind = "ketone"
        add("carbonyl", True, f"Carbonyl C=O near {round(position)} cm-1 ({carbonyl_kind.replace('_', ' ')}).", [carbonyl_peak])
    else:
        add("carbonyl", False, "No strong carbonyl between 1650 and 1765 cm-1.", [])

    add("alkene", bool(alkene) and not aromatic, "C=C stretch, 1600-1680 cm-1.", alkene)
    add(
        "aromatic",
        bool(aromatic) and (bool(sp2) or bool(monosub) or len(aromatic) >= 2),
        "Aromatic ring bands between 1450 and 1610 cm-1.",
        aromatic,
    )
    add(
        "monosubstituted_benzene",
        bool(monosub),
        "Out-of-plane C-H bends near 690 and 750 cm-1: monosubstituted benzene.",
        _near(peaks, 680, 770),
    )
    add(
        "c_o_stretch",
        bool(co_stretch),
        "Strong C-O stretch between 1000 and 1300 cm-1.",
        co_stretch,
    )

    present = [group["group"] for group in groups if group["present"]]
    return {
        "peaks": [{"wavenumber": round(peak.wavenumber), "intensity": peak.intensity, "shape": peak.shape} for peak in peaks],
        "groups": groups,
        "present": present,
        "carbonyl_kind": carbonyl_kind,
        "nh_band_count": len(nh),
    }
