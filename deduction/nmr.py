"""Assign 1H-NMR signals from chemical shift, splitting, and integration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NmrSignal:
    delta: float
    multiplicity: str = "unknown"  # singlet, doublet, triplet, quartet, multiplet
    integral: float | None = None


def _region(delta: float) -> str:
    if delta >= 11:
        return "carboxylic_acid"
    if delta >= 9:
        return "aldehyde"
    if 6.5 <= delta < 9:
        return "aromatic"
    if 4.5 <= delta < 6.5:
        return "vinylic"
    if 3.3 <= delta < 4.8:
        return "deshielded_alkyl"
    if 2.0 <= delta < 3.3:
        return "alpha_or_benzylic"
    if 0.5 <= delta < 2.0:
        return "alkyl"
    return "other"


def _assignment(signal: NmrSignal) -> str:
    region = _region(signal.delta)
    protons = f"{signal.integral:g}H, " if signal.integral else ""
    splitting = signal.multiplicity
    if region == "carboxylic_acid":
        return f"{protons}{splitting}: carboxylic acid proton."
    if region == "aldehyde":
        return f"{protons}{splitting}: aldehyde proton."
    if region == "aromatic":
        return f"{protons}{splitting}: aromatic protons."
    if region == "vinylic":
        return f"{protons}{splitting}: alkene proton."
    if region == "deshielded_alkyl":
        if splitting == "quartet" and signal.integral == 2:
            return "2H quartet: OCH2 of an ethyl ester, or a similar group on oxygen."
        if splitting == "quartet" and signal.integral == 1:
            return "1H quartet: CH attached to Cl or another electronegative group, next to a methyl."
        return f"{protons}{splitting}: hydrogen on a carbon next to O, Cl, or another electronegative atom."
    if region == "alpha_or_benzylic":
        if splitting == "triplet" and signal.integral == 2:
            return "2H triplet: CH2 next to NH2 or next to a carbonyl, with one neighboring CH2."
        if splitting == "quartet" and signal.integral == 2:
            return "2H quartet: CH2 of an ethyl group next to a carbonyl."
        if splitting == "singlet" and signal.integral == 3:
            return "3H singlet: methyl on a benzene ring or on a carbon with no neighboring hydrogen."
        if splitting == "singlet" and signal.integral == 6:
            return "6H singlet: two equivalent methyl groups, as in a methyl ketone with identical methyls."
        return f"{protons}{splitting}: hydrogen next to C=O, nitrogen, or a benzene ring."
    if region == "alkyl":
        if splitting == "triplet" and signal.integral == 3:
            return "3H triplet: methyl next to CH2."
        if splitting == "doublet" and signal.integral == 3:
            return "3H doublet: methyl next to CH."
        if splitting == "singlet" and signal.integral == 2:
            return "2H singlet: NH2, which has no neighboring carbon hydrogen."
        if splitting == "multiplet":
            return f"{protons}multiplet: methylene chain."
        return f"{protons}{splitting}: alkyl hydrogen."
    return f"{protons}{splitting}: unassigned."


def _patterns(signals: list[NmrSignal]) -> list[str]:
    patterns = []
    quarts = [signal for signal in signals if signal.multiplicity == "quartet"]
    trips = [signal for signal in signals if signal.multiplicity == "triplet" and signal.integral == 3]
    if any(signal.integral == 2 and signal.delta >= 3.3 for signal in quarts) and trips:
        patterns.append("ethyl_ester")
    if any(signal.integral == 2 and 2.0 <= signal.delta < 3.2 for signal in quarts) and any(signal.delta < 1.6 for signal in trips):
        patterns.append("ethyl_next_to_carbonyl")
    if any(signal.multiplicity == "triplet" and signal.integral == 3 and signal.delta < 1.5 for signal in signals):
        patterns.append("methyl_next_to_ch2")
    if any(signal.multiplicity == "doublet" and signal.integral == 3 for signal in signals) and any(
        signal.multiplicity == "quartet" and signal.integral == 1 for signal in signals
    ):
        patterns.append("ch3chx")
    if any(signal.delta >= 11 for signal in signals):
        patterns.append("acid_proton")
    if any(9 <= signal.delta < 11 for signal in signals):
        patterns.append("aldehyde_proton")
    if any(6.5 <= signal.delta < 8.5 for signal in signals):
        patterns.append("aromatic_protons")
    if any(signal.multiplicity == "singlet" and signal.integral == 2 and signal.delta < 2 for signal in signals):
        patterns.append("nh2")
    if len(signals) == 1:
        patterns.append("all_protons_equivalent")
    return patterns


def interpret_nmr(signals: list[NmrSignal]) -> dict:
    ordered = sorted(signals, key=lambda signal: signal.delta, reverse=True)
    rows = []
    for signal in ordered:
        rows.append(
            {
                "delta": round(signal.delta, 2),
                "multiplicity": signal.multiplicity,
                "integral": signal.integral,
                "region": _region(signal.delta),
                "assignment": _assignment(signal),
            }
        )
    return {
        "signals": rows,
        "signal_count": len(rows),
        "patterns": _patterns(ordered),
        "all_equivalent": len(rows) == 1,
    }
