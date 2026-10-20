"""Molecular formula parsing, exact mass, and the index of hydrogen deficiency."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
_MASSES = {
    "C": 12,
    "H": 1,
    "N": 14,
    "O": 16,
    "F": 19,
    "Cl": 35,
    "Br": 79,
    "I": 127,
    "S": 32,
    "P": 31,
}
_HALOGENS = ("F", "Cl", "Br", "I")


@dataclass
class Formula:
    raw: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def C(self) -> int:
        return self.counts.get("C", 0)

    @property
    def H(self) -> int:
        return self.counts.get("H", 0)

    @property
    def N(self) -> int:
        return self.counts.get("N", 0)

    @property
    def O(self) -> int:
        return self.counts.get("O", 0)

    @property
    def X(self) -> int:
        return sum(self.counts.get(element, 0) for element in _HALOGENS)

    @property
    def mass(self) -> int:
        total = 0
        for element, count in self.counts.items():
            if element not in _MASSES:
                raise ValueError(f"No integer mass is defined for {element}")
            total += _MASSES[element] * count
        return total

    def isotope_masses(self) -> list[int]:
        """Nominal M and M+2 when Cl or Br is present. Cl is 3:1, Br is 1:1."""
        masses = [self.mass]
        if self.counts.get("Cl", 0) or self.counts.get("Br", 0):
            masses.append(self.mass + 2)
        return masses


def parse_formula(text: str) -> Formula:
    cleaned = text.replace(" ", "").replace("·", "")
    if not cleaned:
        raise ValueError("Molecular formula is empty")
    counts: dict[str, int] = {}
    consumed = 0
    for match in _TOKEN.finditer(cleaned):
        element, number = match.group(1), match.group(2)
        if element not in _MASSES:
            raise ValueError(f"Unsupported element {element} in {text}")
        counts[element] = counts.get(element, 0) + (int(number) if number else 1)
        consumed += len(match.group(0))
    if consumed != len(cleaned):
        raise ValueError(f"Could not parse molecular formula {text}")
    return Formula(raw=_canonical(counts), counts=counts)


def _canonical(counts: dict[str, int]) -> str:
    order = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
    parts = []
    for element in order:
        count = counts.get(element, 0)
        if count:
            parts.append(element + (str(count) if count > 1 else ""))
    for element, count in counts.items():
        if element not in order and count:
            parts.append(element + (str(count) if count > 1 else ""))
    return "".join(parts)


def hydrogen_deficiency(formula: Formula) -> float:
    """IHD = (2C + 2 + N - H - X) / 2. Oxygen and sulfur are omitted."""
    value = (2 * formula.C + 2 + formula.N + formula.counts.get("P", 0) - formula.H - formula.X) / 2
    if value < 0 or abs(value * 2 - round(value * 2)) > 1e-6:
        raise ValueError(f"IHD for {formula.raw} is {value}, which is not a valid deficiency")
    return float(value)


def ihd_meaning(value: float) -> str:
    if value == 0:
        return "Fully saturated: only single bonds and no rings."
    if value == 1:
        return "One double bond or one ring."
    if value == 2:
        return "One triple bond, or two double bonds, or two rings, or one double bond and one ring."
    if value == 3:
        return "Three degrees of unsaturation, in any mix of rings, double bonds, and triple bonds."
    if value >= 4:
        return (
            "Four or more degrees of unsaturation. A benzene ring accounts for four "
            "(one ring and three double bonds) and is a strong candidate."
        )
    return "The deficiency does not match a stable closed-shell organic formula."
