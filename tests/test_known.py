"""The six worked examples must come back from their peak lists."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deduction.engine import deduce_from_peaks
from deduction.ir import IrPeak
from deduction.ms import MsPeak
from deduction.nmr import NmrSignal


CASES = [
    (
        "acetone",
        "C3H6O",
        [IrPeak(1715, "strong", "sharp"), IrPeak(2960, "medium", "sharp"), IrPeak(3400, "weak", "sharp")],
        [NmrSignal(2.2, "singlet", 6)],
        [MsPeak(58, 60), MsPeak(43, 100), MsPeak(15, 18)],
    ),
    (
        "toluene",
        "C7H8",
        [
            IrPeak(3050, "medium", "sharp"),
            IrPeak(2920, "medium", "sharp"),
            IrPeak(1605, "medium", "sharp"),
            IrPeak(1500, "medium", "sharp"),
            IrPeak(750, "strong", "sharp"),
            IrPeak(690, "strong", "sharp"),
        ],
        [NmrSignal(7.2, "singlet", 5), NmrSignal(2.3, "singlet", 3)],
        [MsPeak(92, 75), MsPeak(91, 100), MsPeak(65, 20), MsPeak(51, 12)],
    ),
    (
        "benzaldehyde",
        "C7H6O",
        [
            IrPeak(3070, "medium", "sharp"),
            IrPeak(2820, "medium", "sharp"),
            IrPeak(2730, "medium", "sharp"),
            IrPeak(1700, "strong", "sharp"),
            IrPeak(1600, "medium", "sharp"),
            IrPeak(750, "strong", "sharp"),
            IrPeak(690, "strong", "sharp"),
        ],
        [NmrSignal(9.9, "singlet", 1), NmrSignal(7.6, "multiplet", 5)],
        [MsPeak(106, 95), MsPeak(105, 100), MsPeak(77, 85), MsPeak(51, 35)],
    ),
    (
        "propanoic acid",
        "C3H6O2",
        [IrPeak(3000, "strong", "broad"), IrPeak(1710, "strong", "sharp")],
        [NmrSignal(11.7, "singlet", 1), NmrSignal(2.4, "quartet", 2), NmrSignal(1.2, "triplet", 3)],
        [MsPeak(74, 70), MsPeak(73, 55), MsPeak(57, 30), MsPeak(45, 45), MsPeak(29, 90), MsPeak(28, 100)],
    ),
    (
        "ethyl 2-chloropropanoate",
        "C5H9O2Cl",
        [IrPeak(2980, "medium", "sharp"), IrPeak(1740, "strong", "sharp"), IrPeak(1200, "strong", "sharp"), IrPeak(1100, "strong", "sharp")],
        [
            NmrSignal(4.39, "quartet", 1),
            NmrSignal(4.23, "quartet", 2),
            NmrSignal(1.68, "doublet", 3),
            NmrSignal(1.31, "triplet", 3),
        ],
        [MsPeak(136, 4), MsPeak(91, 8), MsPeak(63, 50), MsPeak(29, 100), MsPeak(28, 30)],
    ),
    (
        "butanamine",
        "C4H11N",
        [IrPeak(3370, "weak", "sharp"), IrPeak(3290, "weak", "sharp"), IrPeak(2930, "strong", "sharp"), IrPeak(800, "strong", "broad")],
        [
            NmrSignal(2.7, "triplet", 2),
            NmrSignal(1.4, "multiplet", 4),
            NmrSignal(1.16, "singlet", 2),
            NmrSignal(0.92, "triplet", 3),
        ],
        [MsPeak(73, 15), MsPeak(30, 100)],
    ),
]


def main() -> None:
    failed = 0
    for name, formula, ir, nmr, ms in CASES:
        result = deduce_from_peaks(formula, ir, nmr, ms, model=None)
        got = None if result["structure"] is None else result["structure"]["name"]
        ihd = result["ihd"]
        print(f"{formula:12} IHD {ihd:g} -> {got}")
        if got != name:
            failed += 1
            print("  candidates", [(item["name"], item["score"]) for item in result["candidates"]])
            print("  pieces", result["pieces"])
            print("  ir", result["ir"]["present"], result["ir"]["carbonyl_kind"])
            print("  nmr", result["nmr"]["patterns"])
    if failed:
        raise SystemExit(f"{failed} structures did not match")
    print("all six structures matched")


if __name__ == "__main__":
    main()
