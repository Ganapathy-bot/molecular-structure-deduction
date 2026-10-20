"""Train the functional-group network on spectra generated from the correlation rules."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from deduction.engine import deduce_from_peaks
from deduction.features import LABELS, featurize
from deduction.formula import hydrogen_deficiency, parse_formula
from deduction.ir import IrPeak
from deduction.ms import MsPeak
from deduction.net import FunctionalGroupNet
from deduction.nmr import NmrSignal

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "saved_model" / "functional_groups.npz"


def _jitter(rng: np.random.Generator, center: float, spread: float) -> float:
    return float(center + rng.normal(0, spread))


def _sample(rng: np.random.Generator, kind: str) -> tuple[str, list[IrPeak], list[NmrSignal], list[MsPeak], dict[str, int]]:
    """Return formula, peaks, and the true functional-group labels."""
    labels = {name: 0 for name in LABELS}
    if kind == "acetone":
        formula = "C3H6O"
        ir = [IrPeak(_jitter(rng, 1715, 8), "strong", "sharp"), IrPeak(_jitter(rng, 2920, 15), "medium", "sharp")]
        nmr = [NmrSignal(_jitter(rng, 2.15, 0.05), "singlet", 6)]
        ms = [MsPeak(58, 55), MsPeak(43, 100), MsPeak(15, 20)]
        labels.update(carbonyl=1, ketone=1, methyl_ketone=1)
    elif kind == "propanal":
        formula = "C3H6O"
        ir = [
            IrPeak(_jitter(rng, 1730, 8), "strong", "sharp"),
            IrPeak(_jitter(rng, 2720, 8), "medium", "sharp"),
            IrPeak(_jitter(rng, 2820, 8), "medium", "sharp"),
        ]
        nmr = [NmrSignal(_jitter(rng, 9.7, 0.08), "triplet", 1), NmrSignal(2.4, "quartet", 2), NmrSignal(1.1, "triplet", 3)]
        ms = [MsPeak(58, 40), MsPeak(29, 100)]
        labels.update(carbonyl=1, aldehyde=1, ethyl_group=1)
    elif kind == "toluene":
        formula = "C7H8"
        ir = [
            IrPeak(3050, "medium", "sharp"),
            IrPeak(2925, "medium", "sharp"),
            IrPeak(1605, "medium", "sharp"),
            IrPeak(1495, "medium", "sharp"),
            IrPeak(740, "strong", "sharp"),
            IrPeak(695, "strong", "sharp"),
        ]
        nmr = [NmrSignal(_jitter(rng, 7.2, 0.05), "singlet", 5), NmrSignal(_jitter(rng, 2.3, 0.04), "singlet", 3)]
        ms = [MsPeak(92, 70), MsPeak(91, 100), MsPeak(65, 15)]
        labels.update(aromatic=1, monosubstituted_benzene=1)
    elif kind == "benzaldehyde":
        formula = "C7H6O"
        ir = [
            IrPeak(3060, "medium", "sharp"),
            IrPeak(2820, "medium", "sharp"),
            IrPeak(2730, "medium", "sharp"),
            IrPeak(_jitter(rng, 1702, 8), "strong", "sharp"),
            IrPeak(1600, "medium", "sharp"),
            IrPeak(750, "strong", "sharp"),
            IrPeak(690, "strong", "sharp"),
        ]
        nmr = [NmrSignal(_jitter(rng, 9.95, 0.08), "singlet", 1), NmrSignal(7.6, "multiplet", 5)]
        ms = [MsPeak(106, 90), MsPeak(105, 100), MsPeak(77, 80), MsPeak(51, 30)]
        labels.update(carbonyl=1, aldehyde=1, aromatic=1, monosubstituted_benzene=1)
    elif kind == "propanoic_acid":
        formula = "C3H6O2"
        ir = [IrPeak(3000, "strong", "broad"), IrPeak(_jitter(rng, 1712, 8), "strong", "sharp")]
        nmr = [NmrSignal(_jitter(rng, 11.6, 0.2), "singlet", 1), NmrSignal(2.35, "quartet", 2), NmrSignal(1.15, "triplet", 3)]
        ms = [MsPeak(74, 70), MsPeak(73, 40), MsPeak(57, 25), MsPeak(45, 50), MsPeak(29, 90), MsPeak(28, 100)]
        labels.update(carbonyl=1, carboxylic_acid=1, ethyl_group=1)
    elif kind == "methyl_acetate":
        formula = "C3H6O2"
        ir = [IrPeak(1745, "strong", "sharp"), IrPeak(1240, "strong", "sharp"), IrPeak(1040, "strong", "sharp")]
        nmr = [NmrSignal(3.65, "singlet", 3), NmrSignal(2.05, "singlet", 3)]
        ms = [MsPeak(74, 20), MsPeak(43, 100), MsPeak(59, 15)]
        labels.update(carbonyl=1, ester=1, methyl_ketone=1)
    elif kind == "ethyl_2_chloropropanoate":
        formula = "C5H9O2Cl"
        ir = [IrPeak(2980, "medium", "sharp"), IrPeak(_jitter(rng, 1742, 6), "strong", "sharp"), IrPeak(1180, "strong", "sharp")]
        nmr = [
            NmrSignal(4.39, "quartet", 1),
            NmrSignal(4.22, "quartet", 2),
            NmrSignal(1.68, "doublet", 3),
            NmrSignal(1.30, "triplet", 3),
        ]
        ms = [MsPeak(136, 5), MsPeak(91, 10), MsPeak(63, 45), MsPeak(29, 100), MsPeak(28, 30)]
        labels.update(carbonyl=1, ester=1, ethyl_group=1)
    elif kind == "butanamine":
        formula = "C4H11N"
        ir = [IrPeak(3370, "weak", "sharp"), IrPeak(3290, "weak", "sharp"), IrPeak(2930, "strong", "sharp")]
        nmr = [NmrSignal(2.68, "triplet", 2), NmrSignal(1.4, "multiplet", 4), NmrSignal(1.15, "singlet", 2), NmrSignal(0.92, "triplet", 3)]
        ms = [MsPeak(73, 12), MsPeak(30, 100)]
        labels.update(primary_amine=1)
    elif kind == "diethylamine":
        formula = "C4H11N"
        ir = [IrPeak(3280, "weak", "sharp"), IrPeak(2960, "strong", "sharp")]
        nmr = [NmrSignal(2.55, "quartet", 4), NmrSignal(1.05, "triplet", 6), NmrSignal(0.9, "singlet", 1)]
        ms = [MsPeak(73, 25), MsPeak(58, 100), MsPeak(30, 40)]
        labels.update(ethyl_group=0)
    else:
        raise ValueError(kind)
    return formula, ir, nmr, ms, labels


def build_dataset(samples: int = 1600, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    kinds = [
        "acetone",
        "propanal",
        "toluene",
        "benzaldehyde",
        "propanoic_acid",
        "methyl_acetate",
        "ethyl_2_chloropropanoate",
        "butanamine",
        "diethylamine",
    ]
    rows = []
    targets = []
    for index in range(samples):
        formula, ir, nmr, ms, labels = _sample(rng, kinds[index % len(kinds)])
        parsed = parse_formula(formula)
        result = deduce_from_peaks(formula, ir, nmr, ms, model=None)
        rows.append(featurize(parsed, hydrogen_deficiency(parsed), result["ir"], result["nmr"], result["ms"]))
        targets.append([labels[name] for name in LABELS])
    return np.vstack(rows).astype(np.float32), np.asarray(targets, dtype=np.float32)


def train(path: Path = MODEL_PATH) -> dict:
    features, labels = build_dataset()
    split = int(len(features) * 0.8)
    model = FunctionalGroupNet([features.shape[1], 48, 24, len(LABELS)], seed=7)
    losses = model.fit(features[:split], labels[:split], epochs=60, learning_rate=0.08)
    probabilities = np.vstack([model.predict_proba(row) for row in features[split:]])
    predicted = (probabilities >= 0.5).astype(np.float32)
    accuracy = float((predicted == labels[split:]).mean())
    model.save(path)
    return {"samples": int(len(features)), "final_loss": losses[-1], "holdout_accuracy": accuracy, "path": str(path)}


if __name__ == "__main__":
    print(train())
