"""Deduce a structure from peak lists or from spectrum images."""

from __future__ import annotations

from pathlib import Path

from deduction.assemble import propose_structure
from deduction.features import featurize
from deduction.formula import hydrogen_deficiency, ihd_meaning, parse_formula
from deduction.images import RawPeak, ir_from_image, load_gray, spikes_from_image, split_textbook_page
from deduction.ir import IrPeak, interpret_ir
from deduction.ms import MsPeak, interpret_ms
from deduction.nmr import NmrSignal, interpret_nmr
from deduction.net import FunctionalGroupNet

MODEL_PATH = Path(__file__).resolve().parent.parent / "saved_model" / "functional_groups.npz"
NMR_SCALES = (8.0, 10.0, 12.0, 14.0)
IR_RANGES = ((4000.0, 500.0), (4000.0, 400.0), (3800.0, 600.0))
# Fraction across the textbook IR chart, from the 4000 cm-1 tick to the 500 cm-1 tick.
IR_CHART = (
    (0.00, 4000.0),
    (0.147, 3333.0),
    (0.306, 2500.0),
    (0.403, 2000.0),
    (0.533, 1667.0),
    (0.630, 1429.0),
    (0.701, 1250.0),
    (0.758, 1111.0),
    (0.838, 909.0),
    (0.899, 769.0),
    (1.00, 500.0),
)


_CACHED_MODEL: FunctionalGroupNet | None = None
_MODEL_LOADED = False


def _model() -> FunctionalGroupNet | None:
    global _CACHED_MODEL, _MODEL_LOADED
    if _MODEL_LOADED:
        return _CACHED_MODEL
    _MODEL_LOADED = True
    if MODEL_PATH.exists():
        _CACHED_MODEL = FunctionalGroupNet.load(MODEL_PATH)
    return _CACHED_MODEL


def deduce_from_peaks(
    formula: str,
    ir_peaks: list[IrPeak] | None = None,
    nmr_signals: list[NmrSignal] | None = None,
    ms_peaks: list[MsPeak] | None = None,
    model: FunctionalGroupNet | None = None,
) -> dict:
    parsed = parse_formula(formula)
    deficiency = hydrogen_deficiency(parsed)
    ir = interpret_ir(ir_peaks or [])
    nmr = interpret_nmr(nmr_signals or [])
    ms = interpret_ms(ms_peaks or [], parsed)
    proposal = propose_structure(parsed, deficiency, ir, nmr, ms)
    vector = featurize(parsed, deficiency, ir, nmr, ms)
    network = model if model is not None else _model()
    probabilities = None if network is None else network.probabilities(vector)
    best = proposal["best"]
    return {
        "formula": parsed.raw,
        "mass": parsed.mass,
        "isotope_masses": parsed.isotope_masses(),
        "counts": parsed.counts,
        "ihd": deficiency,
        "ihd_meaning": ihd_meaning(deficiency),
        "ir": ir,
        "nmr": nmr,
        "ms": ms,
        "pieces": proposal["pieces"],
        "candidates": proposal["candidates"],
        "confident": proposal["confident"],
        "structure": None
        if best is None
        else {
            "name": best["name"],
            "iupac": best["iupac"],
            "smiles": best["smiles"],
            "structure": best["structure"],
            "reasons": best["reasons"],
            "score": best["score"],
        },
        "functional_group_probabilities": probabilities,
    }


def _chart_wavenumber(fraction: float) -> float:
    fraction = min(1.0, max(0.0, fraction))
    for (left_fraction, left_cm), (right_fraction, right_cm) in zip(IR_CHART, IR_CHART[1:]):
        if fraction <= right_fraction or right_fraction == 1.0:
            span = right_fraction - left_fraction
            if span <= 0:
                return left_cm
            weight = (fraction - left_fraction) / span
            return left_cm + weight * (right_cm - left_cm)
    return 500.0


def _ir_from_raw(peaks: list[RawPeak], left: float, right: float, chart: bool = False) -> list[IrPeak]:
    converted = []
    for peak in peaks:
        if chart:
            wavenumber = _chart_wavenumber(peak.fraction)
        else:
            wavenumber = left + peak.fraction * (right - left)
        if peak.strength >= 0.55:
            intensity = "strong"
        elif peak.strength >= 0.30:
            intensity = "medium"
        else:
            intensity = "weak"
        shape = "broad" if peak.width >= 0.04 else "sharp"
        converted.append(IrPeak(wavenumber=float(wavenumber), intensity=intensity, shape=shape))
    return converted


def _nmr_from_raw(peaks: list[RawPeak], ppm_left: float) -> list[NmrSignal]:
    lines = []
    for peak in peaks:
        if peak.strength < 0.22:
            continue
        delta = ppm_left * (1.0 - peak.fraction)
        if delta < 0.15:
            continue
        lines.append((delta, peak.strength))
    if not lines:
        return []
    lines.sort(key=lambda item: item[0], reverse=True)
    groups: list[list[tuple[float, float]]] = [[lines[0]]]
    for line in lines[1:]:
        if groups[-1][-1][0] - line[0] <= 0.16:
            groups[-1].append(line)
        else:
            groups.append([line])
    signals = []
    for group in groups:
        center = sum(delta for delta, _strength in group) / len(group)
        multiplicity = {1: "singlet", 2: "doublet", 3: "triplet", 4: "quartet"}.get(len(group), "multiplet")
        signals.append(NmrSignal(delta=float(center), multiplicity=multiplicity, integral=None))
    return signals


def _scale_ms(peaks: list[RawPeak], axis_max: float) -> list[MsPeak]:
    by_mz: dict[int, float] = {}
    for peak in peaks:
        mz = int(round(peak.fraction * axis_max))
        if mz < 1:
            continue
        intensity = float(round(peak.strength * 100))
        by_mz[mz] = max(by_mz.get(mz, 0.0), intensity)
    return [MsPeak(mz=float(mz), intensity=intensity) for mz, intensity in sorted(by_mz.items())]


def _ms_axis_candidates(peaks: list[RawPeak], mass: int) -> list[float]:
    """Chart m/z 0 is the left axis. A strong interior peak may be the molecular ion."""
    if not peaks:
        return []
    axes = []
    for peak in peaks:
        if peak.fraction < 0.5 or peak.strength < 0.45:
            continue
        axis_max = mass / peak.fraction
        if mass * 0.98 <= axis_max <= max(mass * 2.6, mass + 80):
            axes.append(axis_max)
    # Charts in this set also end on a round m/z a little past the molecular ion.
    for end in (50, 80, 90, 100, 120, 150, 200, 250, 300, 400):
        if end >= mass:
            axes.append(float(end))
    # Keep a handful of distinct scales.
    unique = []
    for axis in axes:
        if all(abs(axis - other) > 3 for other in unique):
            unique.append(axis)
    return unique[:8]


def deduce_from_raw(
    formula: str,
    ir_raw: list[RawPeak] | None = None,
    nmr_raw: list[RawPeak] | None = None,
    ms_raw: list[RawPeak] | None = None,
    nmr_ppm_left: float | None = None,
    ir_left: float | None = None,
    ir_right: float | None = None,
    ms_axis_max: float | None = None,
    ir_chart: bool = False,
) -> dict:
    """Calibrate image peaks on the axes given, or on the usual chart scales."""
    parsed = parse_formula(formula)
    nmr_scales = (nmr_ppm_left,) if nmr_ppm_left else NMR_SCALES
    if ir_chart:
        ir_jobs = ((4000.0, 500.0, True),)
    elif ir_left and ir_right:
        ir_jobs = ((ir_left, ir_right, False),)
    else:
        ir_jobs = tuple((left, right, False) for left, right in IR_RANGES) + ((4000.0, 500.0, True),)
    if ms_axis_max:
        ms_axes = [float(ms_axis_max)]
    else:
        ms_axes = _ms_axis_candidates(ms_raw or [], parsed.mass) or [float(parsed.mass)]
    best = None
    trials = []
    for axis_max in ms_axes:
        ms_peaks = _scale_ms(ms_raw or [], axis_max)
        for ppm_left in nmr_scales:
            nmr_signals = _nmr_from_raw(nmr_raw or [], ppm_left)
            for left, right, chart in ir_jobs:
                ir_peaks = _ir_from_raw(ir_raw or [], left, right, chart=chart)
                result = deduce_from_peaks(formula, ir_peaks, nmr_signals, ms_peaks)
                score = 0 if result["structure"] is None else result["structure"]["score"]
                trials.append(
                    {
                        "nmr_ppm_left": ppm_left,
                        "ir_left": left,
                        "ir_right": right,
                        "ir_chart": chart,
                        "ms_axis_max": round(axis_max, 1),
                        "score": score,
                        "name": None if result["structure"] is None else result["structure"]["name"],
                    }
                )
                if best is None or score > best[0]:
                    result["calibration"] = {
                        "nmr_ppm_left": ppm_left,
                        "ir_left": left,
                        "ir_right": right,
                        "ir_chart": chart,
                        "ms_axis_max": round(axis_max, 1),
                    }
                    best = (score, result)
    assert best is not None
    # A photographed chart is noisier than a typed peak list. Keep a structure
    # only when the rules agree by a clear margin.
    if best[0] < 8:
        best[1]["structure"] = None
        best[1]["confident"] = False
    trials.sort(key=lambda item: item["score"], reverse=True)
    best[1]["calibration_trials"] = trials[:12]
    return best[1]


def deduce_from_images(
    formula: str,
    ir_path: str | None = None,
    nmr_path: str | None = None,
    ms_path: str | None = None,
    page_path: str | None = None,
    nmr_ppm_left: float | None = None,
    ir_left: float | None = None,
    ir_right: float | None = None,
) -> dict:
    ir_raw: list[RawPeak] = []
    nmr_raw: list[RawPeak] = []
    ms_raw: list[RawPeak] = []
    if page_path:
        gray = load_gray(page_path)
        boxes = split_textbook_page(gray)
        kinds = ("mass", "ir", "nmr")
        for kind, box in zip(kinds, boxes):
            if kind == "mass":
                ms_raw = spikes_from_image(gray, box)
            elif kind == "ir":
                ir_raw = ir_from_image(gray, box)
            else:
                nmr_raw = spikes_from_image(gray, box)
    if ms_path:
        ms_raw = spikes_from_image(ms_path)
    if ir_path:
        ir_raw = ir_from_image(ir_path)
    if nmr_path:
        nmr_raw = spikes_from_image(nmr_path)
    result = deduce_from_raw(
        formula,
        ir_raw=ir_raw,
        nmr_raw=nmr_raw,
        ms_raw=ms_raw,
        nmr_ppm_left=nmr_ppm_left,
        ir_left=ir_left,
        ir_right=ir_right,
    )
    result["extracted"] = {
        "ir_fractions": [round(peak.fraction, 3) for peak in ir_raw],
        "nmr_fractions": [round(peak.fraction, 3) for peak in nmr_raw],
        "ms_fractions": [round(peak.fraction, 3) for peak in ms_raw],
    }
    return result
