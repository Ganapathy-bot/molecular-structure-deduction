"""Read peak positions from a photographed IR, 1H-NMR, or mass spectrum.

Stick spectra (mass and NMR) are columns that are darker than the paper
beside them, so a faint pen still counts. The IR trace is the lower edge of
the pen after horizontal grid lines are removed. Axis units are applied later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class RawPeak:
    fraction: float  # 0 at the left edge of the plot, 1 at the right edge
    strength: float  # 0 to 1, relative to the strongest peak in that spectrum
    width: float = 0.0  # fraction of the plot width


def load_gray(source) -> np.ndarray:
    if isinstance(source, np.ndarray):
        return source
    image = Image.open(source).convert("L")
    return np.asarray(image, dtype=np.uint8)


def _crop(gray: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    height, width = gray.shape
    left, top, right, bottom = box
    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)
    if right - left < 30 or bottom - top < 30:
        return gray
    return gray[top:bottom, left:right]


def _contrast(gray: np.ndarray) -> np.ndarray:
    """How much darker each pixel is than the paper a few columns away."""
    values = gray.astype(np.float32)
    height, width = values.shape
    total = np.zeros_like(values)
    for offset in (4, 5, 6, 7, 8):
        left = np.pad(values, ((0, 0), (offset, 0)), mode="edge")[:, :width]
        right = np.pad(values, ((0, 0), (0, offset)), mode="edge")[:, offset:]
        total += left + right
    return total / 10.0 - values


def _longest_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    best = np.zeros(width, dtype=np.int32)
    end = np.zeros(width, dtype=np.int32)
    current = np.zeros(width, dtype=np.int32)
    for y_pos in range(height):
        current = np.where(mask[y_pos], current + 1, 0)
        better = current > best
        end = np.where(better, y_pos, end)
        best = np.where(better, current, best)
    return best, end


def _baseline(gray: np.ndarray) -> tuple[int, int, int]:
    """Return the axis row and the left and right ends of that axis."""
    height, width = gray.shape
    threshold = float(np.percentile(gray, 22))
    ink = gray < threshold
    counts = ink[:, int(width * 0.08) : int(width * 0.92)].sum(axis=1)
    y0 = int(height * 0.28)
    y1 = int(height * 0.88)
    baseline = y0 + int(np.argmax(counts[y0:y1]))
    xs = np.where(ink[max(0, baseline - 1) : baseline + 2].any(axis=0))[0]
    if len(xs) < 8:
        return baseline, int(width * 0.06), int(width * 0.94)
    return baseline, int(np.percentile(xs, 2)), int(np.percentile(xs, 98))


def extract_spikes(gray: np.ndarray) -> list[RawPeak]:
    """Vertical sticks of a mass or 1H-NMR spectrum."""
    height, width = gray.shape
    if height < 30 or width < 40:
        return []
    contrast = _contrast(gray)
    baseline, axis_left, axis_right = _baseline(gray)
    hot = contrast > 12
    hot[baseline + 6 :] = False
    runs, ends = _longest_runs(hot)
    # A real stick either reaches the axis or is a long dark run. A speck at the
    # top of the page fails both tests, so it cannot be given the full axis height.
    columns = []
    for x_pos in range(width):
        if runs[x_pos] <= 12:
            continue
        bottom = int(ends[x_pos])
        top = bottom - int(runs[x_pos]) + 1
        span = baseline - top
        reaches_axis = baseline - bottom <= 28
        long_stroke = runs[x_pos] >= 40 and span > 0 and runs[x_pos] >= 0.45 * span
        if span < 10 or not (reaches_axis or long_stroke):
            continue
        columns.append(x_pos)
    if len(columns) == 0:
        return []
    groups: list[list[int]] = [[int(columns[0])]]
    for x_pos in columns[1:]:
        if int(x_pos) - groups[-1][-1] <= 3:
            groups[-1].append(int(x_pos))
        else:
            groups.append([int(x_pos)])

    strokes = []
    for group in groups:
        tops = []
        medians = []
        for x_pos in group:
            bottom = int(ends[x_pos])
            top = bottom - int(runs[x_pos]) + 1
            sample = contrast[top : bottom + 1, x_pos]
            tops.append(top)
            medians.append(float(np.median(sample)))
        if not tops:
            continue
        top = min(tops)
        median = max(medians)
        x_pos = int(np.median(group))
        strokes.append({"x": x_pos, "top": top, "median": median, "height": baseline - top})

    if not strokes:
        return []
    best_median = max(stroke["median"] for stroke in strokes)
    strokes = [stroke for stroke in strokes if stroke["median"] >= max(22.0, 0.45 * best_median)]
    strokes = [stroke for stroke in strokes if stroke["height"] >= 10]
    if not strokes:
        return []

    # Full-height borders sit in the outer part of the photograph.
    frames = [
        stroke
        for stroke in strokes
        if stroke["top"] < height * 0.22
        and stroke["height"] > (baseline - stroke["top"]) * 0.7
        and (stroke["x"] < width * 0.18 or stroke["x"] > width * 0.82)
    ]
    left_frames = [stroke["x"] for stroke in frames if stroke["x"] < width * 0.5]
    right_frames = [stroke["x"] for stroke in frames if stroke["x"] > width * 0.5]
    left = min(left_frames) if left_frames else axis_left
    right = max(right_frames) if right_frames else axis_right
    if right - left < width * 0.25:
        left, right = axis_left, axis_right
    span = max(right - left, 1)

    peaks = []
    for stroke in strokes:
        if not (left + span * 0.04 < stroke["x"] < right - span * 0.03):
            continue
        if stroke["height"] < 10:
            continue
        peaks.append(stroke)
    if not peaks:
        return []
    tallest = max(stroke["height"] for stroke in peaks)
    raw = []
    for stroke in peaks:
        if stroke["height"] < tallest * 0.10:
            continue
        raw.append(
            RawPeak(
                fraction=(stroke["x"] - left) / span,
                strength=min(1.0, stroke["height"] / tallest),
                width=0.01,
            )
        )
    return _merge_fractions(raw, gap=0.012)


def extract_ir(gray: np.ndarray) -> list[RawPeak]:
    """Valleys in an IR trace. Fraction 0 is the left axis, fraction 1 the right."""
    height, width = gray.shape
    if height < 30 or width < 40:
        return []
    threshold = float(np.percentile(gray, 8))
    ink = gray < threshold
    top_counts = ink.sum(axis=1).copy()
    top_counts[int(height * 0.4) :] = 0
    top = int(np.argmax(top_counts))
    bottom_counts = ink.sum(axis=1).copy()
    bottom_counts[: int(height * 0.42)] = 0
    bottom_counts[int(height * 0.9) :] = 0
    bottom = int(np.argmax(bottom_counts))
    if bottom - top < 40:
        return []
    xs = np.where(ink[top])[0]
    if len(xs) < 10:
        xs = np.where(ink[max(0, bottom - 2) : bottom + 3].any(axis=0))[0]
    if len(xs) < 10:
        return []
    left = int(np.percentile(xs, 2))
    right = int(np.percentile(xs, 98))
    if right - left < width * 0.4:
        left, right = int(width * 0.06), int(width * 0.94)
    plot = ink[top + 2 : bottom - 2, left:right]
    if plot.size == 0:
        return []
    keep = plot.mean(axis=1) < 0.42
    plot_height, plot_width = plot.shape
    trace = np.full(plot_width, np.nan)
    for x_pos in range(plot_width):
        rows = np.where(plot[:, x_pos] & keep)[0]
        if len(rows):
            trace[x_pos] = float(rows.max())
    good = np.where(~np.isnan(trace))[0]
    if len(good) < 10:
        return []
    filled = np.interp(np.arange(plot_width), good, trace[good])
    smooth = np.convolve(filled, np.ones(9) / 9, mode="same")
    floor = float(np.percentile(smooth, 15))
    span = max(float(np.percentile(smooth, 98) - floor), 1.0)
    dips: list[RawPeak] = []
    x_pos = 8
    while x_pos < plot_width - 8:
        depth = float(smooth[x_pos] - floor)
        left_max = float(smooth[x_pos - 6 : x_pos].max())
        right_max = float(smooth[x_pos + 1 : x_pos + 7].max())
        if smooth[x_pos] < left_max or smooth[x_pos] < right_max or depth < span * 0.28:
            x_pos += 1
            continue
        left_edge = x_pos
        while left_edge > 1 and smooth[left_edge] > floor + depth * 0.5:
            left_edge -= 1
        right_edge = x_pos
        while right_edge < plot_width - 2 and smooth[right_edge] > floor + depth * 0.5:
            right_edge += 1
        dips.append(
            RawPeak(
                fraction=x_pos / max(plot_width - 1, 1),
                strength=min(1.0, depth / span),
                width=(right_edge - left_edge) / max(plot_width - 1, 1),
            )
        )
        x_pos = max(right_edge + 2, x_pos + 1)
    return _merge_fractions(dips, gap=0.02)


def _merge_fractions(peaks: list[RawPeak], gap: float) -> list[RawPeak]:
    if not peaks:
        return []
    ordered = sorted(peaks, key=lambda peak: peak.fraction)
    merged = [ordered[0]]
    for peak in ordered[1:]:
        if peak.fraction - merged[-1].fraction <= gap:
            if peak.strength > merged[-1].strength:
                merged[-1] = peak
        else:
            merged.append(peak)
    return merged


def spikes_from_image(source, box: tuple[int, int, int, int] | None = None) -> list[RawPeak]:
    gray = load_gray(source)
    if box is not None:
        gray = _crop(gray, box)
    return extract_spikes(gray)


def ir_from_image(source, box: tuple[int, int, int, int] | None = None) -> list[RawPeak]:
    gray = load_gray(source)
    if box is not None:
        gray = _crop(gray, box)
    return extract_ir(gray)


def split_textbook_page(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Three stacked plots. Pages in this set are mass, then IR, then NMR."""
    height, width = gray.shape
    threshold = float(np.percentile(gray, 14))
    projection = (gray < threshold).mean(axis=1)
    smooth = np.convolve(projection, np.ones(41) / 41, mode="same")
    margin = int(height * 0.08)
    minima: list[tuple[float, int]] = []
    for y_pos in range(margin, height - margin):
        if smooth[y_pos] > smooth[y_pos - 1] or smooth[y_pos] > smooth[y_pos + 1]:
            continue
        window = smooth[max(0, y_pos - 90) : y_pos + 90]
        score = float(window.max() - smooth[y_pos])
        if score >= 0.02 and smooth[y_pos] <= float(np.percentile(smooth, 40)):
            minima.append((score, y_pos))
    kept: list[tuple[float, int]] = []
    for score, y_pos in sorted(minima, reverse=True):
        if all(abs(y_pos - other) > height * 0.16 for _other_score, other in kept):
            kept.append((score, y_pos))
    cuts: list[int] = []
    best_total = -1.0
    for index, (score_a, y_a) in enumerate(kept):
        for score_b, y_b in kept[index + 1 :]:
            top_cut, bottom_cut = sorted((y_a, y_b))
            bands = (top_cut, bottom_cut - top_cut, height - bottom_cut)
            if min(bands) < height * 0.18 or max(bands) > height * 0.52:
                continue
            total = score_a + score_b
            if total > best_total:
                best_total = total
                cuts = [top_cut, bottom_cut]
    if len(cuts) < 2:
        third = height // 3
        cuts = [third, 2 * third]
    bands = [(int(height * 0.02), cuts[0]), (cuts[0], cuts[1]), (cuts[1], height - 2)]
    boxes = []
    for top, bottom in bands:
        rows = np.where(projection[top:bottom] > 0.01)[0]
        if len(rows) == 0:
            boxes.append((int(width * 0.04), top, int(width * 0.96), bottom))
            continue
        band_top = top + int(rows[0])
        band_bottom = top + int(rows[-1])
        boxes.append((int(width * 0.03), max(0, band_top - 8), int(width * 0.97), min(height, band_bottom + 8)))
    return boxes
