"""Structure deduction from a formula and three uploaded spectra.

The page collects evidence in one order: molecular formula and the index of
hydrogen deficiency, then the infrared chart, the mass spectrum, and the
proton NMR. Each image is described before a structure is proposed.

Run from this folder:

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from deduction.engine import deduce_from_raw
from deduction.formula import hydrogen_deficiency, ihd_meaning, parse_formula
from deduction.images import ir_from_image, spikes_from_image

STEPS = ("formula", "infrared", "mass", "nmr", "conclusion")
STEP_LABELS = {
    "formula": "1  Formula",
    "infrared": "2  Infrared",
    "mass": "3  Mass",
    "nmr": "4  ¹H-NMR",
    "conclusion": "5  Structure",
}
GROUP_NAMES = {
    "carbonyl": "Carbonyl",
    "aldehyde": "Aldehyde",
    "ketone": "Ketone",
    "carboxylic_acid": "Carboxylic acid",
    "ester": "Ester",
    "primary_amine": "Primary amine",
    "aromatic": "Aromatic ring",
    "monosubstituted_benzene": "Monosubstituted benzene",
    "ethyl_group": "Ethyl group",
    "methyl_ketone": "Methyl ketone",
}


def _default_ms_axis(mass: int) -> int:
    for end in (50, 80, 100, 120, 150, 200, 250, 300, 400, 500, 800):
        if end >= mass:
            return end
    return mass + 20


def _style() -> None:
    st.html(
        """
        <style>
          .stApp {
            background:
              radial-gradient(900px 280px at 0% -10%, rgba(31,111,91,0.08), transparent 50%),
              #f3efe4;
          }
          [data-testid="stHeader"] { background: transparent; }
          h1, h2, h3 { font-family: Palatino Linotype, Palatino, Georgia, serif; letter-spacing: -0.02em; }
          div[data-testid="stMetric"] {
            background: #fffaf2;
            border: 1px solid rgba(31,111,91,0.22);
            padding: 0.55rem 0.75rem;
          }
        </style>
        """
    )


def _header() -> None:
    st.html(
        """
        <div style="padding:4px 0 8px;">
          <div style="font-family:Cascadia Mono,Consolas,monospace; letter-spacing:0.22em; font-size:12px; color:#1f6f5b;">
            STRUCTURE FROM SPECTRA
          </div>
          <div style="font-family:Palatino Linotype,Palatino,Georgia,serif; font-size:42px; line-height:1; color:#1c1914; margin-top:4px;">
            Deduce the structure
          </div>
          <div style="font-family:Palatino Linotype,Palatino,Georgia,serif; font-size:18px; color:#5c5346; margin-top:6px;">
            Formula and index first. Then the infrared chart, the mass spectrum, and the proton NMR.
          </div>
        </div>
        """
    )


def _formula_text() -> str:
    """The draft is edited on the first step. Later steps keep the committed formula."""
    if st.session_state.get("step", "formula") == "formula" and "formula_draft" in st.session_state:
        return st.session_state.get("formula_draft") or ""
    return st.session_state.get("formula_committed") or ""


def _formula_state() -> tuple[object | None, float | None, str]:
    text = _formula_text()
    try:
        parsed = parse_formula(text)
        value = hydrogen_deficiency(parsed)
    except ValueError as error:
        return None, None, str(error) if text.strip() else "Enter a molecular formula to calculate the index."
    return parsed, value, ihd_meaning(value)


def _seed(widget_key: str, saved_key: str, default):
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state.get(saved_key, default)


def _keep(widget_key: str, saved_key: str) -> None:
    if widget_key in st.session_state:
        st.session_state[saved_key] = st.session_state[widget_key]


def _saved(saved_key: str, default):
    return st.session_state.get(saved_key, default)


def _step_open(name: str) -> bool:
    parsed, value, _meaning = _formula_state()
    formula_ok = parsed is not None and value is not None and st.session_state.get("formula_committed") == parsed.raw
    if name == "formula":
        return True
    if name == "infrared":
        return formula_ok
    if name == "mass":
        return formula_ok and bool(st.session_state.get("ir_ready"))
    if name == "nmr":
        return formula_ok and bool(st.session_state.get("ms_ready"))
    return formula_ok and bool(st.session_state.get("nmr_ready"))


def _rail() -> None:
    step = st.session_state.get("step", "formula")
    columns = st.columns(5)
    for column, name in zip(columns, STEPS):
        opened = _step_open(name)
        kind = "primary" if name == step else "secondary"
        if column.button(STEP_LABELS[name], key=f"go_{name}", disabled=not opened, type=kind, width="stretch"):
            st.session_state.step = name
            st.rerun()


def _arithmetic(parsed) -> str:
    phosphorus = parsed.counts.get("P", 0)
    numerator = 2 * parsed.C + 2 + parsed.N + phosphorus - parsed.H - parsed.X
    return (
        f"(2×{parsed.C} + 2 + {parsed.N} + {phosphorus} − {parsed.H} − {parsed.X}) / 2 "
        f"= {numerator:g} / 2 = {numerator / 2:g}"
    )


def _card(kicker: str, title: str, body: str) -> None:
    paragraphs = "".join(f"<p style='margin:8px 0 0;'>{line}</p>" for line in body.split("\n") if line.strip())
    st.html(
        f"""
        <div style="background:#fffaf2; border:1px solid rgba(28,25,20,0.08); border-left:4px solid #1f6f5b; padding:18px 20px; margin:12px 0;">
          <div style="font-family:Cascadia Mono,Consolas,monospace; letter-spacing:0.16em; font-size:11px; color:#1f6f5b;">{kicker}</div>
          <div style="font-family:Palatino Linotype,Palatino,Georgia,serif; font-size:26px; line-height:1.1; margin-top:4px; color:#1c1914;">{title}</div>
          <div style="font-family:Palatino Linotype,Palatino,Georgia,serif; font-size:17px; color:#3c3830;">{paragraphs}</div>
        </div>
        """
    )


def _ihd_ladder(value: float | None) -> None:
    rungs = [
        (0, "0", "single bonds only"),
        (1, "1", "one double bond or one ring"),
        (2, "2", "two degrees"),
        (3, "3", "three degrees"),
        (4, "≥ 4", "a benzene ring is allowed"),
    ]
    cells = []
    for number, label, caption in rungs:
        active = value is not None and ((number < 4 and value == number) or (number == 4 and value >= 4))
        background = "#1f6f5b" if active else "#fffaf2"
        color = "#fffaf2" if active else "#1c1914"
        border = "#1f6f5b" if active else "rgba(28,25,20,0.12)"
        cells.append(
            f"""
            <div style="flex:1; min-width:120px; border:1px solid {border}; background:{background}; color:{color}; padding:12px 14px;">
              <div style="font-family:Cascadia Mono,Consolas,monospace; font-size:11px; letter-spacing:0.14em;">IHD {label}</div>
              <div style="font-family:Palatino Linotype,Palatino,Georgia,serif; font-size:15px; margin-top:4px;">{caption}</div>
            </div>
            """
        )
    st.html(f'<div style="display:flex; gap:8px; flex-wrap:wrap; margin:8px 0 4px;">{"".join(cells)}</div>')


def _write_upload(data: bytes, name: str, folder: Path) -> str:
    suffix = Path(name).suffix.lower() or ".png"
    if suffix not in {".png", ".jpg", ".jpeg"}:
        suffix = ".png"
    target = folder / f"spectrum{suffix}"
    target.write_bytes(data)
    return str(target)


def _analyze(formula: str, *, ir: bool = False, mass: bool = False, nmr: bool = False) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        ir_raw = []
        ms_raw = []
        nmr_raw = []
        if ir and st.session_state.get("ir_bytes"):
            ir_raw = ir_from_image(_write_upload(st.session_state.ir_bytes, st.session_state.get("ir_name", "ir.png"), folder))
        if mass and st.session_state.get("ms_bytes"):
            ms_raw = spikes_from_image(_write_upload(st.session_state.ms_bytes, st.session_state.get("ms_name", "ms.png"), folder))
        if nmr and st.session_state.get("nmr_bytes"):
            nmr_raw = spikes_from_image(_write_upload(st.session_state.nmr_bytes, st.session_state.get("nmr_name", "nmr.png"), folder))
        use_ir = bool(_saved("use_ir_axis_saved", False))
        use_ms = bool(_saved("use_ms_axis_saved", False))
        use_nmr = bool(_saved("use_nmr_axis_saved", False))
        chart = bool(_saved("ir_chart_saved", False)) and use_ir
        return deduce_from_raw(
            formula,
            ir_raw=ir_raw,
            nmr_raw=nmr_raw,
            ms_raw=ms_raw,
            nmr_ppm_left=float(_saved("nmr_left_saved", 10)) if use_nmr else None,
            ir_left=None if chart or not use_ir else float(_saved("ir_left_saved", 4000)),
            ir_right=None if chart or not use_ir else float(_saved("ir_right_saved", 500)),
            ms_axis_max=float(_saved("ms_axis_saved", 150)) if use_ms else None,
            ir_chart=chart,
        )


def _describe_ir(result: dict) -> str:
    ir = result["ir"]
    peaks = ir["peaks"]
    calibration = result.get("calibration") or {}
    if calibration.get("ir_chart"):
        axis = "The wavenumber axis is the compressed chart scale, 4000 cm⁻¹ on the left and 500 cm⁻¹ on the right."
    else:
        left = float(calibration.get("ir_left", 4000))
        right = float(calibration.get("ir_right", 500))
        axis = f"The wavenumber axis runs from {left:.0f} cm⁻¹ on the left to {right:.0f} cm⁻¹ on the right."
    if not peaks:
        return (
            axis
            + " No bands were read from this image. The trace may be faint, or the axis may not match the chart. Adjust the axis and read it again."
        )
    strong = [peak for peak in peaks if peak["intensity"] == "strong"]
    listed = ", ".join(
        f"{peak['wavenumber']:.0f} cm⁻¹ ({peak['intensity']}, {peak['shape']})" for peak in peaks[:8]
    )
    extra = "" if len(peaks) <= 8 else f" {len(peaks) - 8} weaker marks were left off this list."
    lines = [
        f"{axis} {len(peaks)} bands were read. The marked positions are {listed}.{extra}",
    ]
    present = [group for group in ir["groups"] if group["present"]]
    if present:
        lines.append(" ".join(group["detail"] for group in present))
    else:
        lines.append("None of those bands was strong enough, or in the right place, to name a functional group.")
    return "\n".join(lines)


def _describe_ms(result: dict) -> str:
    ms = result["ms"]
    axis = float((result.get("calibration") or {}).get("ms_axis_max") or _saved("ms_axis_saved", 0))
    axis_line = f"The mass axis runs from m/z 0 on the left to m/z {axis:.0f} on the right."
    peaks = ms["peaks"]
    if not peaks:
        return axis_line + " No sticks were read from this image. Check that the right-hand end of the axis is the m/z value printed there."
    shown = ", ".join(f"m/z {peak['mz']} ({peak['intensity']}%)" for peak in peaks[:10])
    extra = "" if len(peaks) <= 10 else f" {len(peaks) - 10} smaller sticks were left off this list."
    lines = [f"{axis_line} {len(peaks)} peaks were read: {shown}.{extra}"]
    lines.extend(ms["notes"])
    named = [item for item in ms["fragments"] if str(item["label"]).startswith("m/z")]
    if named:
        lines.append(" ".join(f"m/z {item['mz']}: {item['detail']}" for item in named[:6]))
    return "\n".join(lines)


def _describe_nmr(result: dict) -> str:
    nmr = result["nmr"]
    left = float((result.get("calibration") or {}).get("nmr_ppm_left") or _saved("nmr_left_saved", 10))
    axis = f"The proton axis runs from δ {left:g} on the left to δ 0 on the right."
    signals = nmr["signals"]
    if not signals:
        return axis + " No signals were read from this image. If the left-hand label is not 10 ppm, set that value and read the chart again."
    bits = []
    for signal in signals:
        protons = "" if signal["integral"] is None else f", {signal['integral']:g}H"
        bits.append(f"δ {signal['delta']:.2f} ({signal['multiplicity']}{protons})")
    lines = [
        f"{axis} {len(signals)} signals were read: {', '.join(bits)}.",
        "Integration is not taken from a photograph, so the proton counts below are blank unless a signal could be assigned from its shift and splitting alone.",
    ]
    lines.extend(signal["assignment"] for signal in signals)
    return "\n".join(lines)


def _remember_upload(upload, bytes_key: str, name_key: str, ready_key: str) -> None:
    if upload is None:
        return
    data = upload.getvalue()
    if data != st.session_state.get(bytes_key):
        st.session_state[bytes_key] = data
        st.session_state[name_key] = upload.name
        st.session_state[ready_key] = False
        st.session_state.pop("conclusion", None)


def _formula_step() -> None:
    st.markdown("### Molecular formula")
    st.caption("This comes before any spectrum. The index of hydrogen deficiency is calculated from the formula alone.")
    if "formula_draft" not in st.session_state and st.session_state.get("formula_committed"):
        st.session_state.formula_draft = st.session_state.formula_committed
    st.text_input(
        "Molecular formula",
        key="formula_draft",
        placeholder="C9H10O2",
        help="Use C, H, N, O, S, P, F, Cl, Br, and I. Example: C7H6O2",
    )
    parsed, value, meaning = _formula_state()
    if parsed is None or value is None:
        st.info(meaning)
        _ihd_ladder(None)
        st.button("Continue to the infrared chart", disabled=True, type="primary")
        return
    one, two, three = st.columns(3)
    one.metric("Formula", parsed.raw)
    two.metric("Nominal mass", parsed.mass)
    three.metric("IHD", f"{value:g}")
    st.markdown(f"**IHD** = {_arithmetic(parsed)}")
    st.caption(meaning)
    _ihd_ladder(value)
    if parsed.X:
        st.caption(f"Halogen is present, so the mass spectrum may also show M+2 at m/z {parsed.mass + 2}.")
    if st.button("Continue to the infrared chart", type="primary"):
        previous = st.session_state.get("formula_committed")
        st.session_state.formula_committed = parsed.raw
        if previous != parsed.raw:
            st.session_state.ms_axis_saved = _default_ms_axis(parsed.mass)
            st.session_state.pop("ms_axis", None)
            for key in ("ir_ready", "ms_ready", "nmr_ready", "conclusion"):
                st.session_state.pop(key, None)
        st.session_state.step = "infrared"
        st.rerun()


def _spectrum_step(kind: str) -> None:
    parsed, _value, _meaning = _formula_state()
    assert parsed is not None
    if kind == "infrared":
        title = "Infrared spectrum"
        lead = "Upload the infrared chart. High wavenumber is on the left."
        upload_label = "Infrared image"
        upload_key = "ir_file"
        bytes_key, name_key, ready_key = "ir_bytes", "ir_name", "ir_ready"
        next_step, next_label, back_step, back_label = "mass", "Continue to the mass spectrum", "formula", "Back to the formula"
        describe = _describe_ir
    elif kind == "mass":
        title = "Mass spectrum"
        lead = "Upload the mass spectrum after the infrared chart. m/z increases from left to right."
        upload_label = "Mass spectrum image"
        upload_key = "ms_file"
        bytes_key, name_key, ready_key = "ms_bytes", "ms_name", "ms_ready"
        next_step, next_label, back_step, back_label = "nmr", "Continue to the proton NMR", "infrared", "Back to the infrared chart"
        describe = _describe_ms
    else:
        title = "¹H-NMR spectrum"
        lead = "Upload the proton NMR last. δ decreases from left to right."
        upload_label = "Proton NMR image"
        upload_key = "nmr_file"
        bytes_key, name_key, ready_key = "nmr_bytes", "nmr_name", "nmr_ready"
        next_step, next_label, back_step, back_label = "conclusion", "Conclude the structure", "mass", "Back to the mass spectrum"
        describe = _describe_nmr

    st.markdown(f"### {title}")
    st.caption(lead)
    upload = st.file_uploader(upload_label, type=["png", "jpg", "jpeg"], key=upload_key)
    _remember_upload(upload, bytes_key, name_key, ready_key)

    with st.expander("The peaks look shifted"):
        st.caption("Leave this closed for an automatic axis. Open it when the numbers on the chart do not match the description.")
        if kind == "infrared":
            _seed("use_ir_axis", "use_ir_axis_saved", False)
            _seed("ir_left", "ir_left_saved", 4000.0)
            _seed("ir_right", "ir_right_saved", 500.0)
            _seed("ir_chart", "ir_chart_saved", False)
            st.checkbox("Use this infrared axis", key="use_ir_axis")
            _keep("use_ir_axis", "use_ir_axis_saved")
            left, right = st.columns(2)
            left.number_input("Left edge (cm⁻¹)", min_value=500.0, max_value=4500.0, step=100.0, key="ir_left")
            right.number_input("Right edge (cm⁻¹)", min_value=400.0, max_value=4000.0, step=100.0, key="ir_right")
            st.checkbox("Compressed textbook axis", key="ir_chart", help="The chart is marked 4000, 3000, 2000, 1500, then squeezes toward 500.")
            _keep("ir_left", "ir_left_saved")
            _keep("ir_right", "ir_right_saved")
            _keep("ir_chart", "ir_chart_saved")
        elif kind == "mass":
            _seed("use_ms_axis", "use_ms_axis_saved", False)
            _seed("ms_axis", "ms_axis_saved", _default_ms_axis(parsed.mass))
            st.checkbox("Use this mass axis", key="use_ms_axis")
            _keep("use_ms_axis", "use_ms_axis_saved")
            st.number_input(
                "Right-hand m/z",
                min_value=20.0,
                max_value=1200.0,
                step=10.0,
                key="ms_axis",
                help="The m/z printed at the right end of the axis.",
            )
            _keep("ms_axis", "ms_axis_saved")
        else:
            _seed("use_nmr_axis", "use_nmr_axis_saved", False)
            _seed("nmr_left", "nmr_left_saved", 10.0)
            st.checkbox("Use this NMR axis", key="use_nmr_axis")
            _keep("use_nmr_axis", "use_nmr_axis_saved")
            st.number_input(
                "Left-hand δ (ppm)",
                min_value=1.0,
                max_value=20.0,
                step=1.0,
                key="nmr_left",
                help="The ppm value printed at the left end of the axis.",
            )
            _keep("nmr_left", "nmr_left_saved")

    data = st.session_state.get(bytes_key)
    if data:
        st.image(data, caption=st.session_state.get(name_key, ""), width="stretch")
    if st.button("Read this chart", type="primary", key=f"read_{kind}"):
        if not data:
            st.error("Upload the chart first.")
        else:
            try:
                flags = {"ir": kind == "infrared", "mass": kind == "mass", "nmr": kind == "nmr"}
                result = _analyze(parsed.raw, **flags)
                st.session_state[f"{kind}_text"] = describe(result)
                st.session_state[ready_key] = True
                st.session_state.pop("conclusion", None)
            except ValueError as error:
                st.session_state[ready_key] = False
                st.error(str(error))
    text = st.session_state.get(f"{kind}_text")
    if st.session_state.get(ready_key) and text:
        _card("WHAT THIS CHART SHOWS", title, text)
        if st.button(next_label, type="primary", key=f"next_{kind}"):
            st.session_state.step = next_step
            st.rerun()
    if st.button(back_label, key=f"back_{kind}"):
        st.session_state.step = back_step
        st.rerun()


def _probabilities(values: dict | None) -> None:
    if not values:
        return
    kept = [(key, float(raw)) for key, raw in values.items() if float(raw) >= 0.5]
    if not kept:
        return
    st.markdown("**Groups also marked from the same reading**")
    rows = []
    for key, raw in sorted(kept, key=lambda item: item[1], reverse=True):
        percent = max(0.0, min(1.0, raw)) * 100
        rows.append(
            f"""
            <div style="display:grid; grid-template-columns: 190px 1fr 42px; gap:8px; align-items:center; margin:4px 0;">
              <div style="font-size:15px; color:#1c1914;">{GROUP_NAMES.get(key, key)}</div>
              <div style="height:8px; background:#e7e1d4;">
                <div style="width:{percent:.0f}%; height:8px; background:#1f6f5b;"></div>
              </div>
              <div style="font-family:Cascadia Mono,Consolas,monospace; font-size:12px;">{percent:.0f}</div>
            </div>
            """
        )
    st.html("".join(rows))


def _conclusion_step() -> None:
    parsed, value, meaning = _formula_state()
    assert parsed is not None and value is not None
    signature = (
        parsed.raw,
        len(st.session_state.get("ir_bytes") or b""),
        len(st.session_state.get("ms_bytes") or b""),
        len(st.session_state.get("nmr_bytes") or b""),
        bool(_saved("use_ir_axis_saved", False)),
        float(_saved("ir_left_saved", 4000)),
        float(_saved("ir_right_saved", 500)),
        bool(_saved("ir_chart_saved", False)),
        bool(_saved("use_ms_axis_saved", False)),
        float(_saved("ms_axis_saved", 0)),
        bool(_saved("use_nmr_axis_saved", False)),
        float(_saved("nmr_left_saved", 10)),
    )
    if st.session_state.get("conclusion_sig") != signature or "conclusion" not in st.session_state:
        try:
            st.session_state.conclusion = _analyze(parsed.raw, ir=True, mass=True, nmr=True)
            st.session_state.conclusion_sig = signature
            st.session_state.conclusion_error = ""
        except ValueError as error:
            st.session_state.conclusion_error = str(error)
    if st.session_state.get("conclusion_error"):
        st.error(st.session_state.conclusion_error)
        return
    result = st.session_state.conclusion

    st.markdown("### Conclusion")
    st.caption("The argument follows the same order as the data: formula, infrared, mass, proton NMR, then the structure.")
    _card(
        "1  ·  FORMULA AND INDEX",
        parsed.raw,
        f"Nominal mass {parsed.mass}. IHD = {_arithmetic(parsed)}. {meaning}",
    )
    _ihd_ladder(value)

    st.markdown("#### Infrared")
    if st.session_state.get("ir_bytes"):
        st.image(st.session_state.ir_bytes, width="stretch")
    _card("2  ·  INFRARED", "What the infrared chart shows", _describe_ir(result))

    st.markdown("#### Mass spectrum")
    if st.session_state.get("ms_bytes"):
        st.image(st.session_state.ms_bytes, width="stretch")
    _card("3  ·  MASS", "What the mass spectrum shows", _describe_ms(result))

    st.markdown("#### Proton NMR")
    if st.session_state.get("nmr_bytes"):
        st.image(st.session_state.nmr_bytes, width="stretch")
    _card("4  ·  ¹H-NMR", "What the proton spectrum shows", _describe_nmr(result))

    structure = result.get("structure")
    if structure:
        reasons = "\n".join(structure["reasons"])
        body = f"{structure['iupac']}\n{structure['structure']}\nSMILES {structure['smiles']}\n{reasons}"
        _card("5  ·  STRUCTURE", structure["name"], body)
    else:
        pieces = result.get("pieces") or []
        if pieces:
            body = "The three spectra support these pieces, and they do not yet fix one complete structure.\n" + "\n".join(pieces)
        else:
            body = "The three spectra do not yet fix one complete structure. Read any faint chart again after checking its axis."
        _card("5  ·  STRUCTURE", "Not fixed as one structure", body)
    _probabilities(result.get("functional_group_probabilities"))
    if st.button("Back to the proton NMR"):
        st.session_state.step = "nmr"
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Structure from spectra", page_icon="⚗", layout="wide")
    if "step" not in st.session_state:
        st.session_state.step = "formula"
    _style()
    _header()
    _rail()
    step = st.session_state.get("step", "formula")
    if not _step_open(step):
        st.session_state.step = "formula"
        step = "formula"
    if step == "formula":
        _formula_step()
    elif step == "infrared":
        _spectrum_step("infrared")
    elif step == "mass":
        _spectrum_step("mass")
    elif step == "nmr":
        _spectrum_step("nmr")
    else:
        _conclusion_step()


main()
