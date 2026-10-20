# Molecular structure deduction

The program calculates the index of hydrogen deficiency and proposes a structure from a molecular formula plus an IR spectrum, a 1H-NMR spectrum, and a mass spectrum.

Two inputs are supported:

- Typed peak lists, through the API.
- Photographs of the charts. Upload an infrared chart, a mass spectrum, and a ¹H-NMR spectrum. The program reads peak positions from each image and interprets them with the formula.

The saved network is `saved_model/functional_groups.npz`. It estimates functional-group probabilities. The structure name, SMILES, and the written reasons come from the correlation rules (IHD, IR, NMR splitting, and the mass fragments). A structure is returned only when those rules agree. Otherwise the response still contains the index, the groups, the fragments, and the probabilities.

## Front end

```powershell
streamlit run streamlit_app.py
```

The page asks for the data in deduction order. The molecular formula comes first, and the index of hydrogen deficiency is shown before any spectrum. Upload the infrared chart, then the mass spectrum, then the ¹H-NMR spectrum. Each image is described on its own step. The last step states the structure those three readings support.

## Public app

Repository: https://github.com/Ganapathy-bot/molecular-structure-deduction

The app is set up for Streamlit Community Cloud from this repository.

- Branch: `main`
- Main file path: `streamlit_app.py`
- Python version: 3.12
- Secrets: leave empty

In [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app**, then **Yup, I have an app**, and paste this file URL:

https://github.com/Ganapathy-bot/molecular-structure-deduction/blob/main/streamlit_app.py

The free public app is served at a `streamlit.app` address.

## Run the API

```powershell
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

CORS is open so a local front end on another port can call it.

`GET /api/health` returns `{ "ok": true, "model": ".../saved_model/functional_groups.npz", "model_ready": true }`.

## Typed peaks

`POST /api/deduce/peaks` with JSON:

```json
{
  "formula": "C3H6O",
  "ir": [
    {"wavenumber": 1715, "intensity": "strong", "shape": "sharp"},
    {"wavenumber": 2920, "intensity": "medium", "shape": "sharp"}
  ],
  "nmr": [
    {"delta": 2.2, "multiplicity": "singlet", "integral": 6}
  ],
  "ms": [
    {"mz": 58, "intensity": 60},
    {"mz": 43, "intensity": 100},
    {"mz": 15, "intensity": 18}
  ]
}
```

`intensity` is `weak`, `medium`, or `strong`. `shape` is `sharp` or `broad`. NMR `multiplicity` is `singlet`, `doublet`, `triplet`, `quartet`, `multiplet`, or `unknown`. Mass `intensity` is the percent of the base peak.

## Chart photographs

`POST /api/deduce` as multipart form data.

| Field | Required | Meaning |
| --- | --- | --- |
| `formula` | yes | For example `C3H6O` or `C5H9O2Cl` |
| `ir` | no | IR photograph |
| `nmr` | no | 1H-NMR photograph |
| `ms` | no | Mass-spectrum photograph |
| `page` | no | One textbook page with mass, then IR, then NMR |
| `nmr_ppm_left` | no | Left-hand ppm of the NMR axis, usually 8, 10, 12, or 14 |
| `ir_left`, `ir_right` | no | IR axis ends in cm-1, for example 4000 and 500 |

Send at least one file. Separate `ir`, `nmr`, and `ms` files override a combined `page`. If you omit the axis fields, the program tries the scales used in this problem set and keeps the reading the rules score highest.

## Response

Useful fields:

- `ihd` and `ihd_meaning`. The index is `(2C + 2 + N + P - H - X) / 2`. Oxygen and sulfur are not counted. Halogens count as X.
- `structure` when the call is confident: `name`, `iupac`, `smiles`, `structure`, `reasons`, `score`. `null` when the evidence is not decisive.
- `candidates` and `pieces` for the partial reading.
- `functional_group_probabilities` from the saved network.
- `calibration` for an image: the ppm scale, the IR scale, and the mass-axis maximum that were used.
- `extracted` for an image: peak positions as fractions across each plot (`0` is the left edge, `1` is the right edge).

## What the photographs can and cannot do

Separate chart photos with a clear axis, like the three compound A files, are read directly. A phone photo of a whole textbook page is split into three bands and is less certain: grid lines and captions can be mistaken for peaks. Pass `nmr_ppm_left`, `ir_left`, and `ir_right` when you know the axis. The typed peak endpoint is the one to trust when a photograph is faint.

The network was trained on spectra generated from these correlation rules, not on a large library of laboratory charts. It reports whether the reading looks like a ketone, aldehyde, acid, ester, amine, aromatic ring, or ethyl group. It does not invent a structure outside the evidence.
