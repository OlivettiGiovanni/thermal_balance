"""Rebuild figure CSVs from archived PDF coordinates, or from the original PDF.

The default needs only NumPy. ``--pdf`` additionally needs PyMuPDF and checks
the original paper's SHA256. Figure curves are digitizations, not raw data.
No model calculation enters this extraction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAPER_SHA256 = "8582dd6c53cd04dea33c3ee13136c4d8aed17b492a11abea64ca8cdd966a4a95"


def _vertices(drawing):
    """Read a connected line path; reject a changed PDF drawing representation."""
    points = []
    for item in drawing["items"]:
        if item[0] != "l":
            raise ValueError("Expected line segments in the archived figure curve")
        start, end = list(item[1]), list(item[2])
        if not points:
            points.append(start)
        elif not np.allclose(points[-1], start, atol=1e-5, rtol=0):
            raise ValueError("The figure path is disconnected")
        points.append(end)
    return points


def _coordinates_from_pdf(pdf, fig4, fig6):
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != PAPER_SHA256:
        raise ValueError("PDF does not match the archived source SHA256")
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError("--pdf requires PyMuPDF: pip install pymupdf") from error
    with fitz.open(pdf) as doc:
        drawings4 = doc[4].get_drawings()
        for curve in fig4["curves"].values():
            drawing = drawings4[curve["drawing_index_zero_based"]]
            if not np.allclose(drawing["color"], curve["rgb"], atol=1e-5):
                raise ValueError("Figure 4 curve color differs from provenance")
            curve["pdf_xy_points"] = _vertices(drawing)
        drawings6 = doc[5].get_drawings()
        for curve in fig6["curves"].values():
            matches = [d for d in drawings6 if d["color"] is not None
                       and np.allclose(d["color"], curve["stroke_rgb"], atol=1e-5)]
            # The plotted path spans more x coordinates than its legend sample.
            drawing = max(matches, key=lambda d: d["rect"].width)
            curve["vertices_pdf_points"] = _vertices(drawing)


def _interpolate(grid, temperature, logarithm):
    """Preserve unavailable support; never extend a published curve."""
    return np.interp(grid, temperature, logarithm, left=np.nan, right=np.nan)


def extract_figures(pdf=None):
    fig4 = json.loads((HERE / "junnemann_fig4.provenance.json").read_text())
    fig6 = json.loads((HERE / "junnemann_fig6.provenance.json").read_text())
    if pdf is not None:
        _coordinates_from_pdf(Path(pdf), fig4, fig6)
    t4 = np.arange(300.0, 1601.0, 10.0)
    xfit = fig4["temperature_calibration"]["temperature_k_per_pdf_x"]
    yfit = fig4["absorption_calibration"]["log10_alpha_per_pdf_y"]
    bounds = fig4["visible_axis_bounds_pdf"]
    low, high = np.sort(np.polyval(yfit, [bounds["top"], bounds["bottom"]]))
    columns4 = {"temperature_k": t4}
    for name, curve in fig4["curves"].items():
        xy = np.asarray(curve["pdf_xy_points"])
        logalpha = _interpolate(t4, np.polyval(xfit, xy[:, 0]), np.polyval(yfit, xy[:, 1]))
        logalpha[(logalpha < low) | (logalpha > high)] = np.nan
        columns4[name] = 10**logalpha
    t6 = np.r_[300.0, np.arange(400.0, 1601.0, 10.0)]
    x400, x1600, y16, y4 = fig6["calibration_x400_x1600_y1e-16_y1e-4_pdf_points"]
    columns6 = {"temperature_k": t6}
    for name, curve in fig6["curves"].items():
        xy = np.asarray(curve["vertices_pdf_points"])
        temperature = 400 + (xy[:, 0]-x400)*1200/(x1600-x400)
        # The orange/purple paths start at the 300 K plot edge; calibration
        # from rounded PDF ticks yields 300.00167 K. Preserve the archived
        # extraction's endpoint correction before interpolation. The blue
        # path starts above 301 K and receives no correction or extrapolation.
        if abs(temperature[0]-300.0) < 0.005:
            temperature[0] = 300.0
        logpower = -16 + (xy[:, 1]-y16)*12/(y4-y16)
        columns6[name] = 10**_interpolate(t6, temperature, logpower)
    return columns4, columns6


def save_and_verify(output, pdf=None):
    """Require agreement with every published CSV value and its NaN mask."""
    tables = extract_figures(pdf)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"coordinate_source": "PDF" if pdf is not None else "archived JSON",
              "paper_sha256": PAPER_SHA256, "figures": {}}
    for figure, table, fmt in zip((4, 6), tables, ("%.9g", "%.10e")):
        filename = f"junnemann_fig{figure}.csv"
        archived = np.genfromtxt(HERE / filename, delimiter=",", names=True)
        max_relative_error = 0.0
        for name, values in table.items():
            reference = archived[name]
            if not np.array_equal(np.isnan(values), np.isnan(reference)):
                raise ValueError(f"Figure {figure}, {name}: support/NaN mask changed")
            np.testing.assert_allclose(values, reference, rtol=5e-9, atol=0, equal_nan=True)
            valid = np.isfinite(reference) & (reference != 0)
            if valid.any():
                max_relative_error = max(max_relative_error,
                    float(np.max(np.abs(values[valid]/reference[valid]-1))))
        np.savetxt(output / filename, np.column_stack(list(table.values())),
                   delimiter=",", header=",".join(table), comments="", fmt=fmt)
        # Also compare the emitted decimal text numerically after reloading.
        rebuilt = np.genfromtxt(output / filename, delimiter=",", names=True)
        for name in table:
            np.testing.assert_array_equal(rebuilt[name], archived[name])
        report["figures"][str(figure)] = {"rows": len(table["temperature_k"]),
            "maximum_relative_error_before_decimal_rounding": max_relative_error,
            "saved_csv_numerically_identical": True, "nan_masks_identical": True}
    (output / "extraction_verification.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="Directory receiving rebuilt CSVs and verification JSON")
    parser.add_argument("--pdf", type=Path,
                        help="Optional original PDF; default uses archived coordinate JSON")
    args = parser.parse_args()
    print(json.dumps(save_and_verify(args.output, args.pdf), indent=2))
