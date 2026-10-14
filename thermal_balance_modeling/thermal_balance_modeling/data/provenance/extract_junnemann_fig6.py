"""Extract plotted Fig. 6 vector curves; requires PyMuPDF and NumPy, not raw author data."""
import argparse
import hashlib
import json
from pathlib import Path

import fitz
import numpy as np


def curve_vertices(page, color):
    """Find the longest matching-color curve, excluding the short legend samples."""
    candidates = [drawing for drawing in page.get_drawings()
                  if drawing.get("color") and np.allclose(drawing["color"], color, atol=2e-5)
                  and drawing["rect"].x0 > 300 and drawing["rect"].y0 < 300]
    drawing = max(candidates, key=lambda item: item["rect"].width)
    points = [drawing["items"][0][1]] + [item[-1] for item in drawing["items"]]
    return np.array([[point.x, point.y] for point in points])


def calibrated_curve(vertices, calibration):
    """Convert PDF points to temperature and logarithmic power using labeled axis ticks."""
    x400, x1600, y16, y4 = calibration
    temperature = 400 + (vertices[:, 0] - x400) * 1200 / (x1600 - x400)
    log_power = -16 + (y16 - vertices[:, 1]) * 12 / (y16 - y4)
    temperature[np.abs(temperature - 300) < 0.01] = 300  # PDF coordinate rounding only.
    unique, reverse = np.unique(temperature[::-1], return_index=True)
    return unique, log_power[::-1][reverse]  # Keep the last vertex if a vertical segment repeats x.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    document = fitz.open(args.pdf)
    page = document[5]
    if "FIG. 6." not in page.get_text() or abs(page.rect.width - 589.5) > 0.01:
        raise ValueError("Expected published PRB 111, 195304 (2025), page 195304-6")
    colors = {"p_abs_w": [0.95294, 0.52941, 0.003922],
              "p_linear_w": [0.74510, 0.18039, 0.83922],
              "p_bb_w": [0.41569, 0.77255, 0.99608]}
    calibration = [356.3080139160156, 533.4190063476562, 260.44598388671875, 72.1190185546875]
    temperature = np.r_[300., np.arange(400., 1601., 10.)]
    columns, raw = [temperature], {}
    for name, color in colors.items():
        vertices = curve_vertices(page, color)
        t_curve, log_power = calibrated_curve(vertices, calibration)
        columns.append(10**np.interp(temperature, t_curve, log_power, left=np.nan, right=np.nan))
        raw[name] = {"stroke_rgb": color, "vertices_pdf_points": vertices.tolist()}
    args.output.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output / "junnemann_fig6.csv", np.column_stack(columns), delimiter=",",
               header="temperature_k," + ",".join(colors), comments="", fmt="%.10e")
    provenance = {"source_filename": args.pdf.name, "source_sha256": hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
                  "doi": "10.1103/PhysRevB.111.195304", "page": "195304-6", "figure": 6,
                  "calibration_x400_x1600_y1e-16_y1e-4_pdf_points": calibration,
                  "method": "PDF vector centerlines; linear interpolation in temperature and log10(power)",
                  "curves": raw}
    (args.output / "provenance").mkdir(exist_ok=True)
    (args.output / "provenance" / "junnemann_fig6.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
