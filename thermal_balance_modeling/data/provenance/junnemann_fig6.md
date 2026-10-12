# Jünnemann et al., Fig. 6: plotted reference curves

Source: M. Jünnemann et al., *Physical Review B* **111**, 195304 (2025),
DOI [10.1103/PhysRevB.111.195304](https://doi.org/10.1103/PhysRevB.111.195304),
published PDF page 195304-6, Fig. 6. The supplied PDF's SHA256 and exact
vector coordinates are recorded in `junnemann_fig6.json`.

**These are curves extracted from the published figure, not raw author data
and not an independent calculation or validation of silicon thermal radiation.**

Figure parameters: silicon sphere diameter 260 nm; laser wavelength 1550 nm;
incident intensity 0.1 W/um² (= 1e11 W/m²); surroundings 300 K.
Columns in `../junnemann_fig6.csv`:

| Column | Published curve |
|---|---|
| `temperature_k` | Particle temperature, K |
| `p_abs_w` | Orange: absorption including the paper's temperature/intensity-dependent contributions |
| `p_linear_w` | Purple: absorption using only alpha0 = 4.3e-6 cm^-1 |
| `p_bb_w` | Blue: the paper's net blackbody-radiation cooling curve |

The orange, purple and blue vector paths contain 37, 2 and 45 vertices,
respectively. Legend samples are excluded. Temperatures are calibrated against
the 400 and 1600 K grid lines; log10(power) against the 1e-16 and 1e-4 W grid
lines. Interpolation is linear in temperature and log10(power), matching the
figure's axes. The CSV includes 300 K and 400-1600 K at 10 K intervals.
No power values are extrapolated. The blue value at 300 K is `nan`: a zero
net power cannot be represented on a logarithmic plot. Setting it to zero in
a thermal model is a separate bath-equilibrium condition, not figure extraction.
The omitted 300-400 K interval must not be used as an independently resolved
reference interval. Original vector vertices remain available in the JSON.

Vector extraction avoids raster-pixel uncertainty, but retains figure-line
simplification and interpolation errors. Stored digits do not imply comparable
physical accuracy. Do not extrapolate these curves above 1600 K. The experimentally
measured loss pressures in Fig. 5 are separate observations; Fig. 6 contains no
gas-cooling curve and does not specify a pressure.

Reproduce extraction (optional dependencies: PyMuPDF, NumPy):

```bash
python data/provenance/extract_junnemann_fig6.py /path/to/published_article.pdf
```

Independent check of the purple curve: with miepython 3.3.0, radius 130 nm,
wavelength 1550 nm, I=1e11 W/m², n=3.48 and k=alpha0*lambda/(4*pi), the Mie
absorption gives approximately 1.69578e-13 W, versus 1.69566e-13 W extracted
from the figure (0.008% difference). This checks the linear-heating scale;
it neither establishes the authors' exact refractive-index choice nor validates
the blue radiation curve.
