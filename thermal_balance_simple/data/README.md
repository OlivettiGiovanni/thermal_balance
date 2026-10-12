# Material spectra

These are the unchanged spectra from the previous project. All are supplied as
`wavelength_um,n,k` and assumed temperature independent. They control thermal
radiation, not the imported laser heating. None is calibrated to your exact film.

| File | Material / numerical origin | Wavelength band |
|---|---|---|
| `sin_cataldo_2012.csv` | Low-stress LPCVD SiNx; published five-oscillator fit | 1–666.7 µm |
| `sin_gautam_2016.csv` | Amorphous LPCVD SiNx; reconstructed rounded fit parameters | 0.2105–15.498 µm |
| `sin_luke_2015.csv` | Annealed LPCVD Si3N4; digitized figure | 1.4–31.6 µm |

Luke's small k values are limited by graphical resolution (~0.02); graphical
zeros are not measured zero absorption. Gautam's long-wavelength tail beyond
15.5 µm is missing. Different bands confound a direct material comparison.
The retained `provenance/` files describe the previous reconstruction; generator
scripts referenced there belong to the original project and are not runtime dependencies.

Sources: [Cataldo 2012](https://doi.org/10.1364/OL.37.004200),
[Gautam 2016](https://doi.org/10.1116/1.4954192),
[Luke 2015](https://doi.org/10.1364/OL.40.004823).

Thermal radiation uses the angle/polarization-averaged absorption cross-section,
following [Greffet et al., generalized Kirchhoff law](https://arxiv.org/abs/1601.00312).
Here ε = (n+ik)², α_j = V(ε−1)/[1+L_j(ε−1)], k0 = 2π/λ, and
C_abs,j = k0 Im(α_j) / |1−i k0³ α_j/(6π)|²; the average is
(C_parallel+2 C_perpendicular)/3. The depolarization factors satisfy
L_parallel+2 L_perpendicular=1. Radiation reaction follows
[Albaladejo et al.](https://doi.org/10.1364/OE.18.003556).
Gas cooling follows the molecular heat-transfer expression used by
[Vinante et al.](https://arxiv.org/abs/1903.08492), Eq. (2).

The diagnostic checks max(1,|n+ik|) 2πa/λ > 0.3, where a is the largest
spheroid semiaxis. The reported fraction is weighted by modeled emitted power
at T_decomp. It is not an error estimate or a correction to the model.
