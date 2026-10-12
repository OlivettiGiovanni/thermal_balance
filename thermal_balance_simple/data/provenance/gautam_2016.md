# Gautam 2016 optical constants

`sin_gautam_2016.csv` contains **n and k reconstructed from the published, rounded fitted dielectric-function parameters**. It is not a direct download of the authors' numerical data and is not a digitization of a graph. No parameters were adjusted to match another material or a desired result. Parameter rounding in the source table means exact reproduction of the authors' internal, unrounded model is not guaranteed.

The CSV has the header `wavelength_um,n,k`, increasing wavelengths, and 4,000 logarithmically spaced rows over **0.210499488–15.498024804 µm**, equivalent to the publication's **0.08–5.89 eV** range. Wavelength is the vacuum wavelength. The convention is `N = n + i*k`, `epsilon = N**2`, with nonnegative k.

## Source and specimen

L. K. Gautam, L. Ye and N. J. Podraza, *LPCVD SiNx thin film on c-Si wafer by spectroscopic ellipsometry*, **Surface Science Spectra 23**, 51–54 (2016), [doi:10.1116/1.4954192](https://doi.org/10.1116/1.4954192). The fit is **Table II, page 53**; Table III describes the silicon substrate and is deliberately excluded.

This sample is a 49 nm amorphous LPCVD film on crystalline silicon, nominally stoichiometric Si3N4 (SiNx, x=1.33), measured at 300 K in air. Source accession: 01363; supplier: Encompass Distribution SVC LLC. The recovered source does **not establish a numerical tensile stress or verify that this particular specimen is high-stress**. It provides a stoichiometric LPCVD surrogate for comparison; it is not a certified match to the user's film.

The actual page containing Table II and its equations is publicly reproduced, with permission, in N. Podraza and M. R. Linford, *Archiving the Optical Constants of Materials with Surface Science Spectra*, **Vacuum Technology & Coating**, October 2017, pp. 2–7. The original page 53 is Figure 3 on magazine page 5; the original specimen description on page 51 is Figure 1 on magazine page 3.

- [Author-uploaded article and PDF](https://www.researchgate.net/publication/320044783_Archiving_the_Optical_Constants_of_Materials_with_Surface_Science_Spectra)
- [Author-uploaded PDF download](https://www.researchgate.net/profile/Matthew-Linford/publication/320044783_Archiving_the_Optical_Constants_of_Materials_with_Surface_Science_Spectra/links/59cab5dda6fdcc451d582142/Archiving-the-Optical-Constants-of-Materials-with-Surface-Science-Spectra.pdf)
- [Parameter-page figure](https://www.researchgate.net/figure/Page-from-the-Podraza-groups-SSS-submission-on-silicon-nitride-1-Reprinted-with_fig3_320044783)

Sources accessed 9 October 2026. The source article is copyrighted; the generated dataset and implementation should link to it, not bundle the complete article as project documentation.

SHA-256 of the downloaded author reprint PDF (3,371,309 bytes):

```text
050c13042c34ae02ae9d525022e16b106d607de541f93afc185b5a53b6eed918
```

SHA-256 of the generated CSV:

```text
3dda84f4afa9a0765b520b92e1e6c485e626ec39ae9bef2d0854f0c809bb24b3
```

## How the reconstruction works

The dielectric function is the sum of a constant, a Sellmeier term, a Cody–Lorentz term, and three causal Gaussian oscillators. The script uses the central values from Table II:

| Term | Parameters |
|---|---|
| Constant | epsilon_inf = 1.54 ± 0.04 |
| Sellmeier | En = 8.46 ± 0.05 eV; A = 158 ± 4 eV² |
| Cody–Lorentz | E0 = 6.11 ± 0.02 eV; A = 2.1 ± 0.8 eV; Gamma = 1.18 ± 0.05 eV; Eg = 1.9 ± 0.3 eV; Ep = 5 ± 2 eV |
| Gaussian 1 | En = 894 ± 14 cm⁻¹; A = 5.0 ± 0.7; Gamma = 189 ± 14 cm⁻¹ |
| Gaussian 2 | En = 821.6 ± 0.9 cm⁻¹; A = 7 ± 1; Gamma = 103 ± 6 cm⁻¹ |
| Gaussian 3 | En = 684 ± 25 cm⁻¹; A = 3.2 ± 0.2; Gamma = 598 ± 35 cm⁻¹ |

Gamma is the Gaussian full width at half maximum, and Gaussian amplitudes are dimensionless. These quoted fit uncertainties are recorded for traceability; the CSV uses the central fit and does not propagate them.

The Sellmeier contribution is `A / (En**2 - E**2)`. It does not introduce an extra `+1`; the separate epsilon_inf term supplies the additive constant. The Cody–Lorentz imaginary component is the exact zero-tail expression printed on page 53. Its real component is obtained by a principal-value Kramers–Kronig integral, including numerical integration of the analytic oscillator expression to infinite energy. This integration is part of evaluating the published oscillator, not a claim of measured optical constants beyond 5.89 eV.

For the Gaussian terms, `sigma = Gamma / (2*sqrt(log(2)))`, and

```text
epsilon2 = A * [exp(-((E-En)/sigma)**2) - exp(-((E+En)/sigma)**2)]
epsilon1 = 2*A/sqrt(pi) * [Dawson((E+En)/sigma) - Dawson((E-En)/sigma)]
```

**Printed-equation correction:** the reproduced page 53 omits the squares in its Gaussian exponents. The code uses the causal Gaussian actually defined by Gautam's cited Ref. 8: D. De Sousa Meneses, M. Malki and P. Echegut, *Structure and lattice dynamics of binary lead silicate glasses investigated by infrared spectroscopy*, **Journal of Non-Crystalline Solids 352**, 769–776 (2006), [doi:10.1016/j.jnoncrysol.2006.02.004](https://doi.org/10.1016/j.jnoncrysol.2006.02.004), Eqs. (1)–(3). Its squared exponents and Dawson transform are explicit in the [available full primary text](https://www.academia.edu/87811083/Structure_and_lattice_dynamics_of_binary_lead_silicate_glasses_investigated_by_infrared_spectroscopy). Omitting those squares would not produce Gaussian absorption and would give an unphysical response. This correction is disclosed rather than silently changing the printed equation.

## Regeneration and validation

From the project root, with NumPy, SciPy and Matplotlib installed:

```bash
python data/generate_gautam_2016.py
python data/validate_gautam_2016.py
```

The generator requires only NumPy and SciPy. Validation separately checks the analytic Gaussian real part by numerical Kramers–Kronig integration, the Cody–Lorentz principal value by singularity subtraction, the integration split convergence, finite/passive optical constants, increasing wavelength, and conversion back from n,k to epsilon. Results are written to `validation_gautam_2016.json`.

The maximum absolute discrepancies were 8.0e-15 for the Gaussian transform, 2.8e-16 for the alternative Cody integral, 2.3e-16 for the Cody integration-split check, and 5.1e-10 for the CSV n,k-to-epsilon round trip. All 4,000 rows were finite and passive, with strictly increasing wavelength. An independent implementation review found no errors in parameter transcription, Gaussian FWHM/sign, units, or principal-value integration.

The optional validation script creates `data/gautam_2016_reconstructed_fit.png`, which shows the calculated full spectrum and the infrared detail. **It is not an overlay against the original plotted curves:** the recovered author reprint contains the parameter table and equations but does not reproduce the SiNx curve figures, and those figures or the raw tabulation were not retrieved. The validation establishes the implementation of the recovered model, not an independent numerical comparison with the authors' raw dataset.

## Limits for thermal balance

- Use it as a room-temperature material spectrum. Temperature affects the Planck distribution in the thermal model; this file supplies no measured n(T), k(T).
- Do not silently extend the spectrum beyond 15.498 µm or below 0.210499 µm. A radiation integral restricted to this interval omits contributions outside it.
- The paper itself notes that the lowest-frequency Gaussian is poorly constrained and may account for effects below the measured energy range. This matters near the long-wavelength boundary.
- Tiny or zero calculated k in the near-infrared reflects the chosen finite-oscillator fit and its detection/model limits; it does not establish the ultra-low laser absorption of an arbitrary high-stress membrane. Keep experimental/resonant laser P_abs separate.
- Do not add the paper's silicon-substrate Drude term to the SiNx spectrum.
