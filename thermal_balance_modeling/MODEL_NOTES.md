# Model definitions and limits

This is a transparent reconstruction of the silicon calculation and a provisional SiN extension. Each particle has one uniform internal temperature. Matching one plotted curve does not validate every heating or cooling term. Internal units are SI; pressure inputs/outputs are mbar, temperature is K, and coefficient plots use cm⁻¹.

## Silicon benchmark

The benchmark uses a homogeneous sphere with radius 130 nm, vacuum wavelength 1550 nm, fixed real refractive index 3.48, and default incident intensity 0.1 W/µm² = 10¹¹ W/m². These reproduce the scale of Jünnemann's Fig. 6; the linear-absorption Mie result differs from the digitized purple curve by approximately 0.007%.

The absorption coefficient is

\[
\alpha_{\rm tot}=\alpha_0+\beta(T)I+\alpha_{\rm FCA}(T)+\alpha_{\rm BGA}(T),
\qquad \alpha_0=4.3\times10^{-4}\ {\rm m^{-1}}.
\]

The driver explicitly uses photon energy **0.797 eV** in the TPA/BGA fits. This is the energy quoted in the Sinclair model, not the exact conversion of 1550 nm (approximately 0.799898 eV). Mie scattering and FCA retain 1550 nm. The coefficient functions independently default to the exact `hc/lambda`; the benchmark override is recorded in `summary.json`.

### Two-photon absorption: two variants

Let \(N_j=[\exp(\theta_j/T)-1]^{-1}\), \(\Omega_j=k_B\theta_j\), and \(D=2E_\gamma-E_g\). The two phonon branches have \(\theta_j=(212,670)\) K. Sinclair's gap fit is \(E_g=1.156-7.021\times10^{-4}T^2/(1108+T)\) eV.

The `printed` variant implements the population/sign pairing printed in the cited preprint:

\[
\beta=10^{-11} E_g^{3/2}\sum_jK_j\{N_j(D-\Omega_j)^2+(N_j+1)(D+\Omega_j)^2\}\quad [{\rm m/W}],
\]

with \(K_j=(0.233,2.138)\) in cm/(GW eV⁷ᐟ²). The factor 10⁻¹¹ converts cm/GW to m/W. The `physical` variant instead pairs phonon emission with \((N_j+1)(D-\Omega_j)_+^2\) and phonon absorption with \(N_j(D+\Omega_j)_+^2\), where \(x_+=\max(x,0)\). It is an explicitly modified model using the same fitted constants, **not a new fit or a claimed erratum**. Sinclair's measurements span 5.5–300 K; both high-temperature curves extrapolate.

### Thermal carriers and empirical FCA

The default carrier density omits Sturm's narrowing correction:

\[
n_e=n_h=n_i=3.87\times10^{22}T^{3/2}\exp[-0.605/(k_BT)]\quad [{\rm m^{-3}}],
\]

where energies and \(k_B\) are in eV and eV/K. Empirical cross sections are \(\sigma_e=1.01\times10^{-12}T\lambda^2\) and \(\sigma_h=0.51\times10^{-12}T\lambda^2\); wavelength in metres gives cross sections in m². Then \(\alpha_{\rm FCA}=n_i(\sigma_e+\sigma_h)\).

The additional diagnostic curve solves Sturm's implicit expression \(n_i=3.87\times10^{16}T^{3/2}\exp[-(0.605+\Delta\epsilon)/(k_BT)]\) in cm⁻³, with \(\Delta\epsilon=-7.1\times10^{-10}\sqrt{n_i/T}\) eV. The principal Lambert-W branch gives the lower-density solution. This empirical relation has no real solution above approximately 1633.84 K; the function raises rather than inventing a continuation. The main comparison deliberately uses the uncorrected density for both FCA models.

Only thermally generated equilibrium carriers are included. There is no photo-carrier generation, carrier lifetime, or carrier-recombination model.

### Drude alternative

For each species \(s=e,h\),

\[
\alpha_{{\rm FCA},s}=\frac{n_s e^2\Gamma_s}{\epsilon_0 n_r c m_s^*(\omega^2+\Gamma_s^2)},
\qquad \Gamma_s=\frac{e}{m_s^*\mu_s}.
\]

This retains finite damping but assumes weak optical loss and a fixed background index. Sinclair's conductivity masses are \((m_e^*,m_h^*)=(0.3,0.4)m_0\), with \(n_r=3.48\). These are not density-of-states masses.

The original Arora low-impurity limits are \(\mu_e=88(T/300)^{-0.57}+7.4\times10^8T^{-2.33}\) and \(\mu_h=54.3(T/300)^{-0.57}+1.36\times10^8T^{-2.23}\), in cm²/(V s); multiply by 10⁻⁴ for SI. Ionized impurity density is zero: intrinsic carriers are not substituted for dopants. Arora's stated validation range is 250–500 K. Above it, mobility and fixed-mass extrapolations are illustrative; additional carrier scattering is omitted. These parameters were not calibrated to force agreement with Sturm.

### Bandgap absorption

The Sturm optical threshold is \(E_g^{\rm opt}=1.155-4.73\times10^{-4}T^2/(635+T)\) eV. Its zero-temperature value includes a 15 meV exciton correction relative to the 1.17 eV electronic gap. With the same phonon occupations and energies defined above,

\[
\alpha_{\rm BGA}=100\sum_j\{(N_j+1)f_j(E_\gamma-E_g^{\rm opt}-\Omega_j)+N_jf_j(E_\gamma-E_g^{\rm opt}+\Omega_j)\}\quad[{\rm m^{-1}}].
\]

Here \(f_1(x)=0.504\sqrt{x_+}+392(x-0.0055)_+^2\) and \(f_2(x)=18.08\sqrt{x_+}+5760(x-0.0055)_+^2\). Numerical energies are in eV and the fitted responses are in cm⁻¹. These phonon-assisted edge fits are silicon-specific. Sturm's measurements chiefly cover 773–1073 K, not the entire plotted range.

### Coefficients, Mie power, and remaining mismatch

`sphere_heating` converts \(\kappa=\alpha_{\rm tot}\lambda/(4\pi)\), evaluates Mie absorption with index \(n-i\kappa\), then calculates \(P_{\rm abs}=I\pi R^2Q_{\rm abs}\). In this driver, TPA uses the incident intensity to form a spatially uniform effective bulk coefficient. This is **not** a self-consistent nonlinear field calculation: true local TPA depends on \(\int I_{\rm local}^2dV\), and absorption can alter the internal field.

The independently digitized Fig. 4 FCA is approximately 1.773 cm⁻¹ at 800 K and 1549.8 cm⁻¹ at 1600 K. The uncorrected Sturm implementation gives 3.950 and 1798.3 cm⁻¹; the corrected implementation gives 4.561 and 8481.1 cm⁻¹. Neither exactly reproduces that published curve. These differences are retained and quantified in `summary.json`; no hidden adjustment is fitted to remove them.

## SiN extension

The bundled `sin_cataldo_2012.csv` is the unchanged five-oscillator reconstruction
from the preceding project, for low-stress LPCVD SiNx, numerically evaluated over
1–666.7 µm. It is not a measurement of the user's particular cylinder or an
annealed high-stress film. Its fixed optical constants are a provisional input.

Radius selection uses the supplied absorption-per-volume sweep and configured height. `max_peak` takes the global maximum; `first_peak` takes the first interior maximum with at least 3% relative prominence. The bundled reported peak is a single reported point, not a reconstructed radius sweep.

The selected linear power remains the calibration anchor. Define \(M_1=\int_V I_{\rm local}dV\) [W m] and \(M_2=\int_V I_{\rm local}^2dV\) [W²/m]. Then \(P_{\rm linear}=\alpha_0M_1\), \(P_{\rm TPA}=\beta M_2\), and \(P_{\rm FCA}=\alpha_{\rm FCA}M_1\). The CSV columns are `intensity_integral_w_m` and `intensity_squared_integral_w2_per_m`. Their local intensity normalization must match the material coefficient convention; an incident flux alone does not specify these moments. One measured/simulated linear power alone identifies neither \(\alpha_0\) nor \(M_2\).

No silicon nonlinear constants are assigned to SiN. An optional material-specific table supplies `temperature_k,beta_m_w,alpha_fca_m_inv`; it must cover the full calculation range. Nonzero terms require their corresponding field moment. Without the table, TPA and FCA are zero **by assumption**. SiN BGA is also zero by assumption. Fixed moments neglect temperature-induced field redistribution; this limitation matters near resonances.

## Cooling and pressure

Gas loss is free-molecular air exchange: \(P_{\rm gas}=g\,p_{\rm mbar}(T-T_g)\), with \(g=aA\,100\bar v(\gamma+1)/[8T_g(\gamma-1)]\), \(\bar v=\sqrt{8k_BT_g/(\pi m)}\), \(m=28.965\) u and \(\gamma=1.4\). The factor 100 converts mbar to Pa. The particle temperature varies; the configured gas bath remains fixed. The accommodation coefficient is an input assumption. Continuum/transition-regime corrections are not included.

For **silicon**, radiation is interpolated from digitized Fig. 6 and used as a reference input, not independently reproduced. Between the first visible point at 400 K and 300 K, the driver explicitly bridges linearly to zero net exchange. Therefore agreement of the plotted blue curve is not BB validation.

For **SiN**, net radiation is \(4\pi\int C_{\rm abs}(\lambda,T)[B_\lambda(T)-B_\lambda(T_g)]d\lambda\). A spheroid matches the cylinder volume and aspect ratio; its orientation-averaged dipole cross section includes radiation reaction. This is provisional, not full-wave cylinder emission. Integration covers only the supplied spectral band. Spectra without a temperature column are held fixed; temperature-dependent spectra are interpolated without extrapolation. The reported emission fraction exceeding the heuristic dipole-size limit 0.3 is a diagnostic, not an error bound.

Pressure analysis follows the cold stable branch of \(p(T)=[P_{\rm abs}(T)-P_{\rm BB}(T)]/[g(T-T_g)]\), using interpolated powers. A branch minimum is reported as a fold, not automatically as decomposition. A vacuum equilibrium or crossing of the configured SiN decomposition threshold has a separate status. A fold alone does not establish the eventual high-temperature state. The reference pressure only selects the displayed gas curve.

## Sources and reference data

- [Jünnemann et al., PRB 111, 195304 (2025)](https://doi.org/10.1103/PhysRevB.111.195304): reference geometry, baseline absorption and Figs. 4/6. The CSVs are independent figure digitizations, not author-supplied raw data; accompanying provenance records PDF coordinates and calibration.
- [Sinclair et al., preprint and Appendix A](https://arxiv.org/pdf/1809.00237), published as [Phys. Rev. Applied 11, 044084 (2019)](https://doi.org/10.1103/PhysRevApplied.11.044084): TPA fit and Drude parameters.
- [Sturm and Reaves (1992), original PDF](https://www.princeton.edu/~sturmlab/pdfs/publications/JP.30.TED.1992.pdf), Eqs. 7–13: optical gap, bandgap absorption, thermal carrier density and empirical FCA.
- [Arora, Hauser and Roulston (1982)](https://ieeexplore.ieee.org/abstract/document/1482195), Eqs. 8/13: mobility laws; note the original hole exponent −2.23.
- [Cataldo et al., Optics Letters 37, 4200 (2012)](https://doi.org/10.1364/OL.37.004200): low-stress SiNx infrared optical fit used for the provisional SiN example.
- [Holdman et al. supplement](https://arxiv.org/src/2110.06185v1/anc/Si_Laser_Sail_Heating_Supplement.pdf): broadband silicon modeling context. Its discussion alone does not supply the complete numerical spectrum needed to validate our radiation calculation.
