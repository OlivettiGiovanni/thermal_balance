# Thermal balance modeling

This program compares **laser heating** with **gas cooling** and **thermal radiation** as the particle temperature changes. All powers are in watts, temperatures in kelvin, and input pressures in **mbar**.

There are two separate calculations:

| What you want | Command to run | Geometry | Figures |
|---|---|---|---|
| Your nanoparticle | `scripts/run.py` with a material config | Cylinder, with your radius and height | One power-balance plot, including the separate absorption contributions |
| Compare the models with Jünnemann | `Junneman/compare_Junneman.py` | Fixed 130 nm-radius Si sphere | Absorption-coefficient comparison and power balance |

**The cylinder's linear absorbed power comes from your optical simulation.** This program does not rerun the cylinder optical calculation. Changing a material table does not generate a new absorption-versus-radius sweep.

The implementation is parameterized: the shared functions contain no hidden silicon parameter set. This does **not** make an approximation equally accurate for every material. In particular, the supplied SiN carrier parameters are exploratory, and the cylinder radiation calculation uses a dipole approximation. These limits are explained below.

## 1. Install and run

Python 3.10 or newer is required. The commands below assume you are in the project folder:

```bash
cd ~/Projects/Thermal_Runaway/thermal_balance/thermal_balance_modeling
```

If your `thermal_balance` environment already exists:

```bash
conda activate thermal_balance
python -m pip install -r requirements.txt
```

If it does not exist yet, create it once:

```bash
conda create -n thermal_balance python=3.11
conda activate thermal_balance
python -m pip install -r requirements.txt
```

If you are currently inside another Conda environment, `conda deactivate` exits it. You can then activate `thermal_balance`. An environment is a collection of Python packages; it does not decide which project folder you are working in.

### First run: the independent Si comparison

This calculation needs no external optical export and no config file:

```bash
python Junneman/compare_Junneman.py --output results/junneman --dpi 200
```

It creates exactly these two figures:

- `absorption_comparison.png`: TPA, FCA and bandgap absorption. Dots are digitized paper curves, labelled `TPA Junn.`, `FCA Junn.` and `BGA Junn.`; solid lines are the first formulas and dashed lines are the alternatives.
- `power_balance.png`: the nine entries below. The particle is a Si sphere with radius 130 nm; all calculated radiation curves use the same geometry and spectral integration band.

| Plot label | Style | Meaning |
|---|---|---|
| `abs_tot Junn` | Dashed | Digitized total absorption from Jünnemann |
| `BB Junn` | Dashed | Digitized radiation from Jünnemann |
| `abs_tot` | Solid | Our selected total laser absorption |
| `gas` | Solid | Gas cooling at the reference pressure |
| `BB_dipole(T_0)` | Solid | Dipole radiation with optical properties fixed at 300 K |
| `BB_dipole(T)` | Solid | Dipole radiation with the fitted temperature-dependent optical properties |
| `BB_Mie(T_0)` | Solid | Full Mie radiation with optical properties fixed at 300 K |
| `BB_Mie(T)` | Solid | Full Mie radiation with the same fitted temperature-dependent optical properties as `BB_dipole(T)` |
| `BB_Franta` | Dots | Dipole radiation calculated directly from Franta's tabulated spectra, at their supplied temperatures through 500 K |

**Fixed optical properties do not mean fixed particle temperature.** Every calculated curve uses the particle temperature on the horizontal axis in Planck's radiation law. `T_0` means only that the refractive index and extinction coefficient stay at their 300 K spectra.

`BB_Franta` contains calculated powers, not measured BB powers from Franta. The available temperatures are 300, 350, 400, 450 and 500 K. The 300 K net power is zero and is therefore invisible on a logarithmic axis. The two-parameter model fits the intermediate powers at 350, 400 and 450 K, preserving the 300 and 500 K optical endpoints. **Above 500 K it continues the fitted optical model; it does not return to room-temperature optics.** That continuation is an extrapolation, identified in the graph footnote. Details are in section 6.

The dipole and Mie curves are alternative radiation calculations: do not add them together. `BB_Mie(T)` uses exactly the same fitted n(lambda,T) and k(lambda,T), including the same extrapolation above 500 K, as `BB_dipole(T)`. It adds no fit parameters. None of these BB changes alters the laser-heating formulas or adds another FCA heating term.

The benchmark prints short progress messages while it fits and calculates the curves. Calculating Mie absorption at every temperature takes longer than calculating it once with room-temperature optics. Its saved `summary.json` reports the fitted parameters and agreement with the Franta powers.

After installing this update, rerun the benchmark command above once to calculate and save `BB_Mie(T)`. Replotting an older `results.npz` still works, but cannot add a curve that was never calculated.

The gas curve at 0.1 mbar is an **added comparison**. Jünnemann's Fig. 6 does not specify that gas curve or a reference gas pressure.

`miepython` is needed here because the benchmark is a sphere. It is **not used to replace your cylinder absorption simulation**.

### Second run: the supplied SiN example

```bash
python scripts/run.py --config config/SiN_HS.toml --output results/sin_hs --dpi 200
```

This is a working example using one previously reported optical point: radius 585 nm, height 400 nm, absorbed power `1.1317201e-9 W`. It is **not a new high-stress SiN simulation or a full radius sweep**. Its original index, absorption coefficient and intensity normalization were not supplied with that point. The example config therefore labels `n_0=2`, `simulation_alpha_m_inv=1` and `simulation_intensity_w_um2=0.1` as assumptions. Replace them with the actual simulation inputs before interpreting the result quantitatively.

The other material configs deliberately ask for their own optical exports. They will stop with a missing-file message until you supply those files. They do not silently reuse the example SiN power for another material.

### Replot without rerunning the model

```bash
python scripts/plotting.py --output results/sin_hs --dpi 300
python scripts/plotting.py --output results/junneman --dpi 300
```

These commands read the saved `results.npz`, which contains the final plotting arrays. They do not read the material tables or recalculate the physics.

The output directory and PNG resolution are command-line arguments, not config settings. Use different output folders for different conditions. Running again in the same output folder overwrites this program's named output files; it does not remove unrelated files from that folder.

## 2. What is in the project?

| Location | Purpose |
|---|---|
| `config/` | One complete config per material scenario. This is where you choose the material table, cylinder absorption export, geometry and model parameters. |
| `data/material/` | Material optical tables, with a same-name `.json` containing their sources and limitations. |
| `data/reported_SiN_point.csv` | The single historical optical point used to make the example runnable. |
| `scripts/prepare_inputs.py` | Reads the config and material tables, checks them, and prepares the wavelength and temperature arrays in memory. |
| `scripts/absorption.py` | Reads the cylinder absorption export, selects its radius, imports linear power and calculates TPA, FCA and BGA powers. |
| `scripts/cooling.py` | Calculates gas cooling and net thermal radiation. |
| `scripts/radiation_optics.py` | Evaluates the two-parameter optical extrapolation and its Drude contribution. Material parameters are passed in; the Si benchmark supplies its values and performs the fit. |
| `scripts/balance.py` | Adds and subtracts the powers and finds equilibria and pressure diagnostics. |
| `scripts/plotting.py` | Makes the figures, either directly after a calculation or from saved results. |
| `scripts/run.py` | Runs the cylinder workflow and saves its results. |
| `Junneman/` | The standalone benchmark script, paper, digitized CSVs, extraction script, source JSONs and benchmark Si spectrum. All are directly in this folder. |
| `tests/` | Checks of units, physical limits, input handling and the calculation/replot workflow. |

The normal model does not import the benchmark or select equations from a material's filename. The **script you run** determines whether you get the cylinder calculation or the Jünnemann comparison.

## 3. Run your own cylinder

1. Run your cylinder optical simulation with the material, height, wavelength, polarization and illumination arrangement you actually want.
2. Export the radius sweep, including total absorbed power or absorbed power per volume.
3. Choose the corresponding config in `config/`, or copy one to a new name.
4. In that config, set `absorption_file` to your export and `optical_file` to the material optical table.
5. Match the height, laser wavelength, room/reference index and optical-simulation normalization. Set your requested intensity and gas pressure.
6. Run `scripts/run.py` with the config and a new output folder.

For example, after supplying the crystalline-Si cylinder export:

```bash
python scripts/run.py --config config/Si_crystalline.toml --output results/si_cylinder --dpi 200
```

Every path **inside a config is relative to that config file**, not to the terminal's current directory. For a config stored in `config/`, `../data/example.csv` points to the project's `data/` folder. An absolute Linux path also works.

These two settings have different jobs:

```toml
[data]
optical_file = "../data/material/SiN_HS.csv"   # Material n and k data.
absorption_file = "/full/path/to/your/radius_sweep.csv" # Your cylinder simulation.
fallback_file = "../data/material/SiN_LS.csv"  # Optional source for missing spectral data.
```

Use a complete config from `config/`; this short snippet is only an explanation of the paths.

### Radius and height

```toml
[geometry]
radius_nm = "max_peak"
height_nm = 400.0
```

`radius_nm` accepts exactly three kinds of input:

| Value | Meaning |
|---|---|
| `"max_peak"` | Use the sampled radius with the largest **P_abs / volume**. The maximum may be at the edge of the supplied sweep. |
| `"first_peak"` | Use the first interior peak of P_abs / volume, in increasing-radius order, with prominence at least 3% of the largest density. |
| A number, for example `585.0` | Use that exact simulated radius, in nm. No nearest-radius substitution is made. |

A single-point CSV works with `max_peak` or its exact radius. It cannot contain an interior `first_peak`.

`height_nm` is always required. It must be the height used by the optical simulation. Changing it here does not resimulate the optical field. The selected radius and height are used consistently for cylinder volume, gas area and the radiation approximation.

### Optical export format

The simplest supported export is:

```csv
radius_nm,height_nm,p_abs_W
585,400,1.1317201e-9
```

Alternatively, use `absorption_density_W_m3` instead of `p_abs_W`. Volume is `pi * R**2 * H`, with R and H converted to metres. If both power columns are supplied, they must agree. `height_nm` in the CSV is optional but recommended because it allows an explicit consistency check.

Optional field-integral columns are:

| CSV column | Meaning | Units |
|---|---|---|
| `intensity_integral_w_m` | Integral of the local internal optical intensity over the particle volume | W m |
| `intensity_squared_integral_w2_per_m` | Integral of the square of that local intensity over the particle volume | W²/m |

**TPA requires the second integral.** It cannot be recovered from total linear power alone. This is why the Si cylinder config needs more optical information than the supplied SiN example, whose TPA is zero.

The local intensity convention is `I_local = n epsilon_0 c |E|² / 2` for a complex peak-amplitude electric field. It is the positive intensity-like quantity used in local absorption, not the signed net Poynting flux of a standing wave. Use compatible field amplitudes and SI units in the optical export.

The code can check additional simulation metadata when you supply it in numeric CSV columns or in a same-stem JSON file. Recognized metadata include `height_nm`, `wavelength_nm`, `n_0`, `simulation_alpha_m_inv` and `simulation_intensity_w_um2`. A mismatch stops the calculation. Missing metadata is reported as unverified; the program cannot reconstruct unrecorded simulation conditions.

### Intensity and the linear normalization

In `[linear]`:

- `simulation_alpha_m_inv` is the homogeneous linear absorption coefficient used to generate the optical export, in 1/m. It must be positive.
- `simulation_intensity_w_um2` is the incident intensity used by that simulation, in W/µm².

In `[experiment]`, `intensity_w_um2` is the intensity you want to analyze now. The program rescales the fixed optical field under the assumption of linear optics. Use the **same incident-intensity convention** as the optical simulation. No extra standing-wave factor is applied.

Changing `simulation_alpha_m_inv` does not change the imported linear power. It changes the inferred internal-field normalization. This parameter describes the simulation you already ran; it is not a free control for adjusting the absorption of a new material.

## 4. Material tables and missing data

Each material has one optical CSV with four columns:

```csv
temperature_k,wavelength_um,n,k
```

Here `n` is the real refractive index and `k` is the dimensionless extinction coefficient. **This k is not Boltzmann's constant.** The equivalent absorption coefficient is `alpha = 4*pi*k/lambda`, with wavelength in metres.

Several rows at the reference temperature describe n(lambda) and k(lambda). Rows at other temperatures provide temperature dependence. At the laser wavelength, the program extracts n(T) and k(T) from those rows. This structure stores all four dependencies in one material table without requiring four unrelated files.

| Table | Wavelength information | Temperature information bundled | Main source and qualification |
|---|---|---|---|
| `Si_crystalline.csv` | n and k over approximately 0.031–310 µm | Spectra at 300, 350, 400, 450 and 500 K | Franta fitted experimental data for float-zone crystalline Si. The long-wavelength table extension exceeds the measured band of the 2017 paper. |
| `Si_polycrystalline.csv` | No verified values bundled | None | Explicit unfilled template. It cannot be used for a calculation until populated with suitable data. |
| `SiN_LS.csv` | Cataldo fitted n and k, 1–666.7 µm | Borrowed Johnson n(T) increments at 1550 nm, 300–460 K; no measured hot k | Low-stress amorphous LPCVD SiN. Near-IR fitted loss is not a replacement for your own laser absorption simulation. |
| `SiN_HS.csv` | Eight sparse near-IR k points, 0.532–1.55 µm; one reference n point | Borrowed Johnson n(T) increments, 300–460 K; no measured hot k | Land high-stress LPCVD samples. Their annealing history is not established. Missing BB spectrum is explicitly borrowed from LS. |
| `SiN_HS_annealed.csv` | Luke n fit, 0.310–5.504 µm, extended by digitized fitted curves to 31.6 µm; resolved k mainly above 5.1 µm, with gaps | Borrowed Johnson n(T) increments, 300–460 K; no measured hot k | Luke annealed LPCVD film. Annealing is documented; its numerical residual stress is not established. Missing spectrum is borrowed from LS. |

The names LS, HS and HS_annealed distinguish the intended material categories. They do not certify that every cited film exactly matches your particle. All these SiN film datasets are amorphous; no sputtered SiN or TiO2 is included.

The same-name JSON files explain where values came from, which entries are fitted or digitized, and what was borrowed. They are provenance records, not extra configuration files to edit for each run.

### Rules for blanks and temperature ranges

- A blank means **unknown**, not zero absorption.
- Interpolation is between adjacent supported entries; an explicit blank breaks interpolation across that interval.
- Missing n(T), or a temperature outside its supplied range, uses the configured reference value `n_0`. It does not extrapolate a low-temperature thermo-optic fit to 1950 K.
- Available n(T) is used as an **increment** relative to its table's reference value, added to `n_0`. Thus the room index remains consistent with your optical simulation.
- Missing k(T) falls back to its reference-temperature value when available. A mask records where hot data actually exists, so fallback values are not advertised as measured high-temperature data.
- For radiation, missing n(lambda) or k(lambda) is filled only from the explicitly named `fallback_file`, at the same wavelength. If the requested band is still incomplete, the calculation stops. It does not extend a single laser-wavelength value over the infrared.

The normal **cylinder** radiation calculation uses **n(lambda,T) and k(lambda,T)** wherever the primary table supplies a supported spectral interval at the particle temperature. It interpolates each property separately between adjacent wavelength and temperature entries. A blank or a temperature outside coverage uses that property's reference-temperature spectrum. Missing room-temperature spectral values still require the explicitly declared fallback material. The separate Jünnemann benchmark uses the explicitly fitted extrapolation described in section 6 instead.

An isolated n(T) measurement at only 1550 nm is not enough to define a hot radiation spectrum: it is used for the laser coefficient, while radiation retains the room spectrum there. The existing SiN tables therefore retain room-temperature BB optics, with a small footnote. Available laser n(T) affects the Drude coefficient; it does not rerun the resonant optical simulation.

In the normal cylinder workflow, returning to room optical values outside their tabulated range can produce a step. This is a data-coverage boundary, not a physical discontinuity. The BB line is broken at such boundaries. Equilibrium searches do not interpolate across optical fallback jumps; continuous cold-branch pressure diagnostics are reported as unresolved when those jumps are present. A balance pressure at a covered individual temperature can still be reported. A table containing only a room-temperature spectrum simply gives fixed optical properties throughout; it requires no temperature fit. The Si benchmark's fit is not automatically applied to SiN or another material.

If the spectrum or the temperature-dependent optical data uses a borrowed reference, the figure shows `* other reference used`. Consult `summary.json` and the input JSON for the specific sources.

## 5. How absorbed power is calculated

The modeled heating is:

$$P_{abs}=P_{linear}+P_{TPA}+P_{FCA}+P_{BGA}.$$

Let `P_sim` be the selected linear power in the optical export and let `s = I / I_sim`. For a spatially uniform simulated linear coefficient:

$$M_1=\int_V I_{local}\,dV=\frac{P_{sim}}{\alpha_{sim}},\qquad
M_2=\int_V I_{local}^2\,dV.$$

The separate terms are:

$$P_{linear}=sP_{sim},\quad P_{TPA}=\beta(T)s^2M_2,\quad
P_{FCA}=\alpha_{FCA}(T)sM_1,\quad P_{BGA}=\alpha_{BGA}(T)sM_1.$$

There is no extra linear term reconstructed from the material table. This avoids counting the imported linear absorption twice.

These equations assume that the **local internal intensity distribution stays fixed apart from intensity scaling**. They do not move or broaden resonances as the material heats. If additional loss or index changes strongly alter the internal field, a new optical simulation or a coupled optical/thermal calculation is needed. The code does not claim that it has done that calculation.

### Linear absorption

Linear power is constant versus particle temperature in this analytical model, because it is the imported reference optical power. Different materials need different matching optical exports. Merely switching `optical_file` while keeping the same absorption export will not create a different linear heating curve.

### TPA: two-photon absorption

Two photons together provide the energy for an electronic transition. The production code uses the physical phonon pairing discussed in the comparison. Define photon energy `E = h*c/lambda`, phonon energy `E_ph,j = k_B*theta_j`, and phonon occupation `N_j = 1/(exp(theta_j/T)-1)`.

$$\beta(T)=E_g(T)^{3/2}\sum_j w_j\left[
(N_j+1)[2E-E_g(T)-E_{ph,j}]_+^2+
N_j[2E-E_g(T)+E_{ph,j}]_+^2\right].$$

`[x]_+` means `max(x,0)`: an energetically forbidden contribution is zero. The `N+1` factor belongs to phonon emission, and `N` to phonon absorption. Energy quantities in these fits are in eV.

| TPA config entry | Meaning |
|---|---|
| `weights` | Strength of each phonon-assisted branch, in m/(W eV^(7/2)). |
| `phonon_temperatures_k` | Phonon energy expressed as a temperature: theta = E_ph/k_B. |
| `gap0_ev` | Optical-gap fit at zero temperature, in eV. |
| `gap_slope_ev_k`, `gap_temperature_k` | Parameters a and b in `E_g(T) = E_g(0) - a*T²/(T+b)`. |

The Si weights are retained from the source fit without refitting the changed pairing. This is a phonon-assisted model, not a universal description of every direct-gap or amorphous material. In the supplied SiN configs, `weights = [0.0]` makes beta exactly zero. No unused phonon parameters are required for that disabled term.

### FCA: free-carrier absorption

A carrier absorbs optical energy while its motion is damped by scattering. Drude describes the response of carriers that are already present. A separate density formula specifies how many carriers the model assumes:

$$N_s(T)=A_sT^{3/2}\exp\left(-\frac{E_{a,s}}{k_BT}\right),\qquad
\tau_s(T)=\sum_j\tau_{0,js}(T/T_0)^{p_{js}},\qquad \Gamma_s=1/\tau_s.$$

The sum over species s includes electrons and holes. The weak-loss Drude absorption, retaining finite damping, is:

$$\alpha_{FCA}(T)=\sum_s\frac{N_s e^2\Gamma_s}
{\epsilon_0 c\,n(T)\,m_s\,[\omega^2+\Gamma_s^2]},\qquad \omega=2\pi c/\lambda.$$

| FCA config entry | Physical meaning |
|---|---|
| `conductivity_masses_m0` | Electron and hole masses used in the optical response, divided by the free-electron mass. |
| `density_prefactors` | A for electrons and holes, in m^-3 K^-3/2. |
| `activation_ev` | Activation energy for each carrier density, in eV. Larger values suppress thermally populated carriers. |
| `relaxation_times_s` | Reference scattering-time terms at T_0, in seconds. Each row is one term; each row's columns are electron, hole. |
| `temperature_exponents` | Exponents p for those same time terms. A negative exponent shortens the scattering time as T rises. |

The lists always use **electron first, hole second**. Two relaxation-time rows for Si reproduce the two terms of its chosen mobility law. SiN uses one row. A density-of-states mass, used to derive A, need not equal a Drude conductivity mass.

For Si, the density is the activated fit used in the reference comparison, and scattering times come from the Arora mobility law. The mobility is extrapolated above its approximately 250–500 K fitted range. Drude is **not fitted to the digitized Jünnemann FCA curve**; disagreement remains visible.

For SiN, the example uses a constant assumed 4.5 eV gap, activation energies 2.25 eV, and exploratory carrier/mass/scattering parameters informed by a photoexcited-film study. Transferring them to equilibrium thermal carriers in LS, HS or annealed particles is an assumption. The code does calculate a nonzero FCA estimate; it is not a validated quantitative prediction for those films.

There is no dopant-ionization, trap-occupation, carrier-lifetime, recombination or photogenerated-carrier solver. There is also no `intrinsic`/`extrinsic` selector suggesting such a capability. To add those mechanisms later, change the density function explicitly.

### BGA: bandgap absorption

The current term is a parameterized indirect-edge expression based on the Si reference:

$$\alpha_{BGA}=\sum_j[(N_j+1)F_j(E-E_g-E_{ph,j})+N_jF_j(E-E_g+E_{ph,j})],$$

$$F_j(x)=a_j\sqrt{[x]_+}+b_j[x-\Delta]_+^2.$$

`sqrt_weights` gives a_j in m^-1 eV^-1/2; `quadratic_weights` gives b_j in m^-1 eV^-2; `quadratic_offset_ev` is Delta. The gap parameters have the same meaning as for TPA, but the two source fits use different numerical values.

These coefficients can be supplied for another material only when this form is appropriate and supported. Changing coefficients alone does not turn an indirect crystalline-edge fit into a general amorphous-band or defect-absorption theory. In all supplied SiN configs, both weight lists are zero, so BGA is disabled by the agreed approximation. This does not remove the imported linear absorption.

### The optional k(T) line

When the primary material table provides positive reference k and covered hot k values, and TPA is disabled, the program can plot a separate estimate:

$$P_{k(T)}=P_{linear}\,k(T)/k(T_0).$$

It is labelled `k(T) estimate`. It is not added to the model and is not described as a measurement of total absorption at the trapping intensity. Ordinary low-intensity k(T) data does not automatically include intensity-dependent TPA. The supplied SiN tables do not contain measured hot k, so they do not produce this line. The standard Si calculation has TPA enabled and uses the Jünnemann comparison instead.

## 6. Cooling

### Gas cooling

The model uses free-molecular gas cooling: gas molecules strike the particle and carry away energy. It assumes the mean free path is large compared with the particle. It does not interpolate to continuum heat transfer at arbitrary high pressure.

$$P_{gas}=G\,p_{mbar}(T-T_{bath}),$$

$$G=\frac{100\,a_g A}{8T_{bath}}\sqrt{\frac{8k_BT_{bath}}{\pi m_g}}
\frac{\gamma+1}{\gamma-1}.$$

The factor 100 converts mbar to Pa. A is the cylinder surface area including both end faces. `mass_amu` is the gas molecular mass, `gamma` is Cp/Cv, and `accommodation` is the fraction describing energy exchange at the surface.

The horizontal plot axis is **particle temperature**. The chamber temperature is the fixed `bath_temperature_k`. The chamber temperature controls molecular speed; the particle-to-chamber temperature difference controls heat transfer. Pumping down changes pressure. It does not give the program a chamber-temperature history. If you measure a different chamber temperature, enter it in the config.

Changing `reference_pressure_mbar` directly scales the gas curve. Both gas cooling and net radiation are zero when particle and bath temperatures are equal.

### Thermal radiation, labelled BB

The emitted-minus-absorbed radiative power is calculated spectrally:

$$P_{BB}=4\pi\int C_{abs}(\lambda,T)
[B_\lambda(T)-B_\lambda(T_{bath})]\,d\lambda.$$

B_lambda is Planck spectral radiance; C_abs is the absorption cross section for thermal radiation. It is not a greybody area with an arbitrary constant emissivity. Its wavelength and available temperature dependence come from the material n and k table or the explicit benchmark extrapolation below. The particle-temperature cross section multiplies both Planck terms: the bath temperature does not select the optical constants. The normal cylinder workflow adds no extra Drude or BGA correction to tabulated thermal-radiation k.

For cylinders, the current compact approximation uses the actual cylinder volume and the electric-dipole factors of a spheroid with the same aspect ratio H/(2R), averaged over orientation and including a radiation-reaction correction. **This is not an exact finite-cylinder optical solver.** It can miss resonances and higher multipoles when the particle is not sufficiently small compared with the relevant wavelengths. High-index Si in the near infrared especially needs care.

`wavelength_range_um` chooses the integration band. Extending the band requires covered optical data. A result uses only that finite interval; completeness of the table does not itself prove that the omitted spectral tails are negligible. SiN's short-wavelength high-temperature emission is particularly sensitive to missing or borrowed loss data.

### The Si benchmark's two-parameter optical extrapolation

The benchmark uses the **0.4–100 µm** band and the five spectra in `Junneman/si_franta_300_500k.csv` (300, 350, 400, 450 and 500 K). These are author-tabulated fitted optical spectra; the adjacent JSON records their source and limitations. The benchmark's laser index and absorption formulas stay unchanged. This fit concerns **thermal radiation only**.

First convert Franta's optical constants to the complex relative permittivity:

$$\varepsilon_F(\lambda,T)=[n_F(\lambda,T)+ik_F(\lambda,T)]^2.$$

Write `D(lambda,T)` for the Drude dielectric contribution calculated with fixed model parameters for carrier density, effective masses and scattering. The carrier density and scattering still vary with temperature; their coefficients are not fitted here.

These BB-only inputs are listed in `BB_CARRIERS` in `Junneman/compare_Junneman.py`: intrinsic equilibrium carriers (zero net doping), DOS masses 1.08 and 0.81 times the free-electron mass, and the Varshni gap `1.166 - 4.73e-4*T**2/(T+636)` in eV. The optical masses and relaxation-time coefficients retain the existing benchmark values. Intrinsic material and the high-temperature continuation of the scattering law are approximations. The laser-heating carrier law is unchanged.

Remove the Drude change between the two endpoint spectra:

$$R(\lambda)=\varepsilon_F(\lambda,500)-\varepsilon_F(\lambda,300)
-[D(\lambda,500)-D(\lambda,300)].$$

The extrapolated optical model is:

$$\varepsilon(\lambda,T)=\varepsilon_F(\lambda,300)+g(T;p,\Theta)R(\lambda)
+D(\lambda,T)-D(\lambda,300),$$

$$g(T;p,\Theta)=\frac{f(T;p,\Theta)-1}{f(500;p,\Theta)-1},\qquad
f(T;p,\Theta)=\left(\frac{T}{300}\right)^p
\exp\left[\Theta\left(\frac{1}{300}-\frac{1}{T}\right)\right].$$

There are **exactly two adjustable parameters**: the dimensionless exponent `p` and the empirical temperature scale `Theta`, in kelvin. `Theta` is not an independently identified band gap or phonon energy. The wavelength dependence comes from the input spectra. The fixed Drude parameters are not fitted to the BB points or to Jünnemann's heating curve.

The model reproduces both endpoint optical spectra by construction. The optimizer compares **dipole BB power at the actual intermediate temperatures 350, 400 and 450 K**, minimizing squared logarithmic power differences. It does not create extra fit targets by interpolating between those temperatures. After fitting, it evaluates the same formula on the full plotting temperature grid, including temperatures above 500 K. Optical constants are recovered from the passive square root of epsilon. Negative absorption is not repaired by silently clipping it.

With the supplied Si data and fixed carrier parameters, the fit gives approximately `p = 2.97` and `Theta = 0 K`, with a maximum error of about **1.6%** at the intermediate BB points. Both parameters are allowed to vary, but the optimum reaches the zero lower bound for `Theta`; the extra activated curvature is not used by this fit. The exact values, bounds and individual errors are saved in `summary.json` under `radiation.fit`.

This is a deliberately compact approximation: all non-carrier spectral changes share one temperature factor. Matching integrated BB power does not guarantee that every wavelength is reproduced accurately. Franta's five spectra come from a fitted optical model, so agreement with these points is not independent experimental validation. **The continuation above 500 K is unvalidated extrapolation.**

The 0.4 µm lower limit is intentional. Extending this simple extrapolation farther into the ultraviolet produces nonphysical negative absorption at high temperatures. The code therefore evaluates and checks the model only over the stated BB band; it does not claim to supply a valid ultraviolet optical model. Removing the previous 0.3–0.4 µm interval changes the dipole power by less than 0.01% at 1600 K when checked with either endpoint spectrum. This checks the size of that spectral tail for the supplied optics, not the accuracy of unknown hot ultraviolet properties. All calculated curves and Franta dots use the same new band.

`BB_dipole(T_0)` and `BB_dipole(T)` both use the shared electric-dipole calculation with the sphere's volume `4*pi*R**3/3` and depolarization factors `[1/3, 1/3, 1/3]`. It includes radiation reaction and excludes elastic scattering. The Mie curves use full sphere absorption, including electric and magnetic responses and higher multipoles. `BB_Mie(T_0)` keeps the optics fixed at 300 K; `BB_Mie(T)` evaluates the same fitted spectra used by `BB_dipole(T)` at each particle temperature. There is no second optical fit for Mie, and the plot includes no fitted scale factor for either radiation calculation. The uncertainty of the optical extrapolation above 500 K therefore applies to both hot-optics curves.

The paper states **dipolar scattering theory** for its radiation curve. The digitized `BB Junn` dashed line is a separate reference, not another cooling term. Our dipole curve is not claimed to reproduce the paper's unverified dielectric inputs exactly.

The extrapolation function is reusable with other supplied material parameters and spectra, but it is only called by the Si benchmark in this version. SiN and future materials with only room-temperature spectra continue to use the simpler fixed-optics calculation; no Si-specific BB fit is imposed on them.

## 7. Equilibrium and pressure

The net heating is `P_net = P_abs - P_gas - P_BB`. A zero gives an equilibrium. It is stable if a small increase in particle temperature makes net heating negative: `dP_net/dT < 0`.

`summary.json` reports equilibria only inside the calculated temperature range. No reported root does not mean a root has been excluded at all higher temperatures.

Two pressure questions are kept distinct:

1. **Pressure that balances the powers at your chosen temperature limit.** If `decomposition_temperature_k` is supplied, the program evaluates `(P_abs - P_BB) / [G*(T_limit-T_bath)]`. A negative value means BB already exceeds heating at that temperature even in vacuum; it is not a negative physical gas pressure.
2. **Pressure at which the cold stable equilibrium branch disappears.** As pressure is reduced, this is a turning point of the balance-pressure curve, if one is found before the first vacuum equilibrium. It is reported as `cold_branch_loss` with one pressure and one temperature, not as a new plot.

These quantities need not be the same. A branch loss does not by itself prove divergence to infinite temperature or destruction: another hotter stable state may exist. Likewise, a chosen balance temperature can lie on an unstable branch. The report includes whether the chosen limit lies on the computed cold stable branch.

The SiN value 1950 K is an assumed analysis threshold. The program does not simulate chemical decomposition, melting, heat capacity or a time-dependent heating trajectory.

## 8. Inspect arrays after a run

Use `-i` to keep Python open after the calculation:

```bash
python -i scripts/run.py --config config/SiN_HS.toml --output results/sin_hs --dpi 200
```

At the Python `>>>` prompt, try:

```python
prepared.keys()
prepared["temperature_k"]
prepared["wavelength_m"]
prepared["n_spectrum"]
prepared["k_spectrum"]       # Reference spectrum, one value per wavelength
prepared["n_radiation"]      # n(T, lambda), one row per particle temperature
prepared["k_radiation"]
prepared["n_room_fallback"]  # True where room n replaces missing hot data
prepared["k_room_fallback"]
prepared["radiation_footnote"]
prepared["n_laser"]
prepared["k_laser"]
prepared["k_measured"]
prepared["particle"]
prepared["coefficients"]
prepared["radiation_cross_section_m2"]
results["p_linear_w"]
results["p_fca_w"]
```

These are ordinary Python variables containing NumPy arrays and dictionaries. The prepared optical arrays are **not exported to CSV or TXT**. They cease to exist when you close that Python process. Type `exit()` or press Ctrl+D to return to Bash.

For the benchmark, use `python -i Junneman/compare_Junneman.py --output results/junneman`. `prepared["radiation_n"]` and `prepared["radiation_k"]` contain the fitted/extrapolated spectra at every plotted temperature. `prepared["radiation_optics"]` contains the native Franta spectra used for fitting; `prepared["franta_temperature_k"]` gives their temperatures. Inspect `prepared["radiation_fit"]` for the fit report. The corresponding hot-optics absorption cross sections are in `prepared["radiation_dipole_cross_section_m2"]` and `prepared["radiation_mie_cross_section_m2"]`. The optical arrays and cross sections remain in memory; only final plotting quantities and annotations are saved. In the saved results, `p_bb_dipole_w` and `p_bb_mie_w` contain the respective hot-optics powers.

The final power arrays are saved for replotting. To inspect them in a later Python session:

```python
import numpy as np
with np.load("results/sin_hs/results.npz", allow_pickle=False) as file:
    results = dict(file)
print(results["temperature_k"])
print(results["p_abs_w"])
```

A cylinder run saves `power_balance.png`, `results.npz`, `summary.json` and `config_used.toml`. The config copy records your input text; its relative paths still refer to the original config location. The report also records the resolved input paths. The benchmark saves two PNGs, `results.npz` and `summary.json`.

Log plots omit zero contributions. For the cylinder figure, values more than seven decades below maximum modeled absorption are outside the displayed vertical range; their full values remain in the arrays. This avoids a tiny room-temperature FCA value stretching the plot over dozens of decades.

## 9. Checks and common errors

Run the included checks with:

```bash
python -m unittest discover -s tests -v
```

| Message or symptom | What to do |
|---|---|
| Missing cylinder absorption file | Set `absorption_file` to the export from your optical calculation. Only the historical SiN example is bundled. |
| No first peak | Supply a real sweep, use `max_peak`, or specify an exact simulated radius. |
| Requested radius absent | Choose a radius in the CSV or rerun the optical simulation at the desired radius. |
| Missing squared-intensity integral | Export `intensity_squared_integral_w2_per_m` for TPA. Total linear P_abs is insufficient. |
| Geometry or metadata mismatch | Correct the config or use the matching optical export; do not relabel another simulation. |
| Missing radiation spectrum | Provide a suitable explicit fallback or choose a covered band and assess its truncation. |
| Same linear curve after changing the material table | Linear heating comes from `absorption_file`; you need a matching new optical simulation for the new material. |
| FCA is small or invisible at low T | The thermally activated carrier density can be extremely small. Inspect `prepared["coefficients"]` and the plot's display range. |
| Module not found | Activate the intended environment and run `python -m pip install -r requirements.txt`. |

The benchmark CSVs are digitized **modeled curves from the paper**, not raw absorption measurements. To regenerate them from the archived PDF coordinates:

```bash
python Junneman/extract.py --output results/redigitized
```

The extraction script also supports reading the bundled PDF directly. That optional route needs PyMuPDF; it is not required for normal calculations:

```bash
python -m pip install pymupdf
python Junneman/extract.py --pdf Junneman/Junnemann2025.pdf --output results/redigitized_pdf
```

The benchmark constants and reference formulas are all visible in `compare_Junneman.py`. It uses 0.1 W/µm², 1550 nm, radius 130 nm and 300–1600 K. The TPA/BGA reference fits use their quoted photon energy 0.797 eV; the laser-heating Drude and Mie calculations use 1550 nm. This small source-fit energy approximation is explicit. The cylinder model uses the exact photon energy from its configured wavelength.

## 10. Install this update over your existing project

`thermal_balance_bb_update.zip` contains **only new or changed project files**, inside a top-level `thermal_balance_modeling/` folder. It requires the existing project and its data. **Do not empty or delete your project folder.** Your material configs, optical data, environment, Git history and previous results are not removed by these commands.

Run this block in Bash/WSL after downloading the update:

```bash
archive="/mnt/c/Users/golivetti/Downloads/thermal_balance_bb_update.zip"
unzip -t "$archive"
unzip -o "$archive" -d "$HOME/Projects/Thermal_Runaway/thermal_balance"

cd ~/Projects/Thermal_Runaway/thermal_balance/thermal_balance_modeling
conda activate thermal_balance
python Junneman/compare_Junneman.py --output results/junneman_updated --dpi 200
```

`unzip -o` copies the files into place and overwrites matching filenames without asking for each one. Extract into the **parent folder** `thermal_balance`, as shown; the archive already contains `thermal_balance_modeling/`. No old source file needs to be deleted for this update. If you have edited one of the files included in the update, commit your edits first so Git can recover or compare them.

Replot the newly calculated results later with:

```bash
python scripts/plotting.py --output results/junneman_updated --dpi 300
```

Replotting an old `results.npz` cannot create the new optical fit. Run the benchmark once after updating, then use the plotting command to change the PNG resolution without repeating the calculation.

## References

- Jünnemann et al., *Optical absorption of subwavelength silicon nanoparticles at telecom wavelengths*, Physical Review B 111, 195304 (2025): [10.1103/PhysRevB.111.195304](https://doi.org/10.1103/PhysRevB.111.195304). Benchmark Fig. 4 absorption coefficients and Fig. 6 powers; supplied paper and extraction provenance are in `Junneman/`.
- Sinclair et al., Physical Review Applied 11, 044084 (2019): [10.1103/PhysRevApplied.11.044084](https://doi.org/10.1103/PhysRevApplied.11.044084). Si TPA fit; its experimental temperature range is much lower than the full plotted range. The high-temperature continuation is an extrapolation.
- Sturm and Reaves, IEEE Transactions on Electron Devices 39, 81–87 (1992): Si free-carrier and band-edge absorption references used in the comparison. The empirical FCA curve and generalized Drude curve remain separate.
- Arora, Hauser and Roulston, IEEE Transactions on Electron Devices 29, 292–295 (1982): Si mobility law used to obtain the Drude scattering times.
- [LIGO T1500619](https://dcc.ligo.org/public/0123/T1500619/001/CarrierDensityIntrinsicSi.pdf): explicit Si DOS masses and Varshni inputs used for BB equilibrium carriers; background discussion in Green (1990), [10.1063/1.345414](https://doi.org/10.1063/1.345414).
- Franta et al., crystalline Si optical dispersion fits: [10.1016/j.apsusc.2017.02.021](https://doi.org/10.1016/j.apsusc.2017.02.021) and [10.1063/1.5026195](https://doi.org/10.1063/1.5026195). Bundled author-tabulated fitted spectra obtained through the refractiveindex.info database; full URLs are in the material JSON.
- Cataldo et al., low-stress SiN infrared dielectric model: [10.1364/OL.37.004200](https://doi.org/10.1364/OL.37.004200).
- Land et al., high-stress SiN absorption: [10.1021/acs.nanolett.4c00737](https://doi.org/10.1021/acs.nanolett.4c00737). Sparse room-temperature extinction values, not a full infrared spectrum.
- Luke et al., annealed LPCVD SiN optical data: [10.1364/OL.40.004823](https://doi.org/10.1364/OL.40.004823). The 1200 °C value is the prior anneal temperature, not the optical measurement temperature.
- Johnson et al., LPCVD SiN thermo-optic coefficient: [10.1364/OE.477102](https://doi.org/10.1364/OE.477102). The transferred n(T) increment is not a matched measurement for all three SiN stress/annealing categories.
- Mincigrucci et al., photoexcited SiN carrier response: [10.1364/OE.26.011877](https://doi.org/10.1364/OE.26.011877). Source informing the exploratory SiN FCA parameters; not validation of the assumed thermal carrier population.
