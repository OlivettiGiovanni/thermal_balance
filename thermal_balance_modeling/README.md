# Thermal balance modeling

A small Python model for comparing silicon absorption mechanisms and moving to a
SiN cylinder selected from an absorption-versus-radius CSV. Python 3.10+.

```bash
cd thermal_balance_modeling
python -m pip install -r requirements.txt
python run.py --config config_si.toml --output results/si --dpi 200
python run.py --config config_sin.toml --output results/sin --dpi 200
```

Choose a **new or empty output folder**. Paths inside the TOML belong to that TOML,
so these commands also work from another directory using absolute script/config paths.
Outputs are PNG plots, numerical CSVs, a JSON summary and copies of input files.
The `examples/` folder contains completed runs, including the pressure-balance plots.
Replot without recalculating or loading the original inputs:

```bash
python plotting.py --output results/si --dpi 300
python -m unittest discover -s tests -v
```

## Silicon comparison

All four combinations are plotted: **printed / physical phonon pairing for TPA**
and **empirical / Drude FCA**. BGA uses the empirical silicon model; carriers remain
in thermal equilibrium. The corrected Sturm density is shown only as a diagnostic.
The sphere radius (130 nm), wavelength (1550 nm) and real index (3.48) are benchmark
constants in the code. Configuring intensity changes both absorption and heating.

Paper curves were independently extracted from Jünnemann Figs. 4 and 6.
Residual comparisons are saved in `summary.json`: this is a **reconstruction with
remaining differences**, not an exact reproduction. Silicon radiation is the
digitized paper curve used as a labelled cooling input, not an independent prediction.
Drude mobility above 500 K is an extrapolation. See [MODEL_NOTES.md](MODEL_NOTES.md)
for equations, constants, reference provenance and limitations.

## SiN inputs

Use columns `radius_nm,absorption_density_W_m3` (or `radius_nm,p_abs_W`). Optional
`height_nm` must agree with the config. `max_peak` selects the largest **P_abs/V**;
`first_peak` selects the first interior peak with 3% prominence. No interpolation
of the selected radius is performed. The included CSV contains just the previously
reported 585 nm point, **not a full radius sweep**.

The selected power is the constant linear heating baseline. To infer its material
coefficient, supply `intensity_integral_w_m` = ∫I_local dV in the same CSV:
**alpha0 = P_selected / ∫I_local dV**. Without it, only the power is identifiable.
Changing laser intensity requires a matching optical sweep and field moments.

Optional `nonlinear_file` columns are `temperature_k,beta_m_w,alpha_fca_m_inv`.
Nonzero FCA needs the first field moment above; nonzero TPA also needs
`intensity_squared_integral_w2_per_m` = ∫I_local² dV. These must come from the same
optical simulation. Missing nonlinear data means TPA=FCA=0, explicitly reported;
SiN BGA is zero by assumption. Silicon constants are never silently reused for SiN.

The radiation spectrum uses `wavelength_um,n,k`, optionally `temperature_k` for
multiple temperature slices covering the full requested range. Fixed n,k are used
when that column is absent. This affects radiation; linear heating stays at the CSV
baseline. The supplied Cataldo spectrum and spheroid approximation are provisional
for this cylinder; the saved dipole-size diagnostic flags their applicability.

## Temperature and pressure

Pressure is always **mbar**. Gas collision speed uses the chamber temperature;
heat flow also depends on **particle minus chamber temperature**. Evacuating the
chamber does not by itself specify its temperature. Defaults: air at 300 K,
accommodation 0.8, 401 temperature samples. These can be overridden with
`environment_temperature_k`, `accommodation`, `temperature_points` (Si bath fixed).
No pressure input is ignored: it changes `P_gas` and plot labels on each new run.

`summary.json` distinguishes a stable threshold crossing, a cold-branch fold,
vacuum equilibrium, and an unresolved upper range. A fold is **not automatically a
decomposition pressure**. Results use free-molecular gas exchange; extrapolation
to tens of mbar requires checking the gas-flow regime. The assumed SiN decomposition
temperature is configurable. Radiation/absorption uncertainty carries into thresholds.
