# Simple thermal balance

Select one cylinder from an absorption-density sweep, calculate the gas pressure
that balances heating and cooling at an assumed decomposition temperature, and
plot **P_abs(T), P_gas(T, p_ref), and net P_BB(T)**. No temperature/pressure sweep
or equilibrium solver is needed for the present constant-heating model.

## Run

Python >=3.10. Your existing environment can be used if it has these dependencies:

```bash
python -m pip install -r requirements.txt
python run.py --config config.toml --output results/cataldo
```

Edit `config.toml` first: absorption CSV, selection, material CSV, height and
reference pressure. **Only the named TOML is loaded: no merging, hidden defaults,
or pressure overrides.** Input paths are relative to that TOML; `--output` is
relative to your shell directory. Repeat with the same output folder to replace
its results. The terminal and figure title show the pressure actually used.

All input/output pressures are **mbar**: `0.01 mbar = 1 Pa`, `10 mbar = 1000 Pa`.
Changing `reference_pressure_mbar` changes the gas curve; it does not change
the critical pressure, which is a separately calculated property of the model.
For the other materials, change only `material.file` to
`data/sin_gautam_2016.csv` or `data/sin_luke_2015.csv`, and use a new output folder.

Quick demonstration (synthetic absorption, not your simulation):

```bash
python run.py --config config_demo.toml --output results/demo
python plotting.py --results results/demo --dpi 300
```

The second command only redraws saved results. **After changing any physical
input, rerun `run.py`.** PNG is the only plot format.

## Inputs and selection

- Absorption CSV: `radius_nm,absorption_density_W_m3`. Density is absolute absorbed
  power per volume in **W/m³**, not efficiency, cross-section, or normalized intensity.
  Existing `radius_nm,p_abs_W` files also work: density is calculated using H.
  Optional `height_nm` must match H; rows with `converged=False` are excluded.
- `max_peak`: largest sampled absorption/V, including endpoints; ties use the
  smallest radius. `first_peak`: first interior local peak in increasing radius
  with prominence >= `prominence_fraction * max(absorption/V)`; no peak means an error.
- Material CSV: `wavelength_um,n,k`, held fixed with temperature, or
  `temperature_k,wavelength_um,n,k` for genuine temperature-dependent optical data.
  The latter needs >=2 temperatures covering the entire requested temperature range.
  n,k are linearly interpolated; only the common wavelength band is integrated.
  **The three bundled datasets do not contain measured temperature dependence.**

## Model and critical pressure

With cylinder volume V = πR²H and exposed area A = 2πR(R+H):

```text
P_abs(T) = sum of absorption_terms(T)     [currently: selected density × V]
P_gas(T,p) = g × p_mbar × (T − T_env)
g = accommodation × A × 100 × sqrt(8 kB T_env / (π m))
    × (gamma + 1) / [8 T_env (gamma − 1)]       [W/(mbar K)]
P_BB(T) = 4π ∫ C_abs_avg(λ,T) [Bλ(T) − Bλ(T_env)] dλ
p* = [P_abs(T_d) − P_BB(T_d)] / [g × (T_d − T_env)]
```

For the current fixed heating and fixed optical constants, **p* > 0 is p_crit**:
below it, equilibrium exceeds T_d (if a finite equilibrium exists); above it,
equilibrium stays below T_d. A negative result means **no positive critical
pressure**: radiation alone prevents the crossing. Zero means vacuum balances
exactly at T_d. T_d = 1950 K is an adjustable assumption, not a universal SiN limit.
Temperature-dependent heating or optical data can create multiple equilibria:
the program then reports only the balance pressure at T_d, with its branch
interpretation explicitly unverified. It does not claim a runaway bifurcation.

Radiation retains the previous volume/aspect-matched spheroid electric-dipole
approximation, with radiation reaction and scattering subtraction. It is **not a
full-wave cylinder calculation**. Gas cooling assumes the free-molecular regime.
Spectral coverage and the dipole approximation limit quantitative accuracy;
see `summary.json` and [data/README.md](data/README.md).

## Outputs and code

- `power_balance.png`: the three powers vs temperature, with p_ref in mbar.
- `absorption_selection.png`: absorption/V vs radius, selected sample marked.
- CSVs: plotted powers and absorption sweep/selection. `summary.json`: critical
  pressure/status, geometry, actual pressure, inputs and assumptions.
- Input CSV copies and `config_used.toml` record the calculation.

Four modules: `inputs.py` reads/validates; `physics.py` contains formulas;
`run.py` calculates/saves; `plotting.py` plots. Add future free-carrier or bandgap
heating as named terms in `physics.absorption_terms()`; `absorbed_power()` sums them.

Regression checks: `python -m unittest discover -s tests -v`.
