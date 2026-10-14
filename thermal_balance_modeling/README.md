# Thermal balance modeling

Python 3.10+. Compare silicon absorption mechanisms and temperature-dependent
heating/cooling against Jünnemann's digitized curves.

```bash
python -m pip install -r requirements.txt
python run.py --config config_si.toml --output results/si --dpi 200
python plotting.py --output results/si --dpi 300  # Replot saved numbers only
python -m unittest discover -s tests -v
```

Use a **new or empty output folder**. Input paths are relative to the TOML.
Completed examples are in `examples/results_si` and `examples/results_sin`.

## Two silicon plots

- `absorption_comparison.png`: TPA, FCA and BGA versus particle temperature.
  Paper dots; first model solid, second dashed. Sinclair is labelled
  **semi-empirical** because its phonon model uses fitted weights.
- `power_balance.png`: Jünnemann's digitized total absorption, our selected
  total absorption, gas cooling and BB radiation, all versus particle temperature.
  Our total uses **linear + physical-pairing TPA + unchanged Drude FCA + Sturm BGA**.

The pressure-versus-intensity plot is removed. Pressure diagnostics remain in
CSV/JSON. Replotting also removes the known obsolete plot filenames, including
`critical_pressure.png`, and works with the previous saved `powers.csv` files.
The internal `printed` key is retained for compatibility; its displayed name
is now `Sinclair (semi-empirical)`.

## Four main Si inputs

`material`, `intensity_w_um2`, `reference_pressure_mbar`, `temperature_max_k`.
Gas pressure is in **mbar** and directly changes the plotted gas cooling.
Defaults: 401 temperatures, 300 K gas bath, air accommodation 0.8.
The Si geometry remains R=130 nm and wavelength=1550 nm, with T_max <= 1600 K.

Fig. 6 absorption dots stay at the paper's **0.1 W/µm²**; they are never rescaled
linearly because total absorption includes TPA. Other model intensities are
explicitly labelled as a different condition. Fig. 4 TPA coefficient dots can
be rescaled via alpha_TPA=beta*I, with a note on that figure.

**Si BB is a digitized reference input, not an independent radiation model.**
Drude is not refitted; its mobility is extrapolated above 500 K. The alternative
TPA pairing retains Sinclair's original weights, also without refitting. Gas
cooling assumes free-molecular flow and a fixed gas bath temperature. Equilibrium
requires P_abs=P_gas+P_BB. Equations and limits: [MODEL_NOTES.md](MODEL_NOTES.md).

## SiN

```bash
python run.py --config config_sin.toml --output results/sin --dpi 200
```

SiN produces only `power_balance.png`, without silicon reference curves.
Absorption CSV: `radius_nm,absorption_density_W_m3` or `radius_nm,p_abs_W`.
`max_peak` selects maximum P_abs/V; `first_peak` selects the first interior peak
with 3% prominence. The supplied CSV contains only the reported 585 nm point.

The selected power anchors linear heating. Optional nonlinear CSV columns are
`temperature_k,beta_m_w,alpha_fca_m_inv`; FCA requires optical CSV field moment
`intensity_integral_w_m`=integral(I dV), and TPA additionally requires
`intensity_squared_integral_w2_per_m`=integral(I² dV). Without nonlinear data,
TPA=FCA=0; SiN BGA is zero by assumption. No Si Drude parameters are applied to SiN.

Radiation CSV: `wavelength_um,n,k`, optionally with `temperature_k` slices.
The supplied Cataldo spectrum and dipole spheroid radiation model are provisional.
