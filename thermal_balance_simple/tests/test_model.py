"""Small physical and workflow regressions; run: python -m unittest discover -s tests."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from scipy.constants import atomic_mass, c, h, k
from scipy.integrate import trapezoid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import inputs
import physics
import plotting
import run


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        (self.folder / "absorption.csv").write_text(
            "radius_nm,absorption_density_W_m3\n100,1e9\n200,4e9\n300,2e9\n400,8e9\n")
        (self.folder / "material.csv").write_text("wavelength_um,n,k\n2,2,0\n40,2,0\n")

    def config(self, pressure=0.01):
        path = self.folder / "case.toml"
        path.write_text(f'''[absorption]
file = "absorption.csv"
selection = "first_peak"
prominence_fraction = 0.03
[material]
file = "material.csv"
[geometry]
height_nm = 400
[gas]
mass_amu = 28.965
gamma = 1.4
accommodation = 0.8
[radiation]
integration_points = 100
[analysis]
environment_temperature_k = 300
reference_pressure_mbar = {pressure}
decomposition_temperature_k = 1950
temperature_max_k = 2500
temperature_points = 31
[plot]
dpi = 50
''')
        return path

    def material(self, extinction=0.1):
        wavelength = np.geomspace(2e-6, 40e-6, 100)
        return dict(wavelength_m=wavelength, temperature_k=None,
                    n=np.full((1, 100), 2.), k=np.full((1, 100), extinction))

    def test_fresh_cli_uses_selected_config_pressure_in_csv_and_plot(self):
        (self.folder / "config.toml").write_text("This is deliberately invalid TOML")
        results, summaries = [], []
        for pressure in (0.01, 1.0):
            output = self.folder / f"result_{pressure}"
            process = subprocess.run([sys.executable, str(ROOT / "run.py"), "--config",
                                      str(self.config(pressure)), "--output", str(output)],
                                     cwd=self.folder, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn(f"Reference pressure used: {pressure:g} mbar", process.stdout)
            results.append(np.genfromtxt(output / "power_balance.csv", delimiter=",", names=True))
            summaries.append(json.loads((output / "summary.json").read_text()))
            self.assertEqual(summaries[-1]["reference_pressure_mbar"], pressure)
            self.assertIn(f"reference_pressure_mbar = {pressure}", (output / "config_used.toml").read_text())
            with patch.object(plotting, "save_figure") as save:
                plotting.plot_results(output)
                title = save.call_args_list[0].args[0].axes[0].get_title()
                self.assertIn(f"Gas pressure = {pressure:g} mbar", title)
            plotting.plt.close("all")
            self.assertEqual(len(list(output.glob("*.png"))), 2)
            self.assertFalse(list(output.glob("*.pdf")))
        np.testing.assert_allclose(results[1]["p_gas_w"], 100 * results[0]["p_gas_w"], rtol=1e-14)
        self.assertEqual(summaries[0]["critical_pressure_mbar"], summaries[1]["critical_pressure_mbar"])

    def test_peak_selection_and_legacy_power_equivalence(self):
        table = inputs.read_absorption(self.folder / "absorption.csv", 400)
        arguments = table["radius_nm"], table["density_w_m3"]
        self.assertEqual(inputs.select_peak(*arguments, "first_peak", 0.03), 1)
        self.assertEqual(inputs.select_peak(*arguments, "max_peak", 0.03), 3)
        with self.assertRaisesRegex(ValueError, "No qualifying"):
            inputs.select_peak([1, 2, 3], [1, 2, 3], "first_peak", 0.03)
        legacy = self.folder / "legacy.csv"
        np.savetxt(legacy, np.c_[table["radius_nm"], table["p_abs_w"]], delimiter=",",
                   header="radius_nm,p_abs_W", comments="")
        np.testing.assert_allclose(inputs.read_absorption(legacy, 400)["density_w_m3"], arguments[1])

    def test_old_pressure_key_is_rejected(self):
        path = self.config()
        path.write_text(path.read_text().replace("reference_pressure_mbar", "reference_pressure_pa"))
        with self.assertRaises(ValueError):
            inputs.load_config(path)

    def test_critical_pressure_matches_gas_only_balance_and_direction(self):
        coefficient = physics.gas_coefficient(200, 400, 300, 28.965, 1.4, 0.8)
        area = 2 * np.pi * 200e-9 * (200e-9 + 400e-9)
        expected = 100 * 0.8 * area * np.sqrt(8 * k * 300 / (np.pi * 28.965 * atomic_mass)) / (8 * 300) * 6
        self.assertAlmostEqual(coefficient / expected, 1, places=14)
        heat = 1e-9
        pressure = physics.critical_pressure(heat, 0, coefficient, 300, 1950)
        self.assertAlmostEqual(pressure / (heat / (expected * 1650)), 1, places=14)
        self.assertGreater(heat - physics.gas_power(1950, pressure / 2, coefficient, 300), 0)
        self.assertLess(heat - physics.gas_power(1950, pressure * 2, coefficient, 300), 0)
        negative = physics.critical_pressure(heat, 2 * heat, coefficient, 300, 1950)
        self.assertLess(negative, 0)
        crossing, status, _ = run.threshold_result(negative, False, heat)
        self.assertIsNone(crossing)
        self.assertEqual(status, "no_positive_crossing")

    def test_material_interpolation_and_temperature_limits(self):
        path = self.folder / "temperature_material.csv"
        path.write_text("temperature_k,wavelength_um,n,k\n300,2,2,0.1\n300,40,2,0.1\n"
                        "2500,2,4,0.3\n2500,40,4,0.3\n")
        material = inputs.read_material(path, 100)
        n, extinction = physics.optical_indices(1400, material)
        np.testing.assert_allclose(n, 3)
        np.testing.assert_allclose(extinction, .2)
        with self.assertRaises(ValueError):
            physics.optical_indices([299, 2501], material)
        material["n"][:] = 2
        material["k"][:] = .1
        temperature = np.array([300., 500., 1950., 2500.])
        np.testing.assert_allclose(physics.radiation_power(temperature, material, 200, 400, 300),
                                   physics.radiation_power(temperature, self.material(), 200, 400, 300))

    def test_net_radiation_zero_and_near_bath_accuracy(self):
        temperature = np.array([300., 300.00000001, 1950.])
        np.testing.assert_array_equal(physics.radiation_power(temperature, self.material(0), 200, 400, 300), 0)
        material = self.material()
        power = physics.radiation_power(temperature, material, 200, 400, 300)
        self.assertEqual(power[0], 0)
        wavelength = material["wavelength_m"]
        x = h * c / (wavelength * k * 300)
        derivative = 2 * h * c**2 / wavelength**5 * x / 300 * np.exp(-x) / (-np.expm1(-x))**2
        cross = physics.spheroid_cross_section(wavelength, material["n"][0], material["k"][0], 200, 400)
        expected = 4 * np.pi * trapezoid(cross * derivative, wavelength) * (temperature[1] - 300)
        np.testing.assert_allclose(power[1], expected, rtol=1e-5, atol=0)


if __name__ == "__main__":
    unittest.main()
