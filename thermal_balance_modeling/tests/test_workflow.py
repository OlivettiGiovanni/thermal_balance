"""Selection, calibration, branch tracing and complete command-line workflows."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inputs import load_config, select_absorption
from run import calculate, calibrated_heating, pressure_analysis


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="thermal_workflow_")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def write_csv(self, name, columns, rows):
        path = self.folder / name
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)
        return path

    def fixture(self):
        absorption = self.write_csv("absorption.csv", ["radius_nm", "height_nm", "p_abs_W"],
                                    [(585., 400., 1.1317201e-9)])
        spectrum = self.write_csv("spectrum.csv", ["wavelength_um", "n", "k"],
                                  [(wave, 2.0, 0.03) for wave in (0.5, 2, 5, 10, 20, 50, 100)])
        config = self.folder / "config.toml"
        config.write_text('material = "SiN"\nabsorption_file = "absorption.csv"\n'
                          'spectrum_file = "spectrum.csv"\nheight_nm = 400.0\n'
                          'temperature_points = 61\nreference_pressure_mbar = 0.01\n')
        return config, absorption, spectrum

    def test_selection_maximizes_density_not_total_power(self):
        radius = np.array([100., 200., 300., 400., 500.])
        density = 1e10 * np.array([1., 5., 2., 4., 1.])
        power = density * np.pi * (radius * 1e-9)**2 * 400e-9
        path = self.write_csv("sweep.csv", ["radius_nm", "p_abs_W"], zip(radius, power))
        selected, table = select_absorption(path, 400., "max_peak")
        self.assertEqual(selected["radius_nm"], 200.)
        self.assertEqual(table["radius_nm"][np.argmax(table["p_abs_w"])], 400.)
        self.assertAlmostEqual(selected["p_abs_w"] / power[1], 1., places=12)

    def test_first_peak_is_distinct_from_global_maximum(self):
        path = self.write_csv("sweep.csv", ["radius_nm", "absorption_density_W_m3"],
                              zip([100., 200., 300., 400., 500.], [1., 5., 2., 8., 1.]))
        first, _ = select_absorption(path, 400., "first_peak")
        largest, _ = select_absorption(path, 400., "max_peak")
        self.assertEqual(first["radius_nm"], 200.)
        self.assertEqual(largest["radius_nm"], 400.)

    def test_nonlinear_heating_uses_correct_field_moments(self):
        path = self.write_csv("nonlinear.csv", ["temperature_k", "beta_m_w", "alpha_fca_m_inv"],
                              [(300., 1e-11, 2.), (700., 3e-11, 4.)])
        selected = {"p_abs_w": 2e-9, "intensity_integral_w_m": 4e-10,
                    "intensity_squared_integral_w2_per_m": 4.}
        terms, calibration = calibrated_heating(selected, np.array([300., 500., 700.]), path)
        np.testing.assert_allclose(terms["p_linear_w"], 2e-9)
        np.testing.assert_allclose(terms["p_tpa_w"], [4e-11, 8e-11, 12e-11], rtol=1e-12, atol=0)
        np.testing.assert_allclose(terms["p_fca_w"], [8e-10, 12e-10, 16e-10], rtol=1e-12, atol=0)
        self.assertAlmostEqual(calibration["alpha0_m_inv"], 5.)
        with self.assertRaisesRegex(ValueError, "intensity_integral_w_m"):
            calibrated_heating({"p_abs_w": 2e-9}, np.array([300., 700.]), path)
        with self.assertRaisesRegex(ValueError, "intensity_squared_integral"):
            calibrated_heating({"p_abs_w": 2e-9, "intensity_integral_w_m": 4e-10},
                               np.array([300., 700.]), path)

    def test_no_nonlinear_data_needs_no_field_moments(self):
        terms, calibration = calibrated_heating({"p_abs_w": 2e-9}, np.array([300., 700.]), "")
        self.assertIsNone(calibration["alpha0_m_inv"])
        np.testing.assert_array_equal(terms["p_tpa_w"], 0.)
        np.testing.assert_array_equal(terms["p_fca_w"], 0.)

    def test_vacuum_equilibrium_arbitrarily_close_to_bath(self):
        temperature = np.linspace(300., 1600., 401)
        result = pressure_analysis(temperature, np.full_like(temperature, 1e-12),
                                   1e-8 * (temperature - 300.), 1e-10, 300.)
        self.assertEqual(result["status"], "cold_branch_reaches_vacuum_equilibrium")
        self.assertAlmostEqual(result["vacuum_temperature_k"], 300.0001, places=8)
        self.assertIsNone(result["critical_pressure_mbar"])

    def test_analytic_fold_is_not_a_decomposition_pressure(self):
        temperature = np.linspace(300., 1000., 701)
        # P=A+B*dT^2 gives p=A/(g*dT)+B*dT/g: minimum at dT=sqrt(A/B)=100 K.
        absorbed = 1e-9 + 1e-13 * (temperature - 300.)**2
        result = pressure_analysis(temperature, absorbed, np.zeros_like(temperature),
                                   1e-10, 300., limit=900.)
        self.assertTrue(result["status"].startswith("cold_branch_fold"))
        self.assertAlmostEqual(result["fold_temperature_k"], 400., delta=0.05)
        self.assertAlmostEqual(result["fold_pressure_mbar"], 0.2, places=5)
        self.assertIsNone(result["critical_pressure_mbar"])

    def test_stable_temperature_threshold_crossing(self):
        temperature = np.linspace(300., 1600., 401)
        result = pressure_analysis(temperature, np.full_like(temperature, 1e-9),
                                   np.zeros_like(temperature), 1e-10, 300., limit=1300.)
        self.assertEqual(result["status"], "temperature_limit_crossing")
        self.assertAlmostEqual(result["critical_pressure_mbar"], 0.01, places=12)

    def test_reference_pressure_changes_only_gas_in_both_profiles(self):
        config_path, _, _ = self.fixture()
        sin_config = load_config(config_path)
        si_path = self.folder / "silicon.toml"
        si_path.write_text('material = "Si"\ntemperature_points = 61\nreference_pressure_mbar = 0.01\n')
        for config in (sin_config, load_config(si_path)):
            with self.subTest(material=config["material"]):
                first, summary_first = calculate(config)
                second, summary_second = calculate({**config, "reference_pressure_mbar": 1.})
                first, second = first["powers.csv"], second["powers.csv"]
                np.testing.assert_allclose(second["p_gas_w"], 100. * first["p_gas_w"], rtol=1e-12)
                self.assertGreater(float(second["p_gas_w"][-1]), float(first["p_gas_w"][-1]))
                for name in first:
                    if name != "p_gas_w":
                        np.testing.assert_array_equal(first[name], second[name])
                self.assertEqual(summary_second["reference_pressure_mbar"], 1.)
                self.assertEqual(summary_first["pressure_analysis"], summary_second["pressure_analysis"])

    def test_cli_from_other_directory_and_replot_without_original_inputs(self):
        config, absorption, spectrum = self.fixture()
        outside = self.folder / "outside"
        outside.mkdir()
        output = self.folder / "results"
        environment = {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(self.folder / "mpl")}
        command = [sys.executable, str(ROOT / "run.py"), "--config", str(config),
                   "--output", str(output), "--dpi", "50"]
        result = subprocess.run(command, cwd=outside, env=environment, text=True,
                                capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((output / "powers.csv").is_file())
        self.assertEqual(json.loads((output / "summary.json").read_text())["profile"], "sin")
        expected = ("power_balance.png", "absorption_selection.png")
        for name in expected:
            self.assertGreater((output / name).stat().st_size, 1000)
            (output / name).unlink()
        # Replotting must consume saved numerical results, not resolve source paths.
        for path in (config, absorption, spectrum):
            path.unlink()
        shutil.rmtree(output / "inputs")
        result = subprocess.run([sys.executable, str(ROOT / "plotting.py"), "--output", str(output),
                                 "--dpi", "50"], cwd=outside, env=environment, text=True,
                                capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for name in expected:
            self.assertGreater((output / name).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
