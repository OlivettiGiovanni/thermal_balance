"""Checks for actual CLI workflows, input fallbacks and pressure interpretation."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from prepare_inputs import prepare_inputs, _interp, _read_table
from balance import find_equilibria, pressure_report


class InputAndBalanceTests(unittest.TestCase):
    def test_missing_values_do_not_become_interpolated_data(self):
        actual = _interp([1, 2, 3], [2, np.nan, 4], [.5, 1, 1.5, 2, 2.5, 3, 4])
        np.testing.assert_array_equal(np.isfinite(actual), [False, True, False, False, False, True, False])

    def test_sin_preparation_retains_reference_index(self):
        prepared = prepare_inputs(ROOT/"config/SiN_HS.toml")
        t = prepared["temperature_k"]
        self.assertEqual(prepared["n_laser"][0], 2.)
        self.assertTrue(np.all(prepared["n_laser"][t > 460] == 2.))
        self.assertTrue(np.all(~prepared["k_measured"][t > 300]))
        self.assertTrue(prepared["borrowed_spectrum"])
        self.assertTrue(np.all(np.isfinite(prepared["k_spectrum"])))

    def test_placeholder_poly_table_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "populate"):
            _read_table(ROOT/"data/material/Si_polycrystalline.csv")

    def test_stability_and_temperature_threshold_are_distinct(self):
        temperature = np.linspace(300, 2000, 1701)
        heating = np.full_like(temperature, 1e-9)
        conductance = 2e-10
        equilibria = find_equilibria(temperature, heating-.1*conductance*(temperature-300))
        self.assertEqual(equilibria[0]["stable"], True)
        self.assertAlmostEqual(equilibria[0]["temperature_k"], 350.)
        report = pressure_report(temperature, heating, np.zeros_like(temperature), conductance, 300, 1950)
        self.assertIsNone(report["cold_branch_loss"])
        self.assertAlmostEqual(report["temperature_limit"]["balance_pressure_mbar"], 1e-9/(2e-10*1650))
        self.assertTrue(report["temperature_limit"]["on_cold_stable_branch"])

    def test_first_branch_loss_precedes_hot_threshold(self):
        temperature = np.linspace(300, 1000, 701)
        heating = 1+(temperature-300)**2/10000
        report = pressure_report(temperature, heating, np.zeros_like(temperature), 1., 300., 700.)
        self.assertAlmostEqual(report["cold_branch_loss"]["temperature_k"], 400.)
        self.assertAlmostEqual(report["cold_branch_loss"]["pressure_mbar"], .02)
        self.assertFalse(report["temperature_limit"]["on_cold_stable_branch"])

    def test_negative_balance_pressure_does_not_mean_negative_gas(self):
        temperature = np.linspace(300, 2000, 1701)
        report = pressure_report(temperature, np.ones_like(temperature), .01*(temperature-300), 1., 300., 1950.)
        self.assertLess(report["temperature_limit"]["balance_pressure_mbar"], 0.)
        self.assertIn("vacuum", report["temperature_limit"]["note"])

    def test_later_stable_branch_is_not_the_cold_branch(self):
        temperature = np.arange(300., 1201., 100.)
        balance_pressure = np.array([0, 5, 2, .5, -.5, -1, -.5, .5, 1, .5])
        radiated = 1000+temperature
        absorbed = radiated+balance_pressure*(temperature-300)
        report = pressure_report(temperature, absorbed, radiated, 1., 300., 1200.)
        self.assertFalse(report["temperature_limit"]["on_cold_stable_branch"])


class CommandTests(unittest.TestCase):
    def command(self, script, *arguments, cwd=ROOT, input_text=None):
        return subprocess.run([sys.executable, str(ROOT/script), *map(str, arguments)],
                              cwd=cwd, input=input_text, capture_output=True, text=True, check=True)

    def test_full_cylinder_run_and_replot_from_another_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"output"
            self.command("scripts/run.py", "--config", ROOT/"config/SiN_HS.toml",
                         "--output", output, "--dpi", 90, cwd=folder)
            self.assertEqual(sorted(p.name for p in output.glob("*.png")), ["power_balance.png"])
            original = (output/"results.npz").read_bytes()
            (output/"power_balance.png").unlink()
            self.command("scripts/plotting.py", "--output", output, "--dpi", 100, cwd=folder)
            self.assertTrue((output/"power_balance.png").is_file())
            self.assertEqual((output/"results.npz").read_bytes(), original)
            with np.load(output/"results.npz", allow_pickle=False) as result:
                self.assertNotIn("n_spectrum", result.files)
                np.testing.assert_allclose(result["p_abs_w"], sum(result[f"p_{term}_w"] for term in ["linear", "tpa", "fca", "bga"]))
                self.assertGreater(result["p_fca_w"][-1], 0.)
                self.assertEqual(result["p_gas_w"][0], 0.)
                self.assertEqual(result["p_bb_w"][0], 0.)
            summary = json.loads((output/"summary.json").read_text())
            self.assertEqual(summary["particle"]["radius_nm"], 585.)

    def test_interactive_prepared_variables_remain_available(self):
        with tempfile.TemporaryDirectory() as folder:
            command = [sys.executable, "-i", str(ROOT/"scripts/run.py"), "--config",
                       str(ROOT/"config/SiN_HS.toml"), "--output", folder, "--dpi", "70"]
            result = subprocess.run(command, input="print('INSPECTABLE', len(prepared['n_spectrum']), len(results['p_fca_w']))\nexit()\n",
                                    capture_output=True, text=True, check=True, cwd=ROOT)
            self.assertIn("INSPECTABLE", result.stdout)

    @unittest.skipUnless(importlib.util.find_spec("miepython"), "Install requirements for the sphere benchmark")
    def test_standalone_benchmark_and_replot(self):
        with tempfile.TemporaryDirectory() as folder:
            self.command("Junneman/compare_Junneman.py", "--output", folder, "--dpi", 90, cwd=folder)
            self.assertEqual(sorted(p.name for p in Path(folder).glob("*.png")), ["absorption_comparison.png", "power_balance.png"])
            original = (Path(folder)/"results.npz").read_bytes()
            self.command("scripts/plotting.py", "--output", folder, "--dpi", 80)
            self.assertEqual((Path(folder)/"results.npz").read_bytes(), original)
            summary = json.loads((Path(folder)/"summary.json").read_text())
            fit = summary["radiation"]["fit"]
            self.assertEqual(fit["fitted_parameter_count"], 2)
            self.assertEqual(fit["fit_temperatures_k"], [350., 400., 450.])
            self.assertLess(fit["maximum_absolute_relative_error"], .025)
            with np.load(Path(folder)/"results.npz", allow_pickle=False) as result:
                self.assertEqual(str(result["kind"]), "junnemann")
                self.assertTrue(np.all(np.isfinite(result["p_abs_w"])))
                t = result["temperature_k"]
                for key in ["p_bb_dipole_w", "p_bb_dipole_room_w", "p_bb_mie_room_w", "p_bb_mie_w"]:
                    self.assertEqual(result[key][0], 0.)
                    self.assertTrue(np.all(np.isfinite(result[key])))
                    self.assertTrue(np.all(result[key][1:] > 0))
                np.testing.assert_array_equal(result["franta_temperature_k"], [300., 350., 400., 450., 500.])
                reference = result["p_bb_franta_w"]
                self.assertEqual(reference[0], 0.)
                self.assertTrue(np.all(np.isfinite(reference)))
                self.assertTrue(np.all(reference[1:] > 0))
                native_model = result["p_bb_dipole_w"][np.searchsorted(t, result["franta_temperature_k"])]
                np.testing.assert_allclose(native_model[-1], reference[-1], rtol=1e-12, atol=0.)
                self.assertLess(np.max(abs(native_model[1:-1]/reference[1:-1]-1)), .025)
                self.assertGreater(result["p_bb_mie_room_w"][t == 1000][0],
                                   10*result["p_bb_dipole_room_w"][t == 1000][0])
                # At the exact 500 K anchor, hot Mie must use the native Franta spectrum.
                spec = importlib.util.spec_from_file_location("benchmark", ROOT/"Junneman/compare_Junneman.py")
                benchmark = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(benchmark)
                table = _read_table(ROOT/"Junneman/si_franta_300_500k.csv")
                source_wavelength = np.unique(table[:, 1])*1e-6
                low, high = np.array(summary["radiation"]["wavelength_range_um"])*1e-6
                wavelength = np.r_[low, source_wavelength[(source_wavelength > low)
                                                         & (source_wavelength < high)], high]
                optics = benchmark.prepare_radiation_optics(table, np.array([500.]), wavelength*1e6, 300.)
                cross_section = benchmark.sphere_absorption(wavelength, optics["n"][0], optics["k"][0])
                anchor_power = benchmark.radiation_power(np.array([500.]), 300., wavelength, cross_section)[0]
                np.testing.assert_allclose(result["p_bb_mie_w"][t == 500][0], anchor_power, rtol=1e-11, atol=0.)
                # Above coverage it must neither freeze the 500 K spectrum nor reset to 300 K.
                frozen_power = benchmark.radiation_power(np.array([1000.]), 300., wavelength, cross_section)[0]
                self.assertGreater(abs(result["p_bb_mie_w"][t == 1000][0]/frozen_power-1), .01)
                self.assertGreater(abs(result["p_bb_mie_w"][t == 1000][0]
                                       /result["p_bb_mie_room_w"][t == 1000][0]-1), .01)
                self.assertIn("p_bb_mie_w", summary["radiation"]["models"])
                self.assertIn("p_bb_mie_w_vs_digitized", summary["comparisons"])
                self.assertFalse(np.any(result["radiation_break_before"]))
                self.assertIn("extrapolated above 500 K", str(result["radiation_footnote"]))
                self.assertEqual(sorted(key for key in result.files if key.startswith("bb_fit_")),
                                 ["bb_fit_p", "bb_fit_theta_k"])
                self.assertEqual(float(result["bb_fit_p"]), fit["p"])
                self.assertEqual(float(result["bb_fit_theta_k"]), fit["theta_k"])
                self.assertNotIn("radiation_cross_section_m2", result.files)

    def test_digitized_curves_can_be_rebuilt(self):
        with tempfile.TemporaryDirectory() as folder:
            self.command("Junneman/extract.py", "--output", folder)
            for name in ["junnemann_fig4.csv", "junnemann_fig6.csv"]:
                original = np.genfromtxt(ROOT/"Junneman"/name, delimiter=",", names=True)
                rebuilt = np.genfromtxt(Path(folder)/name, delimiter=",", names=True)
                for key in original.dtype.names:
                    np.testing.assert_allclose(original[key], rebuilt[key], rtol=1e-8, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
