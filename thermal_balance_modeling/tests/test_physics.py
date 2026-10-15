"""Independent physical limits and optical-input safeguards (standard unittest)."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.constants import atomic_mass, c, e, h, k as K_B, sigma

from scripts.absorption import absorption_powers, bandgap_alpha, drude_alpha, tpa_beta
from scripts.cooling import cylinder_area, cylinder_cross_section, gas_coefficient, gas_power, radiation_power


class PhysicsTests(unittest.TestCase):
    def test_gas_units_and_fixed_bath(self):
        area, bath = 3e-13, 300.0
        gas = {"mass_amu": 28.0, "gamma": 1.4, "accommodation": 0.8}
        speed = np.sqrt(8 * K_B * bath / (np.pi * 28 * atomic_mass))
        G = gas_coefficient(area, bath, gas)
        expected = 0.8 * area * 100 * speed / (8 * bath) * 6
        self.assertAlmostEqual(G / expected, 1.0)
        np.testing.assert_allclose(gas_power([300, 600, 900], 2, bath, G), [0, 600 * G, 1200 * G])
        np.testing.assert_allclose(gas_power([300, 900], 0, bath, G), 0)
        self.assertLess(gas_power(200, 1, bath, G), 0)
        self.assertAlmostEqual(cylinder_area(100, 200) / (6 * np.pi * 1e-14), 1)

    def test_radiation_matches_stefan_boltzmann(self):
        # C_abs=A/4 is the isotropic absorption cross section of a black surface.
        wavelength = np.geomspace(1e-9, 1e-2, 12000)
        area, bath = 2e-12, 300.0
        temperatures = np.array([200.0, 300.0, 900.0])
        actual = radiation_power(temperatures, bath, wavelength, np.full_like(wavelength, area / 4))
        expected = area * sigma * (temperatures**4 - bath**4)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-20)

    def test_lossless_scattering_is_not_heat(self):
        wavelength = np.geomspace(1e-6, 20e-6, 50)
        cross_section = cylinder_cross_section(wavelength, 3.5, 0, 100, 400)
        np.testing.assert_array_equal(cross_section, 0)
        absorbing = cylinder_cross_section(wavelength, 3.5, 0.1, 100, 400)
        self.assertTrue(np.all(absorbing > 0))
        self.assertEqual(float(radiation_power(300, 300, wavelength, absorbing)), 0.0)

    def test_spherical_depolarization_limit(self):
        # H=2R gives L=1/3, with the actual cylinder volume retained.
        wavelength, radius, height = 1e-4, 20.0, 40.0
        volume = np.pi * (radius * 1e-9)**2 * height * 1e-9
        contrast = (2 + 0.2j)**2 - 1
        polarizability = 3 * volume * contrast / (contrast + 3)
        wave = 2 * np.pi / wavelength
        expected = wave * polarizability.imag / abs(1 - 1j * wave**3 * polarizability / (6 * np.pi))**2
        actual = cylinder_cross_section(wavelength, 2, 0.2, radius, height)
        self.assertAlmostEqual(actual / expected, 1)

    def test_drude_density_and_index_scaling(self):
        params = {"density_prefactors": [2e18, 0], "activation_ev": [0, 0],
                  "conductivity_masses_m0": [0.2, 0.5],
                  "relaxation_times_s": [[1e-13, 1e-13], [2e-13, 3e-13]],
                  "temperature_exponents": [[0, 0], [0, 0]]}
        result = drude_alpha(np.array([300, 1200]), 1e-6, 2.0, 300, params)
        self.assertAlmostEqual(result["alpha_m_inv"][1] / result["alpha_m_inv"][0], 8)
        np.testing.assert_allclose(result["density_m3"][:, 0], 2e18 * np.array([300, 1200])**1.5)
        np.testing.assert_allclose(result["damping_s_inv"], [[1 / 3e-13, 1 / 4e-13]] * 2)
        double_n = drude_alpha([300, 1200], 1e-6, 4.0, 300, params)
        np.testing.assert_allclose(double_n["alpha_m_inv"], result["alpha_m_inv"] / 2)

    def test_tpa_phonon_emission_threshold_at_low_temperature(self):
        params = {"weights": [2.0], "phonon_temperatures_k": [1000],
                  "gap0_ev": 1.0, "gap_slope_ev_k": 0, "gap_temperature_k": 1}
        below = h * c / (0.53 * e)
        above = h * c / (0.60 * e)
        self.assertEqual(float(tpa_beta(1, below, params)), 0.0)
        expected = 2 * (1.2 - 1.0 - 1000 * K_B / e)**2
        self.assertAlmostEqual(float(tpa_beta(1, above, params)), expected)

    def test_band_edge_phonon_pairing_and_disabled_channels(self):
        params = {"sqrt_weights": [3.0], "quadratic_weights": [0.0],
                  "quadratic_offset_ev": 0, "phonon_temperatures_k": [1000],
                  "gap0_ev": 1, "gap_slope_ev_k": 0, "gap_temperature_k": 1}
        self.assertEqual(float(bandgap_alpha(1, h * c / (1.05 * e), params)), 0.0)
        expected = 3 * np.sqrt(1.2 - 1.0 - 1000 * K_B / e)
        self.assertAlmostEqual(float(bandgap_alpha(1, h * c / (1.2 * e), params)), expected)
        np.testing.assert_array_equal(tpa_beta([300, 600], 1e-6, {"weights": [0]}), 0)
        np.testing.assert_array_equal(bandgap_alpha([300, 600], 1e-6, {"sqrt_weights": [0], "quadratic_weights": [0]}), 0)
        np.testing.assert_array_equal(drude_alpha([300, 600], 1e-6, 1, 300, {"density_prefactors": [0, 0]})["alpha_m_inv"], 0)


class OpticalPowerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "optical.csv"
        self.volume = np.pi * (100e-9)**2 * 200e-9
        self.moment1 = 1e-6 / 1000
        self.row = {"radius_nm": 100, "p_abs_W": 1e-6,
                    "intensity_squared_integral_w2_per_m": 2 * self.moment1**2 / self.volume}
        self.config = {
            "material": {"name": "synthetic", "wavelength_nm": 1000, "n_0": 2,
                         "reference_temperature_k": 300},
            "data": {"absorption_file": str(self.path)},
            "geometry": {"radius_nm": 100, "height_nm": 200},
            "linear": {"simulation_alpha_m_inv": 1000, "simulation_intensity_w_um2": 1},
            "experiment": {"intensity_w_um2": 1},
            "tpa": {"weights": [1e-11], "phonon_temperatures_k": [500],
                    "gap0_ev": 1, "gap_slope_ev_k": 0, "gap_temperature_k": 1},
            "fca": {"density_prefactors": [0, 0]},
            "bga": {"sqrt_weights": [0], "quadratic_weights": [0]},
        }
        self.write_rows([self.row])

    def write_rows(self, rows):
        with self.path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def prepared(self):
        return {"config": deepcopy(self.config), "temperature_k": np.array([300, 600]), "n_laser": 2}

    def test_simulation_linear_power_and_intensity_scaling(self):
        first = self.prepared()
        power1 = absorption_powers(first)
        second = self.prepared()
        second["config"]["experiment"]["intensity_w_um2"] = 3
        power3 = absorption_powers(second)
        np.testing.assert_array_equal(power1["p_linear_w"], [1e-6, 1e-6])
        np.testing.assert_allclose(power3["p_linear_w"], power1["p_linear_w"] * 3)
        np.testing.assert_allclose(power3["p_tpa_w"], power1["p_tpa_w"] * 9)
        self.assertTrue(any("Unverified" in note for note in first["optical_notes"]))

    def test_tpa_requires_independent_squared_field_moment(self):
        self.row.pop("intensity_squared_integral_w2_per_m")
        self.write_rows([self.row])
        with self.assertRaisesRegex(ValueError, "cannot be inferred"):
            absorption_powers(self.prepared())
        prepared = self.prepared()
        prepared["config"]["tpa"] = {"weights": [0]}
        absorption_powers(prepared)
        self.assertIsNone(prepared["field_squared_integral_w2_m"])

    def test_jensen_bound_and_metadata_mismatch(self):
        self.row["intensity_squared_integral_w2_per_m"] = 0.5 * self.moment1**2 / self.volume
        self.write_rows([self.row])
        with self.assertRaisesRegex(ValueError, "violates"):
            absorption_powers(self.prepared())
        self.path.with_suffix(".json").write_text(json.dumps({"wavelength_nm": 1500}))
        with self.assertRaisesRegex(ValueError, "metadata wavelength_nm"):
            absorption_powers(self.prepared())

    def test_duplicate_headers_and_malformed_rows_are_rejected(self):
        self.path.write_text("radius_nm,p_abs_W,p_abs_W\n100,1e-6,1e-6\n")
        with self.assertRaisesRegex(ValueError, "unique"):
            absorption_powers(self.prepared())
        for data in ("100,1e-6,extra", "100"):
            self.path.write_text("radius_nm,p_abs_W\n" + data + "\n")
            with self.assertRaisesRegex(ValueError, "number of fields"):
                absorption_powers(self.prepared())

    def test_sidecar_notes_preserve_unverified_provenance(self):
        self.path.with_suffix(".json").write_text(json.dumps({"note": "Historical illustrative input", "status": "unverified"}))
        prepared = self.prepared()
        absorption_powers(prepared)
        self.assertIn("Simulation sidecar note: Historical illustrative input", prepared["optical_notes"])
        self.assertIn("Simulation sidecar status: unverified", prepared["optical_notes"])
        self.assertTrue(any("Unverified simulation metadata" in note for note in prepared["optical_notes"]))

    def test_peak_selection_uses_density_not_total_power(self):
        density = [1, 5, 2, 6, 1]
        rows = [{"radius_nm": radius, "absorption_density_W_m3": rho * 1e12}
                for radius, rho in zip([100, 200, 300, 400, 500], density)]
        self.write_rows(rows)
        prepared = self.prepared()
        prepared["config"]["tpa"] = {"weights": [0]}
        prepared["config"]["geometry"]["radius_nm"] = "first_peak"
        absorption_powers(prepared)
        self.assertEqual(prepared["particle"]["radius_nm"], 200)
        prepared["config"]["geometry"]["radius_nm"] = "max_peak"
        absorption_powers(prepared)
        self.assertEqual(prepared["particle"]["radius_nm"], 400)
        prepared["config"]["geometry"]["radius_nm"] = 201
        with self.assertRaisesRegex(ValueError, "no interpolation"):
            absorption_powers(prepared)


if __name__ == "__main__":
    unittest.main()
