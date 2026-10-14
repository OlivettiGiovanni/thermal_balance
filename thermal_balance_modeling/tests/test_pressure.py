"""Independent analytic checks of cold-branch loss and finite-domain handling."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pressure import critical_pressure


class PressureTests(unittest.TestCase):
    def test_quadratic_heating_fold_satisfies_balance_and_tangency(self):
        temperature = np.linspace(300., 1000., 1401)
        absorbed = 1e-9 + 1e-13 * (temperature - 300.)**2
        result = critical_pressure(temperature, absorbed, np.zeros_like(temperature), 1e-10, 300.)
        self.assertEqual(result["status"], "cold_branch_fold")
        self.assertAlmostEqual(result["critical_pressure_mbar"], 0.2, places=6)
        self.assertAlmostEqual(result["critical_temperature_k"], 400., delta=.002)
        delta = result["critical_temperature_k"] - 300.
        pressure = result["critical_pressure_mbar"]
        self.assertAlmostEqual((1e-9 + 1e-13*delta**2)/(1e-10*pressure*delta), 1., places=6)
        self.assertAlmostEqual(2e-13*delta/(1e-10*pressure), 1., delta=1e-5)

    def test_first_fold_is_not_the_global_minimum_or_proof_of_divergence(self):
        temperature = np.linspace(300., 850., 5501)
        x = (temperature - 300.) / 100.
        # dp_eq/dx has roots 1,2,4: minima at 1 and 4, with the latter lower.
        absorbed = 1e-8 * (32/7 + 10*x**2 - 45*x**3/14 + x**4/3)
        result = critical_pressure(temperature, absorbed, np.zeros_like(temperature), 1e-10, 300.)
        self.assertEqual(result["status"], "cold_branch_fold")
        self.assertAlmostEqual(result["critical_temperature_k"], 400., delta=.002)
        self.assertAlmostEqual(result["critical_pressure_mbar"], 491/42, places=5)
        later_minimum = 32/(7*4) + 10*4 - 45*4**2/14 + 4**3/3
        self.assertGreater(result["critical_pressure_mbar"], later_minimum)

    def test_vacuum_equilibrium_very_close_to_bath_precedes_any_fold(self):
        temperature = np.linspace(300., 1600., 401)
        result = critical_pressure(temperature, np.full_like(temperature, 1e-12),
                                   1e-8*(temperature - 300.), 1e-10, 300.)
        self.assertEqual(result["status"], "stable_to_vacuum")
        self.assertEqual(result["critical_pressure_mbar"], 0.)
        self.assertAlmostEqual(result["vacuum_temperature_k"], 300.0001, places=8)

    def test_a_material_limit_is_not_a_fold(self):
        temperature = np.linspace(300., 1600., 401)
        result = critical_pressure(temperature, np.full_like(temperature, 1e-9),
                                   np.zeros_like(temperature), 1e-10, 300., limit=1300.)
        self.assertEqual(result["status"], "temperature_limit_crossing")
        self.assertIsNone(result["critical_pressure_mbar"])
        self.assertAlmostEqual(result["threshold_pressure_mbar"], .01, places=12)
        self.assertEqual(result["threshold_temperature_k"], 1300.)

    def test_monotone_branch_ending_at_domain_boundary_is_unresolved(self):
        temperature = np.linspace(300., 1600., 401)
        result = critical_pressure(temperature, np.full_like(temperature, 1e-9),
                                   np.zeros_like(temperature), 1e-10, 300.)
        self.assertEqual(result["status"], "unresolved")
        self.assertIsNone(result["critical_pressure_mbar"])

    def test_no_laser_heating_has_no_positive_threshold(self):
        temperature = np.linspace(300., 1600., 401)
        result = critical_pressure(temperature, np.zeros_like(temperature),
                                   1e-8*(temperature-300.), 1e-10, 300.)
        self.assertEqual(result["status"], "no_heating")
        self.assertEqual(result["critical_pressure_mbar"], 0.)

    def test_vacuum_equilibrium_at_domain_endpoint_is_resolved(self):
        temperature = np.linspace(300., 500., 201)
        result = critical_pressure(temperature, np.full_like(temperature, 2e-9),
                                   1e-11*(temperature-300.), 1e-10, 300.)
        self.assertEqual(result["status"], "stable_to_vacuum")
        self.assertAlmostEqual(result["vacuum_temperature_k"], 500., places=8)

    def test_invalid_domain_and_gas_coefficient_raise(self):
        temperature = np.array([300., 400., 500.])
        with self.assertRaisesRegex(ValueError, "Gas coefficient"):
            critical_pressure(temperature, np.ones(3), np.zeros(3), 0., 300.)
        with self.assertRaisesRegex(ValueError, "Temperature limit"):
            critical_pressure(temperature, np.ones(3), np.zeros(3), 1., 300., limit=600.)


if __name__ == "__main__":
    unittest.main()
