"""Optical coverage jumps must not masquerade as thermal equilibria."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from balance import find_equilibria, pressure_report


class FallbackBoundaryTests(unittest.TestCase):
    def test_no_equilibrium_created_by_a_fallback_jump(self):
        temperature = np.array([300., 400., 500., 501., 600.])
        net = np.array([1., .8, .5, -.5, -.6])
        breaks = np.array([False, False, False, True, False])
        self.assertEqual(find_equilibria(temperature, net, breaks), [])

    def test_threshold_power_is_available_but_branch_is_unresolved(self):
        temperature = np.array([300., 400., 500., 501., 600.])
        breaks = np.array([False, False, False, True, False])
        report = pressure_report(temperature, np.ones(5), np.zeros(5), 1., 300., 600., breaks)
        self.assertEqual(report["status"], "unresolved_optical_fallback")
        self.assertAlmostEqual(report["temperature_limit"]["balance_pressure_mbar"], 1/300)
        self.assertIsNone(report["temperature_limit"]["on_cold_stable_branch"])


if __name__ == "__main__":
    unittest.main()
