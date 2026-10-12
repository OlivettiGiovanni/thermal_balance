"""Physics checks anchored to source values, limiting cases and energy balance."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.constants import c, e, epsilon_0, m_e, sigma
from scipy.integrate import trapezoid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import silicon
import thermal


class SiliconTests(unittest.TestCase):
    def test_tpa_cold_phonon_limit(self):
        # Spontaneous phonon emission survives as the phonon bath empties.
        physical = silicon.tpa_beta(0.01, "physical", photon_energy_ev=0.797)
        printed = silicon.tpa_beta(0.01, "printed", photon_energy_ev=0.797)
        self.assertAlmostEqual(physical / 1e-11, 0.43527, places=5)
        self.assertAlmostEqual(printed / 1e-11, 0.713337, places=5)
        # Sinclair reports approximately .42 cm/GW at 5.5 K: only one pairing fits.
        self.assertLess(abs(physical / 1e-11 - 0.42), 0.03)
        self.assertGreater(abs(printed / 1e-11 - 0.42), 0.2)

    def test_tpa_si_units_and_explicit_photon_energy(self):
        # 1 cm/GW = 1e-11 m/W; source-energy beta at 300 K is .777757 cm/GW.
        beta = silicon.tpa_beta(300.0, "physical", photon_energy_ev=0.797)
        self.assertAlmostEqual(beta / 1e-11, 0.77775734, places=7)
        default = silicon.tpa_beta(300.0, "physical")
        self.assertAlmostEqual(default / 1e-11, 0.79803871, places=7)
        self.assertGreater(default, beta)  # hc/1550 nm is greater than .797 eV.
        terms = silicon.silicon_terms(300.0, 1e11, "physical", photon_energy_ev=0.797)
        self.assertAlmostEqual(float(terms["tpa"]), 0.77775734, places=7)

    def test_band_edge_threshold_and_reference_value(self):
        # At this photon energy neither one-phonon channel opens below 1000 K.
        alpha = silicon.bandgap_absorption(np.array([300.0, 600.0, 1000.0]),
                                          photon_energy_ev=0.797)
        np.testing.assert_array_equal(alpha, 0.0)
        self.assertAlmostEqual(silicon.bandgap_absorption(1100.0, photon_energy_ev=0.797)
                               / 100.0, 7.700838594, places=6)

    def test_intrinsic_carrier_units_and_implicit_root(self):
        self.assertAlmostEqual(silicon.intrinsic_carriers(300.0) / 1e16, 1.37987636, places=7)
        temperatures = np.array([400.0, 800.0, 1200.0, 1600.0])
        density_cm3 = silicon.intrinsic_carriers(temperatures, "corrected") / 1e6
        # Substitute the solution into Sturm's ORIGINAL implicit cm^-3 equation.
        shift_ev = -7.1e-10 * np.sqrt(density_cm3 / temperatures)
        rhs = 3.87e16 * temperatures**1.5 * np.exp(
            -(0.605 + shift_ev) / (8.617333262145e-5 * temperatures))
        np.testing.assert_allclose(density_cm3, rhs, rtol=2e-11)
        with self.assertRaisesRegex(ValueError, "no real root"):
            silicon.intrinsic_carriers(1640.0, "corrected")

    def test_drude_mobility_units_and_high_frequency_limit(self):
        np.testing.assert_allclose(silicon.carrier_mobilities(300.0),
                                   [0.133981224732, 0.046126635868], rtol=1e-10)
        self.assertAlmostEqual(silicon.drude_cross_section(300.0) / 1e-23,
                               7.9351326602, places=8)
        wavelength = 1e-9  # omega >> damping gives Sinclair Appendix-A formula.
        electron = 1.0 / ((0.3 * m_e)**2 * 0.133981224732)
        hole = 1.0 / ((0.4 * m_e)**2 * 0.046126635868)
        asymptote = e**3 * wavelength**2 * (electron + hole) / (
            4 * math.pi**2 * epsilon_0 * c**3 * 3.48)
        self.assertAlmostEqual(silicon.drude_cross_section(300.0, wavelength)
                               / asymptote, 1.0, places=8)
        self.assertLess(silicon.fca_absorption(300.0, "drude"),
                        silicon.fca_absorption(300.0, "empirical"))

    def test_zero_intensity_and_scalar_vector_consistency(self):
        temperatures = np.array([300.0, 800.0, 1500.0])
        for tpa in ("printed", "physical"):
            for fca in ("empirical", "drude"):
                for density in ("uncorrected", "corrected"):
                    with self.subTest(tpa=tpa, fca=fca, density=density):
                        values = silicon.silicon_terms(temperatures, 0.0, tpa, fca, density)
                        np.testing.assert_array_equal(values["tpa"], 0.0)
                        self.assertTrue(np.all(values["fca"] > 0.0))
                        for index, temperature in enumerate(temperatures):
                            scalar = silicon.silicon_terms(temperature, 0.0, tpa, fca, density)
                            for name in values:
                                self.assertAlmostEqual(float(scalar[name]),
                                                       float(values[name][index]), places=10)


class ThermalTests(unittest.TestCase):
    @staticmethod
    def material(extinction=0.03):
        wavelength = np.geomspace(0.5e-6, 100e-6, 1200)
        return {"temperature_k": None, "wavelength_m": wavelength,
                "n": np.full((1, len(wavelength)), 2.0),
                "k": np.full((1, len(wavelength)), extinction)}

    def test_mie_intrinsic_silicon_reference_and_lossless_limit(self):
        # d=260 nm, lambda=1550 nm, n=3.48, I=.1 W/um^2, alpha=4.3e-6 cm^-1.
        power = thermal.sphere_heating(4.3e-4, 1e11)
        self.assertAlmostEqual(float(power) / 1.69578364e-13, 1.0, places=5)
        self.assertEqual(float(thermal.sphere_heating(0.0, 1e11)), 0.0)

    def test_mie_power_scaling_and_weak_tpa_quadratic_heating(self):
        # The electromagnetic solver stays linear in illumination at fixed alpha.
        first = thermal.sphere_heating(2.0, 1e10)
        second = thermal.sphere_heating(2.0, 1e11)
        self.assertAlmostEqual(float(second / first), 10.0, places=10)
        # beta*I changes alpha; at weak absorption, total TPA heating scales as I^2.
        first = thermal.sphere_heating(0.1, 1e10)
        second = thermal.sphere_heating(1.0, 1e11)
        self.assertAlmostEqual(float(second / first), 100.0, delta=0.01)

    def test_gas_mbar_normalization_and_thermal_direction(self):
        area = 2 * math.pi * 585e-9 * (585e-9 + 400e-9)
        coefficient = thermal.gas_coefficient(area)
        power = thermal.gas_power(400.0, 0.1, coefficient)
        self.assertAlmostEqual(float(power) / 3.39088429595e-9, 1.0, places=10)
        self.assertAlmostEqual(float(thermal.gas_power(400.0, 10.0, coefficient) / power),
                               100.0, places=10)
        self.assertEqual(float(thermal.gas_power(300.0, 1.0, coefficient)), 0.0)
        self.assertLess(float(thermal.gas_power(250.0, 1.0, coefficient)), 0.0)

    def test_planck_integral_recovers_stefan_boltzmann(self):
        wavelength = np.geomspace(1e-8, 1e-2, 16000)
        radiance = thermal.planck_lambda(wavelength, 500.0)
        flux = math.pi * trapezoid(radiance, wavelength)
        self.assertAlmostEqual(float(flux) / (sigma * 500.0**4), 1.0, places=6)

    def test_net_radiation_equilibrium_sign_and_monotonicity(self):
        power = thermal.radiation_power(np.array([250., 300., 400., 800., 1600.]),
                                        self.material(), 585., 400., 300.)
        self.assertEqual(float(power[1]), 0.0)
        self.assertLess(float(power[0]), 0.0)
        self.assertTrue(np.all(np.diff(power) > 0.0))
        lossless = thermal.radiation_power(np.array([300., 1600.]), self.material(0.0),
                                           585., 400., 300.)
        np.testing.assert_array_equal(lossless, 0.0)


if __name__ == "__main__":
    unittest.main()
