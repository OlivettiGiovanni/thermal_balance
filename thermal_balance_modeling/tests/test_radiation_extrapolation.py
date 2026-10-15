"""Physical and numerical limits of the optional radiation optical model."""

from copy import deepcopy
import unittest

import numpy as np
from scipy.constants import e, epsilon_0, m_e, c

from scripts.radiation_optics import (carrier_densities, drude_permittivity,
    epsilon_to_nk, extrapolate_permittivity, temperature_scale)


class RadiationOpticsTests(unittest.TestCase):
    def setUp(self):
        # Deliberately synthetic material; no silicon constants are implicit.
        self.params = {"dos_masses_m0": [0.8, 0.4], "conductivity_masses_m0": [0.2, 0.6],
            "gap0_ev": 1.0, "gap_slope_ev_k": 2e-4, "gap_temperature_k": 200.,
            "net_doping_m3": 0., "relaxation_times_s": [[1e-13, 2e-13], [2e-13, 1e-13]],
            "temperature_exponents": [[0., 0.], [0., 0.]]}

    def test_scale_anchors_and_linear_limit(self):
        temperature = np.array([[200., 300., 400.], [500., 800., 1500.]])
        np.testing.assert_allclose(temperature_scale(temperature, 1., 0.), (temperature - 300) / 200)
        for p, theta in [(0., 800.), (2., 200.), (1e-12, 0.), (0., 1e8)]:
            np.testing.assert_array_equal(temperature_scale([300., 500.], p, theta), [0., 1.])
        # f(500) would overflow here; its normalized ratio remains representable.
        self.assertTrue(np.isfinite(temperature_scale(450., 0., 1e8)))
        for args in [(0., 0.), (-1., 200.), (1., -1.)]:
            with self.assertRaises(ValueError):
                temperature_scale(400., *args)

    def test_carrier_charge_neutrality_and_mass_action(self):
        temperatures = np.array([100., 300., 800., 1500.])
        intrinsic = carrier_densities(temperatures, self.params)
        np.testing.assert_array_equal(intrinsic[:, 0], intrinsic[:, 1])
        self.assertTrue(np.all(np.diff(intrinsic[:, 0]) > 0))
        for doping in [1e20, -1e20]:
            params = dict(self.params, net_doping_m3=doping)
            densities = carrier_densities(temperatures, params)
            # Compare populations directly: subtracting large intrinsic values
            # would amplify ordinary roundoff when the dopant fraction is tiny.
            np.testing.assert_allclose(np.max(densities, axis=-1), np.min(densities, axis=-1) + abs(doping), rtol=3e-15)
            np.testing.assert_allclose(densities[:, 0] * densities[:, 1], intrinsic[:, 0]**2, rtol=1e-14)

    def test_drude_sign_shape_and_relaxation_law(self):
        original = deepcopy(self.params)
        wavelength = np.array([1e-6, 3e-6, 10e-6])
        temperatures = np.array([[300., 500.], [800., 1200.]])
        actual = drude_permittivity(temperatures, wavelength, self.params)
        self.assertEqual(actual.shape, (2, 2, 3))
        self.assertTrue(np.all(actual.real < 0))
        self.assertTrue(np.all(actual.imag > 0))
        omega = 2 * np.pi * c / wavelength
        plasma = carrier_densities(300., self.params) * e**2 / (epsilon_0 * np.array([0.2, 0.6]) * m_e)
        expected = -plasma.sum() / (omega * (omega + 1j / 3e-13))
        np.testing.assert_allclose(actual[0, 0], expected, rtol=1e-14)
        self.assertEqual(self.params, original)

    def test_optical_conversion_preserves_weak_loss_and_rejects_gain(self):
        expected_n, expected_k = np.array([3.5, 0., 1e-15]), np.array([1e-25, 2., 3.])
        epsilon = (expected_n + 1j * expected_k)**2
        n, k = epsilon_to_nk(epsilon)
        np.testing.assert_allclose(n, expected_n, rtol=1e-14, atol=0.)
        np.testing.assert_allclose(k, expected_k, rtol=1e-14, atol=0.)
        for epsilon in [2. - 1e-25j, np.inf + 1j]:
            with self.assertRaisesRegex(ValueError, "passive"):
                epsilon_to_nk(epsilon)

    def test_extrapolation_exact_anchors_and_no_drude_double_counting(self):
        temperature = np.array([[300., 400., 500.], [600., 700., 800.]])
        epsilon0 = np.array([4. + .01j, 9. + .05j])
        d0, d1 = np.array([-.2 + .01j, -.3 + .02j]), np.array([-.4 + .02j, -.6 + .04j])
        drude = d0 + ((temperature - 300) / 200)[..., None] * (d1 - d0)
        # If the entire anchor change is Drude, no residual should be added.
        epsilon1 = epsilon0 + d1 - d0
        actual = extrapolate_permittivity(temperature, epsilon0, epsilon1, drude, d0, d1, 2., 200.)
        np.testing.assert_allclose(actual, epsilon0 + drude - d0, rtol=1e-14)
        np.testing.assert_array_equal(actual[0, 0], epsilon0)
        np.testing.assert_array_equal(actual[0, 2], epsilon1)
        self.assertFalse(np.array_equal(actual[-1, -1], epsilon0))


if __name__ == "__main__":
    unittest.main()
