"""Temperature-resolved radiation, explicit missing data, and reference fallbacks."""
import unittest
import numpy as np
from scipy.constants import sigma
from scripts.prepare_inputs import prepare_radiation_optics
from scripts.cooling import dipole_cross_section, radiation_power


def spectrum(temperature, n, k, wavelength=(1., 2., 3.)):
    """Build a small optical slice with independent n and k values."""
    wavelength = np.asarray(wavelength)
    return np.column_stack((np.full(wavelength.size, temperature), wavelength,
                            np.broadcast_to(n, wavelength.shape), np.broadcast_to(k, wavelength.shape)))


class TemperatureOpticsTests(unittest.TestCase):
    def test_supported_interpolation_and_no_temperature_extrapolation(self):
        table = np.vstack([spectrum(300, 2., .1), spectrum(400, 3., .2), spectrum(500, 4., .3)])
        result = prepare_radiation_optics(table, [250, 300, 350, 500, 501], [1., 1.5, 3.])
        np.testing.assert_allclose(result["n"], np.array([2., 2., 2.5, 4., 2.])[:, None] * np.ones((1, 3)))
        np.testing.assert_allclose(result["k"][2], .15)
        np.testing.assert_array_equal(result["n_room_fallback"][:, 0], [True, False, False, False, True])
        np.testing.assert_array_equal(result["break_before"], [False, False, False, False, True])
        self.assertIn("300–500 K", result["footnote"])

    def test_missing_n_and_k_are_independent_and_temperature_gaps_are_not_bridged(self):
        table = np.vstack([spectrum(300, 2., .1), spectrum(400, 3., np.nan), spectrum(500, 4., .3)])
        result = prepare_radiation_optics(table, [300, 350, 400, 450, 500], [1., 2., 3.])
        np.testing.assert_allclose(result["n"][:, 0], [2., 2.5, 3., 3.5, 4.])
        np.testing.assert_allclose(result["k"][:, 0], [.1, .1, .1, .1, .3])
        self.assertFalse(result["n_room_fallback"].any())
        np.testing.assert_array_equal(result["k_room_fallback"][:, 0], [False, True, True, True, False])
        np.testing.assert_array_equal(result["break_before"], [False, False, False, False, True])

    def test_explicit_wavelength_gaps_are_not_bridged(self):
        wavelength = [1., 2., 3., 4., 5.]
        table = np.vstack([spectrum(300, 2., .1, wavelength),
                           spectrum(400, [3., 3., np.nan, 3., 3.], .2, wavelength)])
        result = prepare_radiation_optics(table, [400], [1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.])
        np.testing.assert_allclose(result["n"][0], [3, 3, 3, 2, 2, 2, 3, 3, 3])
        np.testing.assert_allclose(result["k"][0], .2)

    def test_single_wavelength_thermooptic_data_are_not_a_broadband_spectrum(self):
        table = np.vstack([spectrum(300, 2., .1, [1., 1.55, 2.]),
                           [[400., 1.55, 3., np.nan], [500., 1.55, 4., np.nan]]])
        result = prepare_radiation_optics(table, [300, 350, 400, 500], [1., 1.55, 2.])
        np.testing.assert_allclose(result["n"], 2.)
        np.testing.assert_allclose(result["k"], .1)
        self.assertTrue(np.all(result["n_room_fallback"][1:]))
        self.assertIn("no broadband T data", result["footnote"])

    def test_other_material_supplies_reference_only_and_never_replaces_primary_hot_data(self):
        primary = np.vstack([spectrum(300, 2., np.nan), spectrum(400, 3., .3)])
        other = np.vstack([spectrum(300, 10., .1), spectrum(400, 20., .9)])
        result = prepare_radiation_optics(primary, [300, 350, 400, 500], [1., 2., 3.], fallback_table=other)
        np.testing.assert_allclose(result["n"][:, 0], [2., 2.5, 3., 2.])
        np.testing.assert_allclose(result["k"][:, 0], [.1, .1, .3, .1])
        self.assertTrue(result["k_borrowed_reference"].all())
        self.assertFalse(result["n_borrowed_reference"].any())

    def test_room_only_table_and_unavailable_room_spectrum(self):
        result = prepare_radiation_optics(spectrum(300, 2., .1), [300, 600], [1., 2.])
        np.testing.assert_allclose(result["n"], 2.)
        np.testing.assert_array_equal(result["n_room_fallback"], [[False, False], [True, True]])
        self.assertTrue(result["footnote"])
        self.assertFalse(result["break_before"].any())
        with self.assertRaisesRegex(ValueError, "reference-spectrum"):
            prepare_radiation_optics(spectrum(400, 3., .2), [400], [1., 2.])


class TemperatureRadiationTests(unittest.TestCase):
    def test_sphere_dipole_matches_analytic_polarizability_and_lossless_limit(self):
        radius = 130e-9
        wavelength = np.geomspace(.5e-6, 100e-6, 100)
        n, k = np.array([[2.], [3.5]]), np.array([[.02], [.1]])
        contrast = (n + 1j * k)**2 - 1
        polarizability = 4 * np.pi * radius**3 * contrast / (contrast + 3)
        wave = 2 * np.pi / wavelength
        expected = wave * polarizability.imag / abs(1 - 1j * wave**3 * polarizability / (6 * np.pi))**2
        actual = dipole_cross_section(wavelength, n, k, 4 * np.pi * radius**3 / 3, [1/3] * 3)
        np.testing.assert_allclose(actual, expected, rtol=2e-15)
        np.testing.assert_array_equal(dipole_cross_section(wavelength, n, 0., 4 * np.pi * radius**3 / 3, [1/3] * 3), 0.)

    def test_dipole_volume_and_depolarization_validation(self):
        for factors in ([.5, .5], [0., .5, .5], [.3, .3, .3], [np.nan, .5, .5]):
            with self.assertRaisesRegex(ValueError, "depolarization"):
                dipole_cross_section(1e-6, 2., .1, 1e-21, factors)
        with self.assertRaisesRegex(ValueError, "volume"):
            dipole_cross_section(1e-6, 2., .1, -1e-21, [1/3] * 3)

    def test_temperature_dependent_greybody_matches_net_stefan_boltzmann(self):
        wavelength = np.geomspace(1e-9, 1e-2, 12000)
        temperature = np.array([200., 300., 900.])
        emissivity, area, bath = np.array([.2, .5, .8]), 2e-12, 300.
        cross_section = emissivity[:, None] * np.full(wavelength.size, area / 4)
        actual = radiation_power(temperature, bath, wavelength, cross_section)
        expected = emissivity * area * sigma * (temperature**4 - bath**4)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-20)
        self.assertEqual(actual[1], 0.)

    def test_cross_section_shape_is_checked_and_fixed_spectrum_is_compatible(self):
        wavelength = np.geomspace(1e-6, 1e-3, 1000)
        temperature = np.array([[300., 400.], [500., 600.]])
        cross_section = np.full_like(wavelength, 1e-12)
        fixed = radiation_power(temperature, 300, wavelength, cross_section)
        repeated = radiation_power(temperature, 300, wavelength, np.broadcast_to(cross_section, (2, 2, 1000)))
        np.testing.assert_array_equal(fixed, repeated)
        with self.assertRaisesRegex(ValueError, "shape"):
            radiation_power(temperature, 300, wavelength, np.ones((2, 1000)))


if __name__ == "__main__":
    unittest.main()
