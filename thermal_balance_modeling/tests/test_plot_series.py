"""The benchmark plots exactly the requested comparisons, including sparse Franta dots."""
import unittest
from unittest.mock import patch
import numpy as np
from matplotlib.figure import Figure
from scripts import plotting


class PowerSeriesTests(unittest.TestCase):
    def fitted_results(self):
        """A saved benchmark with separate fixed-optics and hot-optics powers."""
        temperature = np.array([300., 500., 800., 1200.])
        results = {"kind": "junnemann", "temperature_k": temperature,
                   "reference_temperature_k": np.arange(300., 1201., 50.),
                   "franta_temperature_k": np.arange(300., 501., 50.)}
        for key in ("p_abs_w", "p_gas_w", "p_bb_dipole_room_w", "p_bb_dipole_w",
                    "p_bb_mie_room_w", "p_bb_mie_w"):
            results[key] = np.array([0., 1., 2., 3.]) * 1e-9
        results["p_bb_mie_w"] *= 2  # Ensure the hot Mie line uses its own power array.
        results["p_bb_dipole_w"][-1] = np.nan  # An invalid extrapolation must leave a gap.
        results["p_abs_reference_w"] = np.full(19, 1e-9)
        results["p_bb_reference_w"] = np.full(19, 2e-9)
        results["p_bb_franta_w"] = np.arange(5.) * 1e-12
        return results

    def test_nine_series_use_full_paper_arrays_and_actual_franta_temperatures(self):
        results = self.fitted_results()
        fig, ax = plotting.plt.subplots()
        try:
            plotting.benchmark_power_lines(ax, results)
            self.assertEqual([line.get_label() for line in ax.lines], [
                "abs_tot Junn.", "BB Junn.", "abs_tot", "gas", "BB_dipole(T_0)",
                "BB_dipole(T)", "BB_Mie(T_0)", "BB_Mie(T)", "BB_Franta"])
            self.assertEqual([line.get_linestyle() for line in ax.lines],
                             ["--", "--", "-", "-", "-", "-", "-", "-", "None"])
            np.testing.assert_array_equal(ax.lines[0].get_xdata(), results["reference_temperature_k"])
            np.testing.assert_array_equal(ax.lines[-1].get_xdata(), results["franta_temperature_k"])
            np.testing.assert_array_equal(ax.lines[7].get_ydata()[1:], results["p_bb_mie_w"][1:])
            self.assertTrue(np.isnan(ax.lines[5].get_ydata()[-1]))
        finally:
            plotting.plt.close(fig)

    def test_eight_series_saved_result_does_not_invent_hot_mie(self):
        results = self.fitted_results()
        del results["p_bb_mie_w"]
        labels = []
        with patch.object(Figure, "savefig", lambda fig, *a, **k: labels.extend(
                line.get_label() for line in fig.axes[0].lines)):
            plotting.power_plot(results, "unused.png", 100)
        self.assertEqual(labels, ["abs_tot Junn.", "BB Junn.", "abs_tot", "gas",
                                 "BB_dipole(T_0)", "BB_dipole(T)", "BB_Mie(T_0)",
                                 "BB_Franta"])

    def test_old_benchmark_remains_replottable_without_new_arrays(self):
        results = {"kind": "junnemann", "temperature_k": np.array([300., 400., 500.]),
                   "reference_temperature_k": np.array([300., 400., 500.])}
        for key in ("p_abs_w", "p_gas_w", "p_bb_w", "p_bb_dipole_w",
                    "p_abs_reference_w", "p_bb_reference_w"):
            results[key] = np.array([0., 1., 2.]) * 1e-9
        labels = []
        with patch.object(Figure, "savefig", lambda fig, *a, **k: labels.extend(
                line.get_label() for line in fig.axes[0].lines)):
            plotting.power_plot(results, "unused.png", 100)
        self.assertIn("BB dipole", labels)
        self.assertIn("BB Mie", labels)
        self.assertNotIn("BB_Franta", labels)


if __name__ == "__main__":
    unittest.main()
