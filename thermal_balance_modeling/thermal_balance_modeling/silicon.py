"""Silicon absorption models; temperatures K, lengths m, densities m^-3.

Sources: Sinclair et al., Phys. Rev. Applied 11, 044084 (2019), Eqs. 3-5;
Sturm & Reaves, IEEE Trans. Electron Devices 39, 81 (1992), Eqs. 7-13;
Arora et al., IEEE Trans. Electron Devices 29, 292 (1982), Eqs. 8, 13;
Junnemann et al., Phys. Rev. B 111, 195304 (2025), Fig. 4.
These are explicit model variants, not a claim to reproduce unpublished code.
Sinclair's fit was measured at 5.5-300 K; its high-temperature use extrapolates.
"""

import numpy as np
from scipy.constants import Boltzmann, c, e, epsilon_0, h, m_e
from scipy.special import lambertw

K_B_EV = Boltzmann / e
HC_EV_M = h * c / e
ALPHA_0 = 4.3e-4  # Junnemann's intrinsic laser absorption: 4.3e-6 cm^-1.
PHONON_K = np.array([212.0, 670.0])  # Transverse acoustic / optical branches.
TPA_WEIGHTS = np.array([0.233, 2.138])  # cm / (GW eV^(7/2)); Sinclair fit.
DRUDE_INDEX = 3.48  # Sinclair Appendix A: fixed background index at telecom lambda.
DRUDE_MASSES = m_e * np.array([0.3, 0.4])  # Electron / hole conductivity masses.


def _positive(value, name):
    """Convert a finite positive scalar/array to floating-point NumPy data."""
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return array


def _phonon_occupation(temperature_k):
    """Return Bose occupations for the two Si phonon branches without overflow."""
    exponent = -PHONON_K / np.asarray(temperature_k)[..., None]
    return np.exp(exponent) / (-np.expm1(exponent))


def _photon_energy(wavelength_m, photon_energy_ev):
    """Use hc/lambda unless a benchmark explicitly supplies its quoted energy."""
    wavelength = _positive(wavelength_m, "wavelength_m")
    if photon_energy_ev is None:
        return HC_EV_M / wavelength
    return _positive(photon_energy_ev, "photon_energy_ev")


def bandgap_energy(temperature_k, model="optical"):
    """Return the selected empirical Si gap in eV (different fits are intentional)."""
    temperature = _positive(temperature_k, "temperature_k")
    if model == "optical":  # Sturm optical edge includes the exciton correction.
        gap = 1.155 - 4.73e-4 * temperature**2 / (635.0 + temperature)
    elif model == "sinclair":
        gap = 1.156 - 7.021e-4 * temperature**2 / (1108.0 + temperature)
    else:
        raise ValueError("gap model must be 'optical' or 'sinclair'")
    if np.any(gap <= 0):
        raise ValueError(f"{model} gap fit is nonpositive at this temperature")
    return gap


def tpa_beta(temperature_k, variant="printed", wavelength_m=1.55e-6,
             photon_energy_ev=None):
    """Return TPA beta [m/W]; printed and physically paired Sinclair variants.

    'printed' preserves the signs/populations in the arXiv preprint equations.
    'physical' pairs phonon emission with N+1 and absorption with N, and enforces
    energy thresholds. It is a model correction, not a new experimental fit.
    Photon energy defaults to hc/lambda; use photon_energy_ev=0.797 explicitly
    to evaluate the quoted source energy without silently altering hc/lambda.
    """
    temperature = _positive(temperature_k, "temperature_k")
    photon = _photon_energy(wavelength_m, photon_energy_ev)
    gap = bandgap_energy(temperature, "sinclair")
    excess = (2.0 * photon - gap)[..., None]
    phonon = K_B_EV * PHONON_K
    occupation = _phonon_occupation(temperature)
    if variant == "printed":
        terms = occupation * (excess - phonon)**2
        terms += (occupation + 1.0) * (excess + phonon)**2
    elif variant == "physical":
        terms = (occupation + 1.0) * np.maximum(excess - phonon, 0.0)**2
        terms += occupation * np.maximum(excess + phonon, 0.0)**2
    else:
        raise ValueError("TPA variant must be 'printed' or 'physical'")
    return 1e-11 * gap**1.5 * np.sum(TPA_WEIGHTS * terms, axis=-1)


def bandgap_absorption(temperature_k, wavelength_m=1.55e-6, photon_energy_ev=None):
    """Return Sturm's phonon-assisted band-edge absorption [m^-1], Eqs. 7-10."""
    temperature = _positive(temperature_k, "temperature_k")
    photon = _photon_energy(wavelength_m, photon_energy_ev)
    gap = bandgap_energy(temperature, "optical")
    occupation = _phonon_occupation(temperature)
    energy = (photon - gap)[..., None]
    phonon = K_B_EV * PHONON_K
    absorption = np.zeros(np.broadcast_shapes(temperature.shape, photon.shape))
    for sign, population in ((-1.0, occupation + 1.0), (1.0, occupation)):
        excess = np.maximum(energy + sign * phonon, 0.0)
        # Energies eV; fitted branch amplitudes produce absorption in cm^-1.
        branch = np.array([0.504, 18.08]) * np.sqrt(excess)
        branch += np.array([392.0, 5760.0]) * np.maximum(excess - 0.0055, 0.0)**2
        absorption += np.sum(population * branch, axis=-1)
    return 100.0 * absorption  # cm^-1 -> m^-1.


def intrinsic_carriers(temperature_k, model="uncorrected"):
    """Return thermal electron density = hole density [m^-3].

    'uncorrected': Sturm's explicit density with band-gap narrowing set to zero.
    'corrected': low-density root of Sturm's implicit narrowing relation. The
    latter has no real solution above approximately 1633.84 K and raises there.
    It is a diagnostic alternative, not a silent extrapolation or cutoff.
    """
    temperature = _positive(temperature_k, "temperature_k")
    density = 3.87e22 * temperature**1.5 * np.exp(-0.605 / (K_B_EV * temperature))
    if model == "uncorrected":
        return density
    if model != "corrected":
        raise ValueError("carrier-density model must be 'uncorrected' or 'corrected'")
    # Sturm uses n in cm^-3: Delta E = -7.1e-10*sqrt(n/T) eV.
    a = 7.1e-10 / (K_B_EV * temperature**1.5)
    argument = a * np.sqrt(density / 1e6)
    if np.any(argument > 2.0 / np.e):
        raise ValueError("Sturm's corrected carrier density has no real root here (T > ~1633.84 K)")
    ratio = np.ones_like(argument)
    mask = argument > 0.0
    root_argument = np.maximum(-argument[mask] / 2.0, np.nextafter(-1.0 / np.e, 0.0))
    ratio[mask] = -2.0 * lambertw(root_argument, k=0).real / argument[mask]
    return density * ratio**2  # Principal Lambert-W branch is the low-density root.


def carrier_mobilities(temperature_k):
    """Arora electron/hole mobilities [m^2/(V s)], last axis has length two.

    Ionized-impurity density is set to zero; intrinsic carrier density is NOT
    substituted for dopant density. Arora's 250-500 K validation range does not
    cover the hot intrinsic plasma considered here. This extrapolation neglects
    additional carrier-carrier scattering and is an illustrative alternative.
    """
    temperature = _positive(temperature_k, "temperature_k")[..., None]
    mobility = np.array([88.0, 54.3]) * (temperature / 300.0)**-0.57
    mobility += np.array([7.4e8, 1.36e8]) * temperature**np.array([-2.33, -2.23])
    return 1e-4 * mobility  # cm^2/(V s) -> m^2/(V s).


def drude_cross_section(temperature_k, wavelength_m=1.55e-6):
    """Return summed electron/hole Drude cross section [m^2] for equal densities.

    Finite damping is retained. The conversion to absorption assumes weak loss
    and fixed background index; this is not a complete complex-index IR model.
    For omega >> damping it reduces to Sinclair's Appendix-A expression.
    """
    wavelength = _positive(wavelength_m, "wavelength_m")
    damping = e / (DRUDE_MASSES * carrier_mobilities(temperature_k))
    omega = 2.0 * np.pi * c / wavelength
    cross_section = e**2 * damping / (epsilon_0 * c * DRUDE_INDEX * DRUDE_MASSES)
    cross_section /= omega[..., None]**2 + damping**2
    return np.sum(cross_section, axis=-1)


def fca_absorption(temperature_k, model="empirical", density_model="uncorrected",
                   wavelength_m=1.55e-6):
    """Return absorption by thermally generated electrons and holes [m^-1]."""
    temperature = _positive(temperature_k, "temperature_k")
    wavelength = _positive(wavelength_m, "wavelength_m")
    density = intrinsic_carriers(temperature, density_model)
    if model == "empirical":
        # Sturm's electron/hole cross sections; SI lambda gives sigma in m^2.
        cross_section = (1.01e-12 + 0.51e-12) * temperature * wavelength**2
    elif model == "drude":
        cross_section = drude_cross_section(temperature, wavelength)
    else:
        raise ValueError("FCA model must be 'empirical' or 'drude'")
    return density * cross_section


def silicon_terms(temperature_k, intensity_w_m2, tpa="printed", fca="empirical",
                  density_model="uncorrected", wavelength_m=1.55e-6,
                  photon_energy_ev=None):
    """Return alpha0, tpa, fca, bga [m^-1] at the specified local intensity.

    Only thermal equilibrium carriers are included: TPA-generated carriers and
    their subsequent absorption require a separate lifetime/recombination model.
    """
    temperature = _positive(temperature_k, "temperature_k")
    intensity = np.asarray(intensity_w_m2, dtype=float)
    if np.any(~np.isfinite(intensity)) or np.any(intensity < 0):
        raise ValueError("intensity_w_m2 must be finite and nonnegative")
    temperature, intensity = np.broadcast_arrays(temperature, intensity)
    return {
        "alpha0": np.full(temperature.shape, ALPHA_0),
        "tpa": tpa_beta(temperature, tpa, wavelength_m, photon_energy_ev) * intensity,
        "fca": fca_absorption(temperature, fca, density_model, wavelength_m),
        "bga": bandgap_absorption(temperature, wavelength_m, photon_energy_ev),
    }
