#!/usr/bin/python
from __future__ import print_function
__author__ = "Dr. Dinga Wonanke"
__status__ = "production"

##############################################################################
# fairmofsyncondition is a machine learning package for predicting the        #
# synthesis condition of the crystal structures of MOFs. It is also intended  #
# for predicting all MOFs the can be generated from a given set of conditions #
# In addition the package also predicts the stability of MOFs, compute their  #
# their PXRD and crystallite sizes. This package is part of our effort to     #
# to accelerate the discovery and optimization of the synthesises of novel    #
# high performing MOFs. This package is being developed by Dr Dinga Wonanke   #
# as part of hos MSCA post doctoral fellowship at TU Dresden.                 #
#                                                                             #
###############################################################################
from ase.io import read
import warnings
import numpy as np
from scipy.signal import find_peaks
from lmfit.models import GaussianModel, PolynomialModel  #
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.diffraction.neutron import NDCalculator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from scipy.optimize import curve_fit
from scipy.special import erf, wofz
from scipy.stats import linregress
from scipy.optimize import OptimizeWarning
warnings.filterwarnings("ignore", category=OptimizeWarning)


class Crystallinity(object):
    def __init__(self,
                 ase_atoms=None,
                 filename=None,
                 wavelength='CuKa',
                 diffraction_type="PXRD"
                 ):
        """
        Compute the crystallinity of a crystal structure.

        Parameters:
            atoms (ase.Atoms): Crystal structure.

        Returns:
            float: Crystallinity.
        """
        if ase_atoms is not None:
            self.ase_atoms = ase_atoms
            self.structure = AseAtomsAdaptor.get_structure(self.ase_atoms)
        else:
            with open(filename, "rt", encoding="utf-8") as f:
                cif_data = f.read()
            self.structure = Structure.from_str(cif_data, fmt="cif")


        self.wavelength = wavelength
        self.diffraction_type = diffraction_type

    def find_diffraction_pattern(self):
        """
        Compute the diffraction pattern of the crystal structure.

        Returns:
            list: Diffraction pattern.
        """
        if self.diffraction_type == "PXRD":
            xrd = XRDCalculator(wavelength=self.wavelength)
            pattern = xrd.get_pattern(self.structure, two_theta_range=(0, 100))
        elif self.diffraction_type == "ND":
            nd = NDCalculator(wavelength=self.wavelength, two_theta_range=(0, 100))
            pattern = nd.get_pattern(self.structure)
        else:
            raise ValueError("""Invalid diffraction type.
                             Choose between PXRD and ND""")

        return pattern

    def get_pattern(self):
        """
        """
        pattern = self.find_diffraction_pattern()
        two_theta = np.array(pattern.x, dtype=np.float32)
        intensity = np.array(pattern.y, dtype=np.float32)
        return two_theta, intensity


    def compute_crystallinity_area_method(self,
                                          peak_prominence=1.0,
                                          height_threshold=0.01,
                                          polynomial_order=2):
        """
        Estimate the Crystallinity Index (CI) via the area method using
        peak deconvolution and background (amorphous) modeling.

        Steps:
        1. Obtain diffraction pattern (2θ vs intensity).
        2. Model the amorphous background with a polynomial.
        3. Identify crystalline peaks (using find_peaks).
        4. Build a composite model: polynomial background + one Gaussian per peak.
        5. Fit the composite model to the data.
        6. Integrate the total area under the fitted curve.
        7. Integrate the area under just the background.
        8. Crystallinity Index = (Total area - background area) / (Total area).

        Parameters:
        peak_prominence (float): Sensitivity for peak detection.
        height_threshold (float): Minimum height threshold as a fraction of max intensity.
        polynomial_order (int): Degree of polynomial for background modeling.

        Returns:
        crystallinity_index (float): Fraction in [0,1] representing the crystalline area.
        """
        # 1. Get the raw diffraction data.
        # Assume self.get_pattern() returns (two_theta, intensity)
        two_theta, intensity = self.get_pattern()

        # Ensure intensities are nonnegative.
        intensity = np.clip(intensity, 0, None)

        # 2. (Optional) Crop the data range if desired.
        # two_theta, intensity = self._crop_data(two_theta, intensity, lower=5, upper=80)

        # To help the polynomial fit, shift two_theta by subtracting its mean.
        two_theta_mean = np.mean(two_theta)
        two_theta_shifted = two_theta - two_theta_mean  # Now values are centered near zero.

        # 3. Identify peaks.
        # Use an optimal prominence if available, otherwise use peak_prominence.
        try:
            optimal_prominence, _ = compute_optimal_prominence(intensity)
        except Exception:
            optimal_prominence = peak_prominence

        peaks, properties = find_peaks(intensity, prominence=optimal_prominence,
                                    height=height_threshold * np.max(intensity))
        peak_centers = two_theta[peaks]

        # 4. Build composite model: background (polynomial) + sum of Gaussians.
        # For the background, use the shifted two_theta values.
        background_model = PolynomialModel(prefix='poly_', degree=polynomial_order, independent_vars=['x'])
        model = background_model
        pars = background_model.make_params(x=two_theta_shifted)

        # Set initial guess for the background constant as the minimum intensity.
        pars['poly_c0'].set(value=np.min(intensity), min=0, max=np.max(intensity))
        # For higher-order terms, set initial guesses to 0 with narrow bounds.
        for i in range(1, polynomial_order+1):
            key = f'poly_c{i}'
            pars[key].set(value=0, min=-np.max(intensity)*1e-3, max=np.max(intensity)*1e-3)

        # Add one Gaussian per detected peak.
        for i, cen in enumerate(peak_centers):
            gauss = GaussianModel(prefix=f'g{i}_', independent_vars=['x'])
            model = model + gauss

            init_pars = gauss.make_params(x=two_theta_shifted)
            # Adjust the center: shift by the same amount.
            shifted_cen = cen - two_theta_mean
            init_pars[f'g{i}_center'].set(value=shifted_cen, min=shifted_cen-0.5, max=shifted_cen+0.5)
            init_pars[f'g{i}_sigma'].set(value=0.2, min=0.01, max=2.0)
            # Use the intensity at the detected peak as the amplitude.
            peak_height = intensity[peaks[i]]
            init_pars[f'g{i}_amplitude'].set(value=peak_height, min=0)

            pars.update(init_pars)

        # 5. Fit the composite model.
        try:
            result = model.fit(intensity, pars, x=two_theta_shifted)
        except Exception as e:
            raise ValueError(f"Model fitting failed: {e}")

        # Check if the fitted model returns any NaN values.
        fitted_intensity = result.eval(x=two_theta_shifted)
        if np.any(np.isnan(fitted_intensity)):
            raise ValueError("""Fitted model produced NaN values.
                             Check initial guesses or parameter bounds.
                             """
                             )

        # 6. Compute total area under the fitted composite model (in original two_theta).
        # Here we evaluate the fitted function on the original two_theta (undo the shift).
        total_area = np.trapz(result.eval(x=two_theta - two_theta_mean), two_theta)

        # 7. Compute the background area using the polynomial model.
        poly_pars = result.params.copy()
        # Zero out all Gaussian parameters.
        for name in poly_pars:
            if name.startswith('g'):
                poly_pars[name].set(value=0)
        background_only = background_model.eval(params=poly_pars, x=two_theta_shifted)
        background_area = np.trapz(background_only, two_theta)

        # 8. Crystalline area is the total area minus the background area.
        crystalline_area = total_area - background_area
        crystallinity_index = crystalline_area / total_area if total_area != 0 else 0.0

        return crystallinity_index

    def _crop_data(self, x, y, lower=5, upper=80):
        """Helper to crop data between lower and upper 2theta values."""
        mask = (x >= lower) & (x <= upper)
        return x[mask], y[mask]

    def get_size_and_strain(self, wavelength=0.15406, k_value=0.9):
        """

        """
        sizes = []
        strains = []
        two_theta, intensity = self.get_pattern()
        fwhm_data, peak_positions = estimate_fwhm_from_pxrd(two_theta, intensity)
        for fwhm, center in zip(fwhm_data, peak_positions):
            theta = center / 2.0
            size, strain = compute_crystallite_size_and_strain(theta, fwhm, wavelength=wavelength, k_value=k_value)
            sizes.append(size)
            strains.append(strain)
        return sizes, strains

    def get_average_size_and_strain(self,
                                    wavelength=0.15406,
                                    k_value=0.9
                                    ):
        """
        """
        sizes, strains = self.get_size_and_strain(wavelength=wavelength, k_value=k_value)
        return np.mean(sizes), np.mean(strains)

    def get_modified_scherrer(self,
                              wavelength=0.15406,
                              k_value=0.9
                              ):
        two_theta, intensity = self.get_pattern()
        fwhms, peak_positions = estimate_fwhm_from_pxrd(two_theta, intensity)
        size = modified_scherrer_eq(fwhms, peak_positions, wavelength=wavelength, k_value=k_value)
        return size



def voigt_profile(x,
                  amplitude,
                  center,
                  sigma,
                  gamma):
    """
    Voigt profile function using the Faddeeva function.
    """
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


def compute_optimal_prominence(intensity,
                               min_prominence=0.01,
                               max_prominence=1.0,
                               tolerance=3,
                               step=0.01):
    """
    Automatically determines the optimal prominence for peak detection.
    """
    max_prominence_avg = 0
    optimal_prominence = min_prominence
    no_improvement_count = 0

    if np.max(intensity) == 0:
        raise ValueError("Maximum intensity is zero, unable to normalize.")

    norm_intensity = intensity / np.max(intensity)

    for current_prominence in np.arange(min_prominence, max_prominence, step):
        peaks, properties = find_peaks(norm_intensity, prominence=current_prominence)
        prominences = properties["prominences"]
        if len(prominences) > 0:
            current_avg_prominence = np.mean(prominences)
            if current_avg_prominence > max_prominence_avg:
                max_prominence_avg = current_avg_prominence
                optimal_prominence = current_prominence
                no_improvement_count = 0
            else:
                no_improvement_count += 1
        if no_improvement_count >= tolerance:
            break
    return optimal_prominence, peaks


def estimate_fwhm_from_pxrd(two_theta,
                            intensities,
                            height_threshold=0.01
                            ):
    """
    Estimate the FWHM values and peak positions from PXRD data.

    This function detects peaks, then for each peak it extracts the region and fits a Voigt
    profile to determine an accurate FWHM.

    Parameters:
        two_theta : np.array
            Array of 2θ values (in degrees).
        intensities : np.array
            Array of intensity values corresponding to the 2θ values.
        height_threshold : float, optional
            Minimum height threshold as a fraction of the maximum intensity.

    Returns:
        fwhm_data : np.array
            Estimated FWHM (in degrees) for each detected peak.
        peak_positions : np.array
            2θ positions (in degrees) of each detected peak.
    """
    # Determine an optimal prominence for the data
    optimal_prominence, _ = compute_optimal_prominence(intensities)
    peaks, properties = find_peaks(intensities, prominence=optimal_prominence,
                                   height=height_threshold * np.max(intensities))
    fwhm_data = []
    peak_positions = []

    for i, peak in enumerate(peaks):
        left_base = properties["left_bases"][i]
        right_base = properties["right_bases"][i]
        peak_region_x = two_theta[left_base:right_base]
        peak_region_y = intensities[left_base:right_base]

        amplitude = peak_region_y.max()
        center = two_theta[peak]
        half_max = amplitude / 2
        # Find indices where the intensity is close to half maximum
        closest_to_half_max = np.where(np.isclose(peak_region_y, half_max, atol=0.1 * half_max))[0]

        if len(closest_to_half_max) >= 2:
            estimated_fwhm = peak_region_x[closest_to_half_max[-1]] - peak_region_x[closest_to_half_max[0]]
            sigma = estimated_fwhm / (2 * np.sqrt(2 * np.log(2)))
        else:
            estimated_fwhm = (peak_region_x[-1] - peak_region_x[0]) / 2
            sigma = estimated_fwhm / (2 * np.sqrt(2 * np.log(2)))

        gamma = estimated_fwhm / 2 if 'estimated_fwhm' in locals() else 0.1

        try:
            popt, _ = curve_fit(voigt_profile, peak_region_x, peak_region_y,
                                p0=[amplitude, center, sigma, gamma],
                                method='trf', max_nfev=5000,
                                bounds=([0, center - 2, 0, 0], [np.inf, center + 2, np.inf, np.inf]))
            fitted_sigma = popt[2]
            fitted_gamma = popt[3]
            # Compute FWHM for Voigt profile using an approximate relation
            fwhm = 0.5346 * (2 * fitted_gamma) + np.sqrt(0.2166 * (2 * fitted_gamma)**2 + (2 * fitted_sigma)**2)
        except RuntimeError:
            fwhm = estimated_fwhm

        fwhm_data.append(fwhm)
        peak_positions.append(center)

    return np.array(fwhm_data), np.array(peak_positions)

def compute_crystallite_size_and_strain(theta_deg, fwhm_deg, wavelength=0.15406, k_value=0.9):
    """
    Compute the crystallite size (D) and microstrain (ε) for a given peak.

    Parameters:
        theta_deg : float
            Bragg angle (θ in degrees, half of the 2θ value).
        fwhm_deg : float
            FWHM (in degrees, as estimated from the PXRD pattern).
        wavelength : float
            X-ray wavelength in nm.
        k : float
            Scherrer constant.

    Returns:
        D : float
            Crystallite size in nm.
        strain : float
            Microstrain (dimensionless).
    """
    theta = np.deg2rad(theta_deg)
    beta = np.deg2rad(fwhm_deg)
    size = (k_value * wavelength) / (beta * np.cos(theta))
    strain = beta / (4 * np.tan(theta))
    return size, strain


def modified_scherrer_eq(fwhms,
                         two_theta_position,
                         wavelength=0.15406,
                         k_value=0.9,
                         fwhm_in_degrees=True
                         ):
    """
    Computes the crystallite size using the modified Scherrer equation.

    The modified Scherrer equation is based on:

        Kλ / L = β cosθ

    Taking logarithms gives:

        ln(β) = ln(1/cosθ) + ln(Kλ/L)

    A linear regression of ln(β) versus ln(1/cosθ) provides an intercept = ln(Kλ/L),
    from which the crystallite size L is calculated as:

        L = (K * wavelength) / exp(intercept)

    Parameters:
    ----------
    fwhms : array-like
        Array of FWHM values for the peaks. If these are in degrees,
        set fwhm_in_degrees=True (default).
    two_theta_position : array-like
        Array of 2θ values (in degrees) for the peaks.
    wavelength : float, optional
        X-ray wavelength in nm (default: 0.15406 nm).
    k_value : float, optional
        Shape factor (default: 0.9).
    fwhm_in_degrees : bool, optional
        If True, converts fwhms from degrees to radians (default: True).

    Returns:
    -------
    crystallite_size_modified : float
        Calculated crystallite size (L) in nm.
    """
    # Convert inputs to numpy arrays
    fwhms = np.array(fwhms)
    two_theta_position = np.array(two_theta_position)

    # Convert FWHM to radians if provided in degrees
    if fwhm_in_degrees:
        fwhms = np.radians(fwhms)

    # Calculate ln(β) where β is in radians
    ln_beta = np.log(fwhms)

    # Compute θ (half of 2θ) in radians
    theta = np.radians(two_theta_position) / 2.0
    # Calculate ln(1/cosθ)
    ln_1_cos_theta = np.log(1 / np.cos(theta))

    # Perform linear regression: ln(β) = m * ln(1/cosθ) + intercept
    slope, intercept, r_value, p_value, std_err = linregress(ln_1_cos_theta, ln_beta)

    # The intercept is ln(K * wavelength / L), so:
    # L = (K * wavelength) / exp(intercept)
    crystallite_size_modified = (k_value * wavelength) / np.exp(intercept)

    return crystallite_size_modified

