import os
import pandas as pd
import numpy as np
import pytest
from fairmofsyncondition.crystal import analysis

def compute_size_from_xrd(filename):
    sizes = []
    strains = []
    df = pd.read_csv(filename)
    # Note: The columns in the CSV are assumed to be named 'Intensity' and 'Theta'
    two_theta = np.array(df['Intensity'])
    intensity = np.array(df['Theta'])
    # Uncomment to visualize the data if needed
    # import matplotlib.pyplot as plt
    # plt.plot(intensity, two_theta)
    # plt.show()

    fwhm_data, peak_positions = analysis.estimate_fwhm_from_pxrd(two_theta, intensity)
    for fwhm, center in zip(fwhm_data, peak_positions):
        theta = center / 2.0
        size, strain = analysis.compute_crystallite_size_and_strain(theta, fwhm)
        print(f"{center:10.2f}\t{fwhm:8.3f}\t{size:21.2f}\t{strain:10.4f}")
        sizes.append(size)
        strains.append(strain)
    av_size, av_strain = np.mean(sizes), np.mean(strains)
    modified_size = analysis.modified_scherrer_eq(fwhm_data, peak_positions)
    print("Average size:", av_size, "Average strain:", av_strain, "Modified Scherrer size:", modified_size)
    return av_size, av_strain, modified_size

def test_compute_size_from_xrd():
    # Path to the test CSV data
    test_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(test_dir, 'test_data/XRD_ZnO.csv')

    # Compute values using the provided function
    av_size, av_strain, modified_size = compute_size_from_xrd(filename)

    # Expected results
    expected_av_size = 15.878129241007354
    expected_av_strain = 0.006476021847988926
    expected_modified_size = 31.772996158046595

    # Check that computed values are within a small tolerance of expected values
    assert np.isclose(av_size, expected_av_size, rtol=1e-5), f"Expected average size {expected_av_size}, got {av_size}"
    assert np.isclose(av_strain, expected_av_strain, rtol=1e-5), f"Expected average strain {expected_av_strain}, got {av_strain}"
    assert np.isclose(modified_size, expected_modified_size, rtol=1e-5), f"Expected modified size {expected_modified_size}, got {modified_size}"

# Optionally, if you have test data for crystallinity analysis, you can add tests like the following:
#
# def test_crystallinity_methods():
#     crystal = analysis.Crystallinity(filename='test_data/AA.cif')
#     get_modified_scherrer = crystal.get_modified_scherrer()
#     av_size, av_str = crystal.get_average_size_and_strain()
#     crystallinity = crystal.compute_crystallinity_area_method()
#
#     assert np.isclose(get_modified_scherrer, 15.161275121187273, rtol=1e-5)
#     assert np.isclose(av_size, 15.006895146485515, rtol=1e-5)
#     assert np.isclose(av_str, 0.04141024664936405, rtol=1e-5)
#     assert np.isclose(crystallinity, 0.16170573852316103, rtol=1e-5)
