import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fairmofsyncondition.crystal import analysis

def compute_size_from_xrd(filename):
    sizes = []
    strains = []
    df = pd.read_csv(filename)
    two_theta = np.array(list(df['Intensity']))
    intensity = np.array(list(df['Theta']))
    # Uncomment the lines below to visualize the data
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
    print(av_size, av_strain, modified_size)
    assert av_size == 15.878129241007354
    assert av_strain == 0.006476021847988926
    assert modified_size == 31.772996158046595

filename = './test_data/XRD_ZnO.csv'
compute_size_from_xrd(filename)

crystal = analysis.Crystallinity(filenames='test_data/AA.cif')
get_modified_scherrer = crystal.get_modified_scherrer()
av_size, av_str = crystal.get_average_size_and_strain()
crystallinity = crystal.compute_crystallinity_area_method()

assert get_modified_scherrer == 15.161275121187273
assert av_size == 15.006895146485515
assert av_str == 0.04141024664936405
assert crystallinity == 0.16170573852316103
