"""Chord length distributions of convex shapes, from the covariogram."""
import numpy as np
import covariogram_explorer as ce

R = ce.get_shape("reuleaux-triangle")
r = np.array([0.2, 0.5, 0.8, 0.95])
print("side direction:  ", np.round(ce.chord_length_cdf(R, r, direction=np.pi / 3), 4))
print("vertex direction:", np.round(ce.chord_length_cdf(R, r, direction=np.pi / 2), 4))
print("isotropic lines: ", np.round(ce.chord_length_cdf(R, r), 4))
print("mean chord length of isotropic lines:", ce.mean_chord_length(R))
