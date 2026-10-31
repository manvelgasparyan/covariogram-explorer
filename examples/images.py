"""Covariogram of a binary image, compared with the exact covariogram."""
import numpy as np
import covariogram_explorer as ce

R = ce.get_shape("reuleaux-triangle")
for pixel in [0.04, 0.02, 0.01, 0.005]:
    image, x, y = ce.rasterize(R, pixel)
    lx, ly, C = ce.covariogram_from_image(image, pixel)
    LX, LY = np.meshgrid(lx, ly)
    error = np.max(np.abs(C - R.covariogram(np.stack([LX, LY], axis=-1))))
    print(f"pixel {pixel:.3f}: image {image.shape[0]} x {image.shape[1]}, max |C - g| = {error:.2e}")
