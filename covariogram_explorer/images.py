"""Covariogram (set covariance) of binary images, and digitisation of shapes.

The covariogram of the set of foreground pixels of a binary image is computed
exactly (up to rounding) with the fast Fourier transform. Comparing it with the
exact covariogram of a shape measures the error caused by digitisation."""

import numpy as np


def rasterize(shape, pixel_size):
    """Binary image of a shape: a pixel is foreground if its centre lies in K.

    Returns (image, x, y), where x and y are the coordinates of the pixel
    centres, so that image[i, j] is the pixel centred at (x[j], y[i])."""
    lo, hi = shape.bounding_box()
    lo = np.floor(lo / pixel_size - 1) * pixel_size
    hi = np.ceil(hi / pixel_size + 1) * pixel_size
    x = np.arange(lo[0], hi[0] + pixel_size / 2, pixel_size)
    y = np.arange(lo[1], hi[1] + pixel_size / 2, pixel_size)
    X, Y = np.meshgrid(x, y)
    return shape.contains(np.stack([X, Y], axis=-1)), x, y


def covariogram_from_image(image, pixel_size=1.0):
    """Covariogram of the foreground of a binary image.

    Returns (lags_x, lags_y, C), where C[i, j] is the area of the intersection
    of the foreground with its translate by (lags_x[j], lags_y[i]), each pixel
    having area pixel_size**2. The values are exact: the counts computed with
    the FFT are rounded to integers."""
    f = np.asarray(image, dtype=bool).astype(float)
    ny, nx = f.shape
    F = np.fft.rfft2(f, s=(2 * ny, 2 * nx))
    counts = np.fft.irfft2(F * np.conj(F), s=(2 * ny, 2 * nx))
    counts = np.maximum(np.rint(np.fft.fftshift(counts)), 0.0)
    lags_x = (np.arange(2 * nx) - nx) * pixel_size
    lags_y = (np.arange(2 * ny) - ny) * pixel_size
    return lags_x, lags_y, counts * pixel_size ** 2
