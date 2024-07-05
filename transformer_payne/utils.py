import numpy as np
import scipy
from scipy import interpolate, ndimage
import os
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def read_in_neural_network(path):
    """
    read in the weights and biases parameterizing a particular neural network.
    You can read in existing networks from the neural_nets/ directory, or you
    can train your own networks and edit this function to read them in.
    """

    # path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'neural_nets/NN_normalized_spectra.npz')
    tmp = np.load(path)
    w_array_0 = tmp["w_array_0"]
    w_array_1 = tmp["w_array_1"]
    w_array_2 = tmp["w_array_2"]
    b_array_0 = tmp["b_array_0"]
    b_array_1 = tmp["b_array_1"]
    b_array_2 = tmp["b_array_2"]
    x_min = tmp["x_min"]
    x_max = tmp["x_max"]
    NN_coeffs = (
        w_array_0,
        w_array_1,
        w_array_2,
        b_array_0,
        b_array_1,
        b_array_2,
        x_min,
        x_max,
    )
    tmp.close()
    return NN_coeffs


def load_wavelength_array():
    """
    read in the default wavelength grid onto which we interpolate all spectra
    """
    path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "other_data/apogee_wavelength.npz"
    )
    tmp = np.load(path)
    wavelength = tmp["wavelength"]
    tmp.close()
    return wavelength


# def load_apogee_mask():
#     '''
#     read in the pixel mask with which we will omit bad pixels during spectral fitting
#     The mask is made by comparing the tuned Kurucz models to the observed spectra from Arcturus
#     and the Sun from APOGEE. We mask out pixels that show more than 2% of deviations.
#     '''
# path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'other_data/apogee_mask.npz')
# tmp = np.load(path)
# mask = tmp['apogee_mask']
# tmp.close()
# return mask


def continuum_normalize_syn(
    wavelength: np.ndarray,
    flux: np.ndarray,
    L: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Parameters
    ----------
    wavelength : np.ndarray
        wave array.
    spec : np.ndarray
        flux array.
    L : int
        smooth window size.

    Returns
    -------
    np.ndarray, np.ndarray
        continuum-normalized flux, continuum flux
    """
    wave_model = np.arange(wavelength.min(), wavelength.max(), 2)

    if wave_model.max() < wavelength.max():
        wave_model = np.append(wave_model, wavelength.max())

    flux_resample = interpolate.interp1d(wavelength, flux)(wave_model)
    conti = ndimage.gaussian_filter1d(flux_resample, L)
    conti_resample = interpolate.interp1d(wave_model, conti)(wavelength)
    flux_sc = flux / conti_resample

    return flux_sc, conti_resample


def smooth_spec(
    flux: np.ndarray, ivar: np.ndarray, wavelength: np.ndarray, L: int = 50
) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    flux : np.ndarray
        _description_
    ivar : np.ndarray
        _description_
    wavelength : np.ndarray
        _description_
    L : int, optional
        _description_, by default 50

    Returns
    -------
    np.ndarray
        _description_
    """
    w = np.exp(-0.5 * (wavelength[:, None] - wavelength[None, :]) ** 2 / L**2)
    denominator = np.dot(ivar, w.T)
    numerator = np.dot(flux * ivar, w.T)
    bad_pixel = denominator == 0
    smoothed = np.zeros(numerator.shape)
    smoothed[~bad_pixel] = numerator[~bad_pixel] / denominator[~bad_pixel]
    return smoothed


def continuum_normalize_obs(
    flux: np.ndarray,
    ivar: np.ndarray,
    wavelength: np.ndarray,
    L: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """_summary_

    Parameters
    ----------
    np : _type_
        _description_
    np : _type_
        _description_
    flux : _type_, optional
        _description_, by default 50)->(np.ndarray

    Returns
    -------
    _type_
        _description_
    """
    wave_model = np.arange(wavelength.min(), wavelength.max(), 2)

    if wave_model.max() < wavelength.max():
        wave_model = np.append(wave_model, wavelength.max())

    flux_resample = interpolate.interp1d(wavelength, flux)(wave_model)
    var_resample = interpolate.interp1d(wavelength, 1 / ivar)(wave_model)

    smoothed_spec = smooth_spec(flux_resample, 1 / var_resample, wave_model, L)
    norm_flux = flux_resample / smoothed_spec
    norm_ivar = smoothed_spec * (1 / var_resample) * smoothed_spec

    smoothed_spec = interpolate.interp1d(wave_model, smoothed_spec)(wavelength)
    norm_flux = interpolate.interp1d(wave_model, norm_flux)(wavelength)
    norm_ivar = 1 / interpolate.interp1d(wave_model, 1 / norm_ivar)(wavelength)

    bad_pixel = ~np.isfinite(norm_flux)
    norm_flux[bad_pixel] = 1.0
    norm_ivar[bad_pixel] = 0.0
    return smoothed_spec, norm_flux, norm_ivar


def load_cannon_contpixels():
    """
    read in the default list of APOGEE pixels for continuum fitting.
    """
    # path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'other_data/cannon_cont_pixels_apogee.npz')
    # tmp = np.load(path)
    # pixels_cannon = tmp['pixels_cannon']
    # tmp.close()
    # return pixels_cannon


class ReadData:
    def __init__(self, fluxpath, parampath, y_norm):
        self.fluxpath = fluxpath
        self.parampath = parampath
        self.y_norm = y_norm

    def readData(self):
        f = pd.read_csv(self.fluxpath, header=None)
        X = f  # .iloc[:, :] # 这是流量数据
        # p = f.iloc[:, :7]
        p = pd.read_csv(self.parampath)
        p.columns = ["Teff", "Logg", "FeH", "AFe", "CFe", "NFe", "OFe"]
        X = X[p.Teff > 3500]
        p = p[p.Teff > 3500]
        p["Teff"] /= 1000
        y = p  # 0 teff	1 logg	2 feh	3 afe	4 cfe	5 nfe	6 ofe   # 每次训练一个参数
        X = np.array(X).astype(np.float32)
        y = np.array(y).astype(np.float32)

        if self.y_norm:
            norm = MinMaxScaler()
            y = norm.fit_transform(y)
            Xtrain, Xtest, ytrain, ytest = train_test_split(
                y, X, test_size=0.3, random_state=420
            )
            return Xtrain, Xtest, ytrain, ytest, norm
        else:
            Xtrain, Xtest, ytrain, ytest = train_test_split(
                y, X, test_size=0.3, random_state=420
            )

        return Xtrain, Xtest, ytrain, ytest


def doppler_shift(wave, flux, dv, wave_obs):
    """
    dv is in km/s
    We use the convention where a positive dv means the object is moving away.

    This linear interpolation is actually not that accurate, but is fine if you
    only care about accuracy to the level of a few tenths of a km/s. If you care
    about better accuracy, you can do better with spline interpolation.
    """
    c = 2.99792458e5  # km/s
    doppler_factor = np.sqrt((1 - dv / c) / (1 + dv / c))
    new_wavelength = wave_obs * doppler_factor
    new_flux = np.interp(new_wavelength, wave, flux)
    return new_flux


def get_apogee_continuum(spec, spec_err=None, cont_pixels=None):
    """
    continuum normalize spectrum.
    pixels with large uncertainty are weighted less in the fit.
    """
    if cont_pixels is None:
        cont_pixels = load_cannon_contpixels()
    cont = np.empty_like(spec)

    wavelength = load_wavelength_array()

    deg = 4

    # if we haven't given any uncertainties, just assume they're the same everywhere.
    if spec_err is None:
        spec_err = np.zeros(spec.shape[0]) + 0.0001

    # Rescale wavelengths
    bluewav = 2 * np.arange(2920) / 2919 - 1
    greenwav = 2 * np.arange(2400) / 2399 - 1
    redwav = 2 * np.arange(1894) / 1893 - 1

    blue_pixels = cont_pixels[:2920]
    green_pixels = cont_pixels[2920:5320]
    red_pixels = cont_pixels[5320:]

    # blue
    cont[:2920] = _fit_cannonpixels(
        bluewav, spec[:2920], spec_err[:2920], deg, blue_pixels
    )
    # green
    cont[2920:5320] = _fit_cannonpixels(
        greenwav, spec[2920:5320], spec_err[2920:5320], deg, green_pixels
    )
    # red
    cont[5320:] = _fit_cannonpixels(
        redwav, spec[5320:], spec_err[5320:], deg, red_pixels
    )
    return cont


def _fit_cannonpixels(wav, spec, specerr, deg, cont_pixels):
    """
    Fit the continuum to a set of continuum pixels
    helper function for get_apogee_continuum()
    """
    chpoly = np.polynomial.Chebyshev.fit(
        wav[cont_pixels], spec[cont_pixels], deg, w=1.0 / specerr[cont_pixels]
    )
    return chpoly(wav)


def whitten_wavelength(wavelength):
    """
    normalize the wavelength of each order to facilitate the polynomial continuum fit
    """

    wavelength_normalized = np.zeros(wavelength.shape)
    for k in range(wavelength.shape[0]):
        mean_wave = np.mean(wavelength[k, :])
        wavelength_normalized[k, :] = (wavelength[k, :] - mean_wave) / mean_wave
    return wavelength_normalized


def transform_coefficients(popt, x_min, x_max, order):
    """
    Transform coefficients into human-readable
    """
    popt_new = popt  # .copy()
    popt_new[:order] = (popt_new[:order] + 0.5) * (x_max - x_min) + x_min
    popt_new[0] = popt_new[0]  # * 1000.
    popt_new[-1] = popt_new[-1] * 100.0
    return popt_new


def normalize_stellar_parameter_labels(labels, NN_coeffs=None):
    """
    Turn physical stellar parameter values into normalized values.
    Teff (K), logg (dex), FeH (solar), aFe (solar)
    """
    # assert len(labels) == 4, "Input Teff, logg, FeH, aFe"
    # Teff, logg, FeH, aFe = labels
    labels = np.ravel(labels)
    # labels[0] = labels[0] / 1000.

    if NN_coeffs is None:
        # NN_coeffs = read_in_neural_network(path='/home/wangr/code/csst_ms_sls_stellar_parameters/model_save/NN_normalized_spectra_h5_1026.npz')
        # w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = NN_coeffs
        if len(labels) == 3:
            x_min = np.array([3.1e03, -0.5, -4.0])
            x_max = np.array([9.8e03, 6.0, 0.5])
        if len(labels) == 7:
            NN_coeffs = read_in_neural_network(
                path="/data_18/code/csst_ms_sls_stellar_parameter/model_save/NN_normalized_spectra_h5_1026.npz"
            )
            x_min, x_max = NN_coeffs[-2], NN_coeffs[-1]
    new_labels = (labels - x_min) / (x_max - x_min)  # - 0.5
    # assert np.all(new_labels >= -0.5), new_labels
    # assert np.all(new_labels <= 0.5), new_labels
    return new_labels


def denormalize_stellar_parameter_labels(labels, x_min, x_max):
    """
    Turn normalized stellar parameter values into physical values.
    Teff (K), logg (dex), FeH (solar),
    """
    # assert len(labels) == 3, "Input Teff, logg, FeH"
    labels = np.ravel(labels)
    # w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = NN_coeffs
    new_labels = labels * (x_max - x_min) + x_min  # +0.5
    # new_labels[0] = new_labels[0] * 1000
    return new_labels
