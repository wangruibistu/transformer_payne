import numpy as np
from scipy.optimize import curve_fit
from scipy import ndimage
from transformer_payne_spec_model import load_weights, get_spectrum_from_multiheadPayne
import utils

# model_path = "/home/wangr/code/spec_stellar_parameter/csst_parameter/csst_transformer_stellar_params/checkpoints/model_weights.pt"
# weights = load_weights(model_path)
wavelength_multiheadPayne = np.arange(2500, 10000, 2)


def fit_normalized_spectrum_single_star_model_curvefit(
    spec_flux,
    spec_err,
    spec_wave,
    wave_model,
    model_weight,
    mask,
    x_min,
    x_max,
    p0=None,
):
    tol = 5e-10  # tolerance for when the optimizer should stop optimizing.
    spec_err[mask] = 999.0
    num_labels = 3

    def fit_func(dummy_variable, *labels):
        norm_spec = get_spectrum_from_multiheadPayne(
            parameters=np.array([labels[:-1]]),
            wavelength=np.arange(2500, 10000, 2),
            weights=model_weight,
        )
        norm_spec = ndimage.gaussian_filter1d(norm_spec, sigma=1.97)
        norm_spec = utils.doppler_shift(wave_model, norm_spec, labels[-1], spec_wave)
        return norm_spec

    if p0 is None:
        p0 = np.zeros(num_labels)
        p0[-1] = 0

    bounds = np.zeros((2, num_labels))
    bounds[0, :] = -0.5
    bounds[1, :] = 0.5
    bounds[0, -1] = -500.0
    bounds[1, -1] = 500.0

    # run the optimizer
    popt, pcov = curve_fit(
        fit_func,
        xdata=[],
        ydata=spec_flux,
        sigma=spec_err,
        p0=p0,
        bounds=bounds,
        ftol=tol,
        xtol=tol,
        absolute_sigma=True,
        method="trf",
    )
    model_spec = fit_func([], *popt)
    pstd = np.sqrt(np.diagonal(pcov))

    popt[:-1] = (popt[:-1] + 0.5) * (x_max - x_min) + x_min
    pstd[:-1] = pstd[:-1] * (x_max - x_min)

    return popt, pstd, model_spec


from scipy.optimize import minimize


def fit_normalized_spectrum_single_star_model(
    spec_flux,
    spec_err,
    spec_wave,
    wave_model,
    model_weight,
    x_min,
    x_max,
    mask=None,
    p0=None,
):
    tol = 5e-10
    spec_err[mask] = 999.0
    num_labels = 4

    def fit_func(dummy_variable, *labels):
        norm_spec = get_spectrum_from_multiheadPayne(
            parameters=np.array([labels[:-1]]),
            wavelength=np.arange(2500, 10000, 2),
            weights=model_weight,
        )
        norm_spec = ndimage.gaussian_filter1d(norm_spec, sigma=1.97)
        norm_spec = utils.doppler_shift(wave_model, norm_spec, labels[-1], spec_wave)
        return np.sum((norm_spec - spec_flux) ** 2 / spec_err**2)

    def produce_spec(labels):
        norm_spec = get_spectrum_from_multiheadPayne(
            parameters=np.array([labels[:-1]]),
            wavelength=np.arange(2500, 10000, 2),
            weights=model_weight,
        )
        norm_spec = ndimage.gaussian_filter1d(norm_spec, sigma=1.97)
        norm_spec = utils.doppler_shift(wave_model, norm_spec, labels[-1], spec_wave)
        return norm_spec

    if p0 is None:
        p0 = np.zeros(num_labels)
        p0[-1] = 0

    bounds = [(-0.5, 0.5)] * (num_labels - 1) + [(-500, 500)]

    result = minimize(
        fit_func,
        x0=p0,
        bounds=bounds,
        tol=tol,
        method="L-BFGS-B",
        options={"ftol": tol, "gtol": tol},
    )

    popt = result.x
    model_spec = produce_spec(popt)

    pcov = np.linalg.inv(result.hess_inv.todense())
    pstd = np.sqrt(np.diagonal(pcov))

    popt[:-1] = (popt[:-1] + 0.5) * (x_max - x_min) + x_min
    pstd[:-1] = pstd[:-1] * (x_max - x_min)

    return popt, pstd, model_spec
