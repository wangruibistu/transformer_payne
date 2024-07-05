import numpy as np
import matplotlib.pyplot as plt
import torch
from transformer_payne import TransformerPayneModelWave


def save_model_state_dict():
    model_path = "/home/wangr/code/spec_stellar_parameter/csst_parameter/csst_transformer_stellar_params/checkpoints/MHA_Payne_model-epoch=47428-train_loss=19.49-val_loss=19.59.ckpt"
    checkpoint = torch.load(model_path)
    torch.save(
        checkpoint["state_dict"],
        "/home/wangr/code/spec_stellar_parameter/csst_parameter/csst_transformer_stellar_params/checkpoints/model_weights.pt",
    )


def load_weights(weight_path):
    state_dict = torch.load(weight_path)
    weights = {}
    for name, param in state_dict.items():
        weights[name] = param.cpu().data.numpy()
    return weights


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, negative_slope=0.01):
    return np.maximum(negative_slope * x, x)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def wavelength_embedding(d_wave, max_len):
    position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
    div_term = np.exp(
        np.arange(0, d_wave, 2).astype(np.float32) * (-np.log(10000.0) / d_wave)
    )

    pe = np.zeros((max_len, d_wave), dtype=np.float32)
    pe[:, 0::2] = np.sin(position * div_term)

    if d_wave % 2 == 0:
        pe[:, 1::2] = np.cos(position * div_term)
    else:
        pe[:, 1::2] = np.cos(position * div_term[:-1])

    return pe


def apply_wavelength_embedding(x, weights):
    d_wave = weights["wave_embedding.pe"].shape[1]
    max_len = x.shape[0]

    if "wave_embedding.pe" in weights:
        pe = weights["wave_embedding.pe"][:max_len, :]  # .cpu().data.numpy()
    else:
        pe = wavelength_embedding(d_wave, max_len)

    return x + pe


def multi_head_attention(q, k, v, num_heads):
    head_dim = q.shape[-1] // num_heads
    q = q.reshape(-1, num_heads, head_dim)
    k = k.reshape(-1, num_heads, head_dim)
    v = v.reshape(-1, num_heads, head_dim)

    scores = np.einsum("qhd,khd->qkh", q, k) / np.sqrt(head_dim)
    attn_weights = softmax(scores, axis=1)

    out = np.einsum("qkh,khd->qhd", attn_weights, v)
    return out.reshape(q.shape[0], -1)


def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x, weight, bias, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return weight * (x - mean) / np.sqrt(var + eps) + bias


def get_spectrum_from_multiheadPayne(
    parameters,
    wavelength,
    weights,
    num_spectrum_points=3750,
    num_multilayers=8,
    num_heads=5,
):
    parameters = parameters.reshape(1, -1).astype(np.float32)
    wavelength = wavelength.reshape(1, -1).astype(np.float32)

    embedded_wavelength = apply_wavelength_embedding(wavelength, weights)
    embedded_parameters = (
        np.dot(parameters, weights["embedding.weight"].T) + weights["embedding.bias"]
    )

    for layer in range(num_multilayers):
        q = (
            np.dot(
                embedded_wavelength,
                weights[f"encoder_layers.in_proj_weight"][:num_spectrum_points, :].T,
            )
            + weights[f"encoder_layers.in_proj_bias"][:num_spectrum_points]
        )

        k = (
            np.dot(
                embedded_parameters,
                weights[f"encoder_layers.in_proj_weight"][
                    num_spectrum_points : 2 * num_spectrum_points, :
                ].T,
            )
            + weights[f"encoder_layers.in_proj_bias"][
                num_spectrum_points : 2 * num_spectrum_points
            ]
        )

        v = (
            np.dot(
                embedded_parameters,
                weights[f"encoder_layers.in_proj_weight"][
                    2 * num_spectrum_points :, :
                ].T,
            )
            + weights[f"encoder_layers.in_proj_bias"][2 * num_spectrum_points :]
        )

        embedded_parameters_mhd = multi_head_attention(q, k, v, num_heads)
        embedded_parameters_mhd = (
            np.dot(
                embedded_parameters_mhd, weights[f"encoder_layers.out_proj.weight"].T
            )
            + weights[f"encoder_layers.out_proj.bias"]
        )
        new_embedded_parameters = embedded_parameters_mhd + embedded_parameters_mhd

        new_embedded_parameters_norm = layer_norm(
            new_embedded_parameters, weights["norm1.weight"], weights["norm1.bias"]
        )

        ff_output = (
            np.dot(
                new_embedded_parameters_norm, weights["feed_forward.linear_1.weight"].T
            )
            + weights["feed_forward.linear_1.bias"]
        )
        ff_output = relu(ff_output)

        ff_output = (
            np.dot(ff_output, weights["feed_forward.linear_2.weight"].T)
            + weights["feed_forward.linear_2.bias"]
        )

        embedded_parameters_dpt = new_embedded_parameters_norm + ff_output
        embedded_parameters = layer_norm(
            embedded_parameters_dpt, weights["norm2.weight"], weights["norm2.bias"]
        )

    spectrum = (
        np.dot(embedded_parameters, weights["fc1.weight"].T) + weights["fc1.bias"]
    )
    spectrum = leaky_relu(spectrum)
    predicted_spectrum = np.dot(spectrum, weights["fc2.weight"].T) + weights["fc2.bias"]

    return predicted_spectrum.flatten()


if __name__ == "__main__":
    save_model_state_dict()
    model_path = "/home/wangr/code/spec_stellar_parameter/csst_parameter/csst_transformer_stellar_params/checkpoints/model_weights.pt"
    # model_path = "/home/wangr/code/spec_stellar_parameter/csst_parameter/csst_transformer_stellar_params/checkpoints/MHA_Payne_model-epoch=47428-train_loss=19.49-val_loss=19.59.ckpt"
    # weights = torch.load(model_path)['state_dict']
    weights = load_weights(model_path)
    wavelength = np.arange(2500, 10000, 2)
    parameters = np.array([0, 0, 0])
    spectrum = get_spectrum_from_multiheadPayne(
        parameters,
        wavelength,
        weights,
        num_spectrum_points=3750,
        num_multilayers=8,
        num_heads=5,
    )
    # print(spectrum.shape, spectrum)
    model_path = "/home/wangr/code/spec_stellar_parameter/csst_parameter/csst_transformer_stellar_params/checkpoints/MHA_Payne_model-epoch=47428-train_loss=19.49-val_loss=19.59.ckpt"
    model = TransformerPayneModelWave(
        num_attn_heads=5,
        num_multilayer=8,
        num_parameters=3,
        num_spectrum_points=3750,
        num_feature=256,
    )
    checkpoint = torch.load(model_path)
    checkpoint["state_dict"].pop("wave_embedding.pe", None)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.cuda().eval()
    model_spectra = (
        model(
            torch.from_numpy(wavelength).type(torch.float),
            torch.from_numpy(parameters).type(torch.float),
        )
        .cpu()
        .data.numpy()
    )

    plt.figure()
    plt.plot(wavelength, spectrum, label="spec")
    plt.plot(wavelength, model_spectra, label="model spec")
    plt.legend()
    plt.show()
