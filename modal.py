import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from numba import jit
import matplotlib.patches as patches
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    precision_recall_fscore_support,
)


@jit(nopython=True)
def modal_sum_fast(
    w: np.ndarray, a_n: np.ndarray, z_n: np.ndarray, w_n: np.ndarray
) -> np.ndarray:
    """
    Fast modal sum implementation. No noise added.
    """
    tf = np.zeros_like(w, dtype=np.complex64)
    for i in range(len(a_n)):
        tf += a_n[i] * 1j * w / (w_n[i] ** 2 - w**2 + 2j * z_n[i] * w * w_n[i])
    return tf


def modal_sum(w, a_n, z_n, w_n, logsigma=None, multiclass=False):
    """Returns transfer function and labels for a modal sum system.
    Additive Gaussian noise added independently to real and imaginary parts."""
    tf = np.zeros_like(w, dtype=np.complex128)
    for i in range(len(a_n)):
        tf += a_n[i] * 1j * w / (w_n[i] ** 2 - w**2 + 1j * 2 * z_n[i] * w * w_n[i])
    if logsigma is not None:
        noise = np.random.normal(0, np.exp(logsigma), len(w)) + 1j * np.random.normal(
            0, np.exp(logsigma), len(w)
        )
        tf += noise

    y = np.zeros(len(w))
    for i, w_n_value in enumerate(w_n):
        closest_index = np.argmin(np.abs(w - w_n_value))
        dw = w_n_value * z_n[i]  # Half power bandwidth
        indices_in_range = np.where((w > w_n_value - dw) & (w < w_n_value + dw))
        combined_indices = np.concatenate((indices_in_range[0], [closest_index]))

        if multiclass:
            y[combined_indices] = np.where(y[combined_indices] == 0, 1, 2)
        else:
            y[combined_indices] = 1
    return tf, y


def to_db(x):
    """Converts a vector (split or not) to dB"""
    if x.shape[-1] == 2:
        return 20 * np.log10(np.linalg.norm(x, axis=-1))
    return 20 * np.log10(np.abs(x))


@jit(nopython=True)
def split_real_imag(x: np.ndarray) -> np.ndarray:
    """Splits a complex vector into real and imaginary parts.
    Input: (N, 1)
    Output: (N, 2)
    """
    return np.column_stack((np.real(x), np.imag(x)))


def normalise_rms(x):
    """normalise each example by dividing by the RMS value.
    Input: (N, 2)
    Output: (N, 2)
    """
    rms = np.sqrt(np.mean(np.linalg.norm(x, axis=-1) ** 2))
    # rms = np.sqrt(np.mean(np.linalg.norm(x, axis=0)**2))
    return x / rms


def generate_data(
    num_data: int,
    num_w_points: int,
    sigma_max=0.15,
    max_modes=7,
    multiclass=False,
    normalise=None,
    neg_an=True,
    extended=False,
):
    """Generate num_data training examples with num_w_points frequency points.
    Pass normalisation function as a parameter.

    Parameters:
    num_data: number of transfer functions to generate
    num_w_points: number of frequency points in each transfer function
    sigma_max: maximum noise standard deviation
    max_modes: maximum number of modes in each transfer function
    multiclass: if True, include a third class with label '2'
    normalise: normalisation function to apply to each transfer function
    extended: if True, include phase and magnitude as additional features

    Outputs:
    X: (num_data, num_w_points, 2)
    Y: (num_data, num_w_points)
    ws: (num_data, max_modes)
    zs: (num_data, max_modes)

    NO LONGER USED -- replaced w/ generate_dat_extended"""
    X = []
    Y = []
    ws = []
    zs = []
    w = np.linspace(0, 1, num_w_points)
    for i in range(num_data):
        num_modes = np.random.randint(0, max_modes + 1)
        w_n = np.random.uniform(0, 1, num_modes)
        if neg_an:
            a_n = np.random.uniform(-2, 2, num_modes)
        else:
            a_n = np.random.uniform(1, 2, num_modes)
        z_n = np.random.uniform(0.01, 0.20, num_modes)
        sigma = np.random.uniform(0.01, sigma_max)
        out, y = modal_sum(w, a_n, z_n, w_n, sigma, multiclass)

        if normalise is not None:
            out = normalise(out)

        ws.append(w_n)
        zs.append(z_n)
        Y.append(y)

        if extended:
            real_imag = split_real_imag(out)
            phase = np.arctan(np.imag(out) / np.real(out))
            mag = np.abs(out)
            extended_op = np.concatenate(
                (real_imag, phase.reshape(-1, 1), mag.reshape(-1, 1)), axis=1
            )
            X.append(extended_op)
        else:
            X.append(split_real_imag(out))

    return np.array(X), np.array(Y), ws, zs


def generate_dat_extended(
    num_data: int,
    num_w_points: int,
    sigma_max: float = 0.15,
    max_modes: int = 7,
    multiclass: bool = False,
    neg_an: bool = False,
    a_max: float = 2.0,
    method: int = 1,
    norm_95: bool = True,
    logmag: bool = False,
    scaled_logmag: bool = False,
    normalise: bool = None,
    w_n: list = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate num_data training examples with num_w_points frequency points.
    Pass normalisation function as a parameter. Include phase and magnintude
    information in the output.

    Args:
        num_data: Number of TF's to generate
        num_w_points: Number of frequency points in each TF
        sigma_max: Max noise level
        max_modes: Max number of modes per TF
        multiclass: If True, include a third class with label '2'
        neg_an: If true, allow negative modal amplitudes
        a_max: Max modal amplitude
        method: 1 or 2. Method for sampling a_n and z_n (Training set A and B in report)
        norm_95: Normalise output features by 95% of maximum TF maginitude
        w_n: Enforces specific mode frequencies for all TF's

        --- Old Arguments (no longer used, retained for testing purposes) ---
        logmag: If true, output log maginitde as an additional feature
        scaled_logmag: If true, scale log magnitude to have zero mean and unit variance
        normalise: Normalisation function to apply to each TF

    Returns:
        X: Output features for each TF, shape: (num_data, num_w_points, 4)
        Y: Labels for each TF, shape: (num_data, num_w_points)
        ws: Natural frequences, shape: (num_data, max_modes)
        zs: Damping ratios, shape: (num_data, max_modes)
        a_s: Modal amplitdes, shape: (num_data, max_modes)
    """
    X = []
    Y = []
    ws = []
    zs = []
    a_s = []
    w = np.linspace(0, 1, num_w_points)
    for i in range(num_data):
        num_modes = np.random.randint(0, max_modes + 1)
        if w_n is not None:
            num_modes = len(w_n)
        else:
            w_n = np.random.uniform(0, 1, num_modes)

        if neg_an:
            if method == 1:
                a_n = np.random.uniform(-a_max, a_max, num_modes)
            elif method == 2:
                a_max = 2.5
                mag = np.random.uniform(-a_max, a_max, num_modes)
                phase = np.random.normal(0, np.pi / 8, num_modes)
                a_n = mag * np.exp(
                    1j * phase
                )  # combine mag and phase to make a complex num
        else:
            a_n = np.random.uniform(1, 2, num_modes)
        if method == 1:
            z_n = np.random.uniform(0.01, 0.20, num_modes)
        elif method == 2:
            z_n = 10 ** (
                np.random.uniform(-3, -0.7, num_modes)
            )  # log-uniform sampling of z_n

        # sigma = np.random.uniform(0.0001, sigma_max)
        # sigma = np.random.uniform(np.log(8e-2), sigma_max)
        # sigma = np.random.uniform(np.log(1e-1), sigma_max)
        sigma = np.random.uniform(np.log(3e-1), sigma_max)

        out, y = modal_sum(w, a_n, z_n, w_n, sigma, multiclass)

        if normalise is not None:
            out = normalise(out)

        real_imag = split_real_imag(out)
        phase = np.arctan(np.imag(out) / np.real(out))
        mag = np.abs(out)
        # phase = np.mod(phase, 2*np.pi)

        if norm_95 is True:
            max_mag = np.max(mag)
            mag = mag / (0.95 * max_mag)
            real_imag = real_imag / (0.95 * max_mag)

        if logmag is True:
            logmagnitude = np.log10(mag)
            if scaled_logmag is True:
                logmagnitude = (logmagnitude - np.mean(logmagnitude)) / np.std(
                    logmagnitude
                )
            extended_op = np.concatenate(
                (
                    real_imag,
                    phase.reshape(-1, 1),
                    mag.reshape(-1, 1),
                    logmagnitude.reshape(-1, 1),
                ),
                axis=1,
            )
        else:
            extended_op = np.concatenate(
                (real_imag, phase.reshape(-1, 1), mag.reshape(-1, 1)), axis=1
            )
        ws.append(w_n)
        zs.append(z_n)
        a_s.append(a_n)
        X.append(extended_op)
        Y.append(y)
    return np.array(X), np.array(Y), ws, zs, a_s


def plot_tf(
    tf, y, todb=True, ws=None, figsize=(8, 6), w=None, model=None, X=None, name=None
):
    """Plot transfer function and show training labels and mode frequencies."""
    fig, ax = plt.subplots(figsize=figsize)
    if w is None:
        w = np.linspace(0, 1, len(y))
    else:
        w = w

    mask = np.zeros_like(w)
    mask[y == 1] = 1
    mask[y == 2] = 1
    if todb:
        tf_db = to_db(tf)
        # ax.scatter(w[y == 1], modal.to_db(tf)[y == 1], c='red', marker='o', label=r'Training Labels $(t_m = 1)$')
        ax.plot(w, tf_db, label="Transfer Function", c="blue", alpha=0.7)

    ax.imshow(
        mask.reshape(1, -1),
        aspect="auto",
        extent=[0, 1, ax.get_ylim()[0], ax.get_ylim()[1]],
        cmap="Greys",
        alpha=0.1,
    )

    if ws is not None:
        for w_n in ws:
            ax.axvline(w_n, c="black", linestyle="--", alpha=0.5)

    if w is not None:
        ax.set_xlabel("Frequency (rad/s)")
    else:
        ax.set_xlabel("Normalised Frequency")
    if todb:
        ax.set_ylabel("Magnitude (dB)")
    else:
        ax.set_ylabel("Magnitude")

    if model is not None and X is not None:
        tf_tensor = torch.from_numpy(X).to(torch.float32)
        model.eval()
        output = model(tf_tensor)
        test_op = np.array(output.detach().numpy())
        predicted = np.argmax(test_op, axis=-1).reshape(-1)

        mask = np.zeros_like(w)
        mask[predicted.reshape(-1) == 1] = 1
        mask[predicted.reshape(-1) == 2] = 1

        segment_start = None
        for i in range(len(w)):
            if predicted[i] == 1:
                if segment_start is None:
                    segment_start = i
            elif segment_start is not None:
                if i == segment_start + 1:
                    ax.scatter(
                        w[segment_start], tf_db[segment_start], c="red", alpha=0.7, s=50
                    )
                ax.plot(
                    w[segment_start:i],
                    tf_db[segment_start:i],
                    c="red",
                    alpha=0.7,
                    linewidth=5,
                    label="Predicted=1",
                )
                segment_start = None
        if segment_start is not None:
            ax.plot(
                w[segment_start:i],
                tf_db[segment_start:i],
                c="red",
                alpha=0.7,
                linewidth=5,
            )
    # legend_elements = []
    # legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Training Labels $(t_m = 1)$')]
    ax.set_xlim(
        -50,
    )
    if name is not None:
        plt.savefig(f"./Figs/{name}.pdf")
    plt.show()
    return fig, ax


def plot_predictions(
    val_inputs,
    val_outputs,
    val_targets,
    multiclass=False,
    nrows=4,
    ncols=4,
    ws=None,
    name=None,
    extended=False,
    figsize=(6, 5.5),
    show_legend=False,
):
    """Plot 4x4 grid of predictions on a validation set of data.
    val_inputs: (num_data, num_w_points, 2)
    val_outputs: (num_data, num_w_points)
    val_targets: (num_data, num_w_points)"""
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize, constrained_layout=True
    )
    w = np.linspace(0, 1, len(val_targets[0]))

    # Create a custom colormap with a single color
    single_color1 = mcolors.LinearSegmentedColormap.from_list(
        "", ["white", "#fb9a99"]
    )  # red
    single_color2 = mcolors.LinearSegmentedColormap.from_list(
        "", ["white", "#a6cee3"]
    )  # blue

    for index, ax in enumerate(axs.flat):

        targets = np.array(val_targets[index].numpy())
        test_op = np.array(val_outputs[index].numpy())

        if multiclass:
            tf_db = to_db(val_inputs[index].numpy()[:, :2])

            predicted = np.argmax(test_op, axis=-1)
            ax.plot(w, tf_db, c="black", alpha=0.7, linewidth=1.15)

            segment_start = None
            segment_color = None

            for i in range(len(w)):
                if predicted[i] == 1 and (
                    segment_start is None or segment_color != "#e31a1c"
                ):  # red
                    if segment_start is not None:
                        ax.plot(
                            w[segment_start:i],
                            tf_db[segment_start:i],
                            c=segment_color,
                            alpha=0.7,
                            linewidth=5,
                            label=f"Predicted={segment_color[-1]}",
                        )
                    segment_start = i
                    segment_color = "#e31a1c"
                elif predicted[i] == 2 and (
                    segment_start is None or segment_color != "royalblue"
                ):  # blue #1f78b4
                    if segment_start is not None:
                        ax.plot(
                            w[segment_start:i],
                            tf_db[segment_start:i],
                            c=segment_color,
                            alpha=0.7,
                            linewidth=5,
                            label=f"Predicted={segment_color[-1]}",
                        )
                    segment_start = i
                    segment_color = "royalblue"
                elif (
                    predicted[i] != 1
                    and predicted[i] != 2
                    and segment_start is not None
                ):
                    if i == segment_start + 1:
                        ax.scatter(
                            w[segment_start],
                            tf_db[segment_start],
                            c=segment_color,
                            alpha=0.7,
                            s=50,
                        )
                    ax.plot(
                        w[segment_start:i],
                        tf_db[segment_start:i],
                        c=segment_color,
                        alpha=0.7,
                        linewidth=5,
                        label=f"Predicted={segment_color[-1]}",
                    )
                    segment_start = None
                    segment_color = None
            if segment_start is not None:
                ax.plot(
                    w[segment_start:i],
                    tf_db[segment_start:i],
                    c=segment_color,
                    alpha=0.7,
                    linewidth=5,
                    label=f"Predicted={segment_color[-1]}",
                )

            if ws is not None:
                for w_n in ws[index]:
                    ax.axvline(w_n, c="black", linestyle="--", alpha=0.4)

            mask = np.zeros_like(w)
            mask[targets == 1] = 1
            ax.imshow(
                mask.reshape(1, -1),
                aspect="auto",
                extent=[0, 1, ax.get_ylim()[0], ax.get_ylim()[1]],
                cmap=single_color1,
                alpha=0.5,
            )  #

            mask2 = np.zeros_like(w)
            mask2[targets == 2] = 1
            ax.imshow(
                mask2.reshape(1, -1),
                aspect="auto",
                extent=[0, 1, ax.get_ylim()[0], ax.get_ylim()[1]],
                cmap=single_color2,
                alpha=0.5,
            )

            if show_legend:
                legend_elements = []
                legend_elements.append(
                    plt.Line2D([0], [0], color="black", label="Transfer Function")
                )
                legend_elements.append(
                    plt.Line2D(
                        [0], [0], color="black", linestyle="--", label="Mode Frequency"
                    )
                )
                rect = patches.Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor="#fb9a99",
                    edgecolor="none",
                    alpha=0.5,
                    label="Training Labels (Class 1)",
                )
                legend_elements.append(rect)
                rect = patches.Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor="#a6cee3",
                    edgecolor="none",
                    alpha=0.5,
                    label="Training Labels (Class 2)",
                )
                legend_elements.append(rect)
                legend_elements.append(
                    plt.Line2D(
                        [0],
                        [0],
                        color="#e31a1c",
                        alpha=0.7,
                        linewidth=5,
                        label="Model Predictions (Class 1)",
                    )
                )
                legend_elements.append(
                    plt.Line2D(
                        [0],
                        [0],
                        color="royalblue",
                        alpha=0.7,
                        linewidth=5,
                        label="Model Predictions (Class 2)",
                    )
                )

                fig.legend(
                    handles=legend_elements,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.1),
                    # bbox_to_anchor=(0.5, -0),
                    bbox_transform=fig.transFigure,
                    ncol=3,
                    fancybox=True,
                    #   shadow=True,
                    # frameon=True,
                )

        else:
            predicted = (test_op > 0.0).astype(int)
            if extended is True:
                tf_db = to_db(val_inputs[index].numpy()[:, :2])
            else:
                tf_db = to_db(val_inputs[index].numpy())

            ax.plot(w, tf_db, c="blue", alpha=0.7, linewidth=1.15)

            mask = np.zeros_like(w)
            mask[predicted.reshape(-1) == 1] = 1

            segment_start = None
            for i in range(len(w)):
                if predicted[i] == 1:
                    if segment_start is None:
                        segment_start = i
                elif segment_start is not None:
                    ax.plot(
                        w[segment_start:i],
                        tf_db[segment_start:i],
                        c="red",
                        alpha=0.7,
                        linewidth=5,
                        label="Predicted=1",
                    )
                    segment_start = None

            if ws is not None:
                for w_n in ws[index]:
                    ax.axvline(w_n, c="black", linestyle="--", alpha=0.4)

            mask = np.zeros_like(w)
            mask[targets == 1] = 1
            ax.imshow(
                mask.reshape(1, -1),
                aspect="auto",
                extent=[0, 1, ax.get_ylim()[0], ax.get_ylim()[1]],
                cmap="Greys",
                alpha=0.1,
            )

    fig.text(0.5, -0.01, "Normalised Frequency", ha="center", va="center", fontsize=14)
    fig.text(
        -0.02,
        0.5,
        "Magnitude (dB)",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=14,
    )

    if name is not None:
        plt.savefig(f"./Figs/{name}.pdf")


def plot_predictions_extended(
    model,
    num_w_points=200,
    normalise=None,
    neg_an=True,
    multiclass=False,
    nrows=4,
    ncols=4,
    s=20,
    figsize=(12, 12),
):
    """Plot 4x4 grid of predictions on a validation set of data.
    val_inputs: (num_data, num_w_points, 2)
    val_outputs: (num_data, num_w_points)
    val_targets: (num_data, num_w_points)"""

    valX, valy, _, __ = generate_data(
        32 * 1,
        num_w_points,
        multiclass=multiclass,
        normalise=normalise,
        neg_an=neg_an,
        extended=True,
    )
    val_X = torch.from_numpy(valX).to(torch.float32)
    if multiclass:
        val_y = torch.from_numpy(valy).to(torch.long)
    else:
        val_y = torch.from_numpy(valy).to(torch.float32)

    if normalise is not None:
        val_X = normalise(val_X)
    dataset = TensorDataset(val_X, val_y)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model.eval()
    for val_inputs, val_targets in val_loader:
        with torch.no_grad():
            val_outputs = model(val_inputs)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    w = np.linspace(0, 1, len(val_targets[0]))
    if nrows == 1 and ncols == 1:
        ax = axs
        index = 1
        targets = np.array(val_targets[index].numpy())
        test_op = np.array(val_outputs[index].numpy())

        tf = val_inputs[index].numpy()[:, :2]

        if multiclass:
            predicted = np.argmax(test_op, axis=-1)
            ax.plot(w, to_db(tf), c="black")

            ax.scatter(
                w[predicted == 1],
                to_db(tf)[predicted == 1],
                c="blue",
                marker="o",
                label="Model Predictions (Class 1)",
                s=s,
            )
            ax.scatter(
                w[predicted == 2],
                to_db(tf)[predicted == 2],
                c="red",
                marker="o",
                s=s,
                label="Model Predictions (Class 2)",
            )
            for i in range(len(w)):
                if targets[i] == 1:
                    color = "green"
                    marker = ","
                    label = r"Training Labels ($y_n = 1$)" if i == 0 else None
                    ax.scatter(w[i], 0, c=color, marker=marker, s=s)
                if targets[i] == 2:
                    color = "darkviolet"
                    marker = ","
                    label = r"Training Labels ($y_n = 2$)" if i == 0 else None
                    ax.scatter(w[i], 0, c=color, marker=marker, s=s)
            # ax.scatter(w[targets == 1], 0, c='green', marker='o', label=r'Training Labels ($y_m = 1$)', s=s)
            # ax.scatter(w[targets == 2], 0, c='m', marker='o',s=s, label=r'Training Labels ($y_m = 2$)')

            ax.set_xlabel("Normalised Frequency")
            ax.set_ylabel("Magnitude (dB)")
            ax.legend()
        else:
            predicted = (test_op > 0.0).astype(int)
            ax.plot(w, to_db(tf), c="black")

            for i in range(len(w)):
                if i == 0:
                    if predicted[i] == 1:
                        ax.scatter(
                            w[i],
                            to_db(tf)[i],
                            c="blue",
                            marker="o",
                            label="Model Predictions",
                        )
                    if targets[i] == 1:
                        ax.scatter(
                            w[i],
                            0,
                            c="green",
                            marker="o",
                            label=r"Training Labels ($y_n = 1$)",
                        )
                else:
                    if predicted[i] == 1:
                        ax.scatter(w[i], to_db(tf)[i], c="blue", marker="o")
                        # ax.legend()
                    if targets[i] == 1:
                        ax.scatter(w[i], 0, c="green", marker="o")

            ax.set_xlabel("Normalised Frequency")
            ax.set_ylabel("Magnitude (dB)")
            ax.legend()

    else:
        for index, ax in enumerate(axs.flat):

            targets = np.array(val_targets[index].numpy())
            test_op = np.array(val_outputs[index].numpy())

            tf = val_inputs[index].numpy()[:, :2]

            if multiclass:
                predicted = np.argmax(test_op, axis=-1)
                ax.plot(w, to_db(tf), c="black")

                ax.scatter(
                    w[predicted == 1],
                    to_db(tf)[predicted == 1],
                    c="blue",
                    marker="o",
                    s=s,
                )
                ax.scatter(
                    w[predicted == 2],
                    to_db(tf)[predicted == 2],
                    c="red",
                    marker="o",
                    s=s,
                )
                for i in range(len(w)):
                    if targets[i] == 1:
                        color = "green"
                        marker = ","
                        label = r"Training Labels ($y_n = 1$)" if i == 0 else None
                        ax.scatter(w[i], 0, c=color, marker=marker, s=s)
                    if targets[i] == 2:
                        color = "darkviolet"
                        marker = ","
                        label = r"Training Labels ($y_n = 2$)" if i == 0 else None
                        ax.scatter(w[i], 0, c=color, marker=marker, s=s)

                ax.set_xlabel("Normalised Frequency")
                ax.set_ylabel("Magnitude (dB)")
                # ax.legend()
            else:
                predicted = (test_op > 0.0).astype(int)
                ax.plot(w, to_db(tf), c="black")

                for i in range(len(w)):
                    if i == 0:
                        if predicted[i] == 1:
                            ax.scatter(
                                w[i],
                                to_db(tf)[i],
                                c="blue",
                                marker="o",
                                label="Model Predictions",
                            )
                        if targets[i] == 1:
                            ax.scatter(
                                w[i],
                                0,
                                c="green",
                                marker="o",
                                label=r"Training Labels ($y_n = 1$)",
                            )
                    else:
                        if predicted[i] == 1:
                            ax.scatter(w[i], to_db(tf)[i], c="blue", marker="o")
                            # ax.legend()
                        if targets[i] == 1:
                            ax.scatter(w[i], 0, c="green", marker="o")

                ax.set_xlabel("Normalised Frequency")
                ax.set_ylabel("Magnitude (dB)")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=8),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="darkviolet", markersize=8
        ),
    ]

    legend_labels = [
        "Model Predictions (Class 1)",
        "Model Predictions (Class 2)",
        "Training Labels ($y_n = 1$)",
        "Training Labels ($y_n = 2$)",
    ]

    ax.legend(legend_handles, legend_labels)
    # plt.legend()
    plt.tight_layout()


def plot_results(model_results: dict) -> None:
    """Plot the training and validation loss, precision and recall for a model."""
    # plot the training and validation loss
    fig, ax = plt.subplots(nrows=1, ncols=2, tight_layout=True)
    ax[0].plot(
        model_results["epochs"],
        model_results["training_loss"],
        label="Training loss",
        c="b",
    )
    ax[0].plot(
        model_results["epochs"],
        model_results["validation_loss"],
        label="Validation loss",
        c="r",
    )
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    # plot the training and validation precision and recall
    ax[1].plot(model_results["training_precision"], label="Training precision", c="b")
    ax[1].plot(
        model_results["validation_precision"],
        label="Validation precision",
        c="b",
        linestyle="--",
    )
    ax[1].plot(model_results["training_recall"], label="Training recall", c="r")
    ax[1].plot(
        model_results["validation_recall"],
        label="Validation recall",
        c="r",
        linestyle="--",
    )
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Precision/Recall")
    ax[1].legend()

    plt.show()


def plot_predictions_with_probs(
    model, normalise=None, multiclass=False, sigma_max=0.15, num_w_points=200
):
    """Plot model predictions on validation transfer function,
    alongside raw predicted probabilites"""

    valX, valy, _, __ = generate_data(
        32 * 1,
        num_w_points,
        multiclass=multiclass,
        normalise=normalise,
        sigma_max=sigma_max,
    )
    val_X = torch.from_numpy(valX).to(torch.float32)
    if multiclass:
        val_y = torch.from_numpy(valy).to(torch.long)
    else:
        val_y = torch.from_numpy(valy).to(torch.float32)

    if normalise is not None:
        val_X = normalise(val_X)
    dataset = TensorDataset(val_X, val_y)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model.eval()
    for val_inputs, val_targets in val_loader:
        with torch.no_grad():
            val_outputs = model(val_inputs)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
    w = np.linspace(0, 1, len(val_targets[0]))
    index = np.random.randint(0, len(val_targets))
    targets = np.array(val_targets[index].numpy())
    test_op = np.array(val_outputs[index].numpy())

    predicted = (test_op > 0.0).astype(int)
    ax[0].plot(w, to_db(val_inputs[index].numpy()), c="black")

    for i in range(len(w)):
        if i == 0:
            if predicted[i] == 1:
                ax[0].scatter(
                    w[i],
                    to_db(val_inputs[index].numpy())[i],
                    c="blue",
                    marker="o",
                    label="Model Predictions",
                )
            if targets[i] == 1:
                ax[0].scatter(
                    w[i], 0, c="green", marker="o", label="Ground Truth Labels"
                )
        else:
            if predicted[i] == 1:
                ax[0].scatter(
                    w[i], to_db(val_inputs[index].numpy())[i], c="blue", marker="o"
                )
                # ax.legend()
            if targets[i] == 1:
                ax[0].scatter(w[i], 0, c="green", marker="o")

        ax[0].set_xlabel("Normalised Frequency")
        ax[0].set_ylabel("Magnitude (dB)")

    model_probs = 1 / (1 + np.exp(-test_op))
    ax[1].plot(w, targets, c="green", label="Training Labels (y_n = 1)")
    ax[1].plot(w, model_probs, c="blue", linestyle="--", label="Raw Model Predictions")
    # ax[1].plot(w, predicted, c='green',linestyle='--', label='Model Predictions')
    ax[1].set_xlabel("Normalised Frequency")
    ax[1].set_ylabel(r"$p(y_n = 1)$")

    plt.legend()
    plt.tight_layout()


def compare_models(
    model1, model2, neg_an=False, extended=False, figsize=(6, 5.5), nrows=2, ncols=2
):
    """
    Compare predictions of two single class models on a validation set of data.
    """
    if extended:
        valX, valy, ws, __ = generate_dat_extended(
            32 * 1,
            200,
            sigma_max=np.log(0.2),
            max_modes=7,
            neg_an=neg_an,
            logmag=False,
            scaled_logmag=False,
            method=1,
            norm_95=False,
        )
    else:
        valX, valy, ws, __ = generate_data(
            32 * 1, 200, neg_an=False, sigma_max=np.log(0.01), max_modes=5
        )
    val_X = torch.from_numpy(valX).to(torch.float32)
    val_y = torch.from_numpy(valy).to(torch.float32)

    dataset = TensorDataset(val_X, val_y)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model1.eval()
    model2.eval()
    for val_inputs, val_targets in val_loader:
        with torch.no_grad():
            val_outputs1 = model1(val_inputs)
            val_outputs2 = model2(val_inputs)

    plot_predictions(
        val_inputs,
        val_outputs1,
        val_targets,
        ws=ws,
        nrows=nrows,
        ncols=ncols,
        extended=extended,
        figsize=figsize,
    )
    plot_predictions(
        val_inputs,
        val_outputs2,
        val_targets,
        ws=ws,
        nrows=nrows,
        ncols=ncols,
        extended=extended,
        figsize=figsize,
    )


def compare_models_multiclass(
    model1,
    model2,
    max_norm=False,
    method=1,
    nrows=2,
    ncols=2,
    figsize=(6, 5.5),
    name1=None,
    name2=None,
    show_legend=True,
):
    valX, valy, ws, __ = generate_dat_extended(
        8 * 1,
        500,
        neg_an=True,
        sigma_max=np.log(0.2),
        max_modes=6,
        multiclass=True,
        norm_95=max_norm,
        method=method,
        a_max=2.0,
    )
    """
    Compare predictions of two multiclass models on a validation set of data.
    """
    val_X = torch.from_numpy(valX).to(torch.float32)
    val_y = torch.from_numpy(valy).to(torch.long)

    dataset = TensorDataset(val_X, val_y)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model1.eval()
    model2.eval()
    for val_inputs, val_targets in val_loader:
        with torch.no_grad():
            val_outputs1 = model1(val_inputs)
            val_outputs2 = model2(val_inputs)

    plot_predictions(
        val_inputs,
        val_outputs1,
        val_targets,
        ws=ws,
        nrows=nrows,
        ncols=ncols,
        multiclass=True,
        figsize=figsize,
        name=name1,
        show_legend=show_legend,
    )
    plot_predictions(
        val_inputs,
        val_outputs2,
        val_targets,
        ws=ws,
        nrows=nrows,
        ncols=ncols,
        multiclass=True,
        figsize=figsize,
        name=name2,
        show_legend=show_legend,
    )


def calculate_precision_and_recall_binary(outputs, targets):
    # Round the output to 0 or 1
    predicted = (outputs > 0.5).float()
    true_positives = (predicted * targets).sum().item()
    false_positives = (predicted * (1 - targets)).sum().item()
    false_negatives = ((1 - predicted) * targets).sum().item()
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    return precision, recall


def train_model_binary(model, X, y, valX, valy, name, num_epochs=150, weight=4.0):

    X = torch.from_numpy(X).to(torch.float32)
    y = torch.from_numpy(y).to(torch.float32)
    val_X = torch.from_numpy(valX).to(torch.float32)
    val_y = torch.from_numpy(valy).to(torch.float32)

    dataset = TensorDataset(X, y)
    train_loader = DataLoader(
        dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4
    )

    val_dataset = TensorDataset(val_X, val_y)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Define a binary cross-entropy loss function and an optimizer
    # pos_weight > 1 aims to increase recall
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([weight])
    )  # Binary cross-entropy loss with logits
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    result_dict = {
        "training_loss": [],
        "validation_loss": [],
        "training_precision": [],
        "training_recall": [],
        "validation_precision": [],
        "validation_recall": [],
        "epochs": [],
    }

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_samples = 0

        val_loss = 0.0
        total_val_precision = 0.0
        total_val_recall = 0.0
        total_val_samples = 0

        model.train()

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)

            # Reshape the outputs and targets for the loss calculation
            outputs = outputs.view(
                -1, 1
            )  # Reshape to [batch_size * sequence_length, 1]
            targets = targets.view(-1, 1)

            loss = criterion(outputs, targets.float())  # Convert targets to float
            loss.backward()
            optimizer.step()

            batch_precision, batch_recall = calculate_precision_and_recall_binary(
                outputs, targets
            )
            total_loss += loss.item() * len(inputs)
            total_precision += batch_precision * len(inputs)
            total_recall += batch_recall * len(inputs)
            total_samples += len(inputs)

        model.eval()  # !!!
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                outputs = outputs.view(-1, 1)
                targets = targets.view(-1, 1)

                loss = criterion(outputs, targets.float())
                val_loss += loss.item() * len(inputs)
                batch_precision, batch_recall = calculate_precision_and_recall_binary(
                    outputs, targets
                )
                total_val_precision += batch_precision * len(inputs)
                total_val_recall += batch_recall * len(inputs)
                total_val_samples += len(inputs)

        average_loss = total_loss / total_samples
        average_precision = total_precision / total_samples
        average_recall = total_recall / total_samples
        # average_accuracy = total_accuracy / total_samples
        average_val_loss = val_loss / total_val_samples
        average_val_precision = total_val_precision / total_val_samples
        average_val_recall = total_val_recall / total_val_samples

        # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}, Accuracy: {average_accuracy}')
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}, Precision: {average_precision}, Recall: {average_recall}"
        )
        print(
            f"Validation Precision: {average_val_precision}, Validation Recall: {average_val_recall}"
        )

        result_dict["training_loss"].append(average_loss)
        result_dict["validation_loss"].append(average_val_loss)
        result_dict["training_precision"].append(average_precision)
        result_dict["training_recall"].append(average_recall)
        result_dict["validation_precision"].append(average_val_precision)
        result_dict["validation_recall"].append(average_val_recall)
        result_dict["epochs"].append(epoch + 1)

    torch.save(model, f"{name}.pth")
    return result_dict


def validate_model(
    model,
    num_w_points=200,
    sigma_max=0.15,
    neg_an=False,
    max_modes=5,
    extended=False,
    normalise=None,
    norm_95=True,
    method=1,
):
    """Validate a single class model on a validation set of data.
    Prints loss, precision, recall and F2 score on the validation set."""
    if extended is True:
        valX, valy, _, __ = generate_dat_extended(
            32 * 250,
            num_w_points,
            sigma_max=sigma_max,
            # max_modes=max_modes,
            neg_an=True,
            max_modes=max_modes,
            multiclass=False,
            normalise=normalise,
            norm_95=norm_95,
            method=method,
        )
    else:
        valX, valy, _, __ = generate_data(
            32 * 150,
            num_w_points,
            sigma_max=sigma_max,
            max_modes=max_modes,
            neg_an=neg_an,
        )
    val_X = torch.from_numpy(valX).to(torch.float32)
    val_y = torch.from_numpy(valy).to(torch.float32)

    total_val_loss = 0.0
    total_val_precision = 0.0
    total_val_recall = 0.0
    total_val_samples = 0

    dataset = TensorDataset(val_X, val_y)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([4.0])
    )  # Binary cross-entropy loss with logits

    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            outputs = outputs.view(-1, 1)
            targets = targets.view(-1, 1)

            loss = criterion(outputs, targets.float())
            total_val_loss += loss.item() * len(inputs)
            batch_precision, batch_recall = calculate_precision_and_recall_binary(
                outputs, targets
            )
            total_val_precision += batch_precision * len(inputs)
            total_val_recall += batch_recall * len(inputs)
            total_val_samples += len(inputs)

    average_val_loss = total_val_loss / total_val_samples
    average_val_precision = total_val_precision / total_val_samples
    average_val_recall = total_val_recall / total_val_samples

    print(f"Loss: {average_val_loss}")
    print(f"Precision: {average_val_precision}")
    print(f"Recall: {average_val_recall}")
    print(
        f"F_2: {5 * average_val_precision* average_val_recall / (average_val_recall + 4*average_val_precision)}"
    )


def calculate_precision_and_recall(outputs, targets):
    """Calculate precision and recall for a multiclass problem."""
    _, predicted = torch.max(outputs, dim=-1)

    # Initialize variables to store true positives, false positives, and false negatives for each class
    true_positives = torch.zeros(outputs.shape[-1])
    false_positives = torch.zeros(outputs.shape[-1])
    false_negatives = torch.zeros(outputs.shape[-1])

    # Loop through each class
    for i in range(outputs.shape[-1]):
        true_positives[i] = ((predicted == i) & (targets == i)).sum().item()
        false_positives[i] = ((predicted == i) & (targets != i)).sum().item()
        false_negatives[i] = ((predicted != i) & (targets == i)).sum().item()

    # Calculate precision and recall for each class
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    return precision, recall


def train_model_multiclass(model, X, y, valX, valy, name, num_epochs=150):
    X = torch.from_numpy(X).to(torch.float32)
    y = torch.from_numpy(y).to(torch.long)
    val_X = torch.from_numpy(valX).to(torch.float32)
    val_y = torch.from_numpy(valy).to(torch.long)

    dataset = TensorDataset(X, y)
    train_loader = DataLoader(
        dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4
    )

    val_dataset = TensorDataset(val_X, val_y)
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4
    )

    # Define a binary cross-entropy loss function and an optimizer
    # pos_weight > 1 aims to increase recall
    # criterion = nn.CrossEntropyLoss(weight = torch.tensor([1.0, 4.0, 20.0]))
    # criterion = nn.CrossEntropyLoss(weight = torch.tensor([1.0, 4.0, 15.0]))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0, 8.0]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    result_dict = {
        "training_loss": [],
        "validation_loss": [],
        "training_precision": [],
        "training_recall": [],
        "validation_precision": [],
        "validation_recall": [],
        "epochs": [],
    }

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_samples = 0

        total_val_loss = 0.0
        total_val_precision = 0.0
        total_val_recall = 0.0
        total_val_samples = 0

        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)

            targets = targets.view(-1)
            loss = criterion(outputs.view(-1, 3), targets)

            loss.backward()
            optimizer.step()

            batch_precision, batch_recall = calculate_precision_and_recall(
                outputs, targets
            )
            total_loss += loss.item() * len(inputs)
            total_precision += batch_precision * len(inputs)
            total_recall += batch_recall * len(inputs)
            total_samples += len(inputs)

        model.eval()  # !!!
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                targets = targets.view(-1)
                loss = criterion(outputs.view(-1, 3), targets)
                total_val_loss += loss.item() * len(inputs)
                batch_precision, batch_recall = calculate_precision_and_recall(
                    outputs, targets
                )
                total_val_precision += batch_precision * len(inputs)
                total_val_recall += batch_recall * len(inputs)
                total_val_samples += len(inputs)

        average_loss = total_loss / total_samples
        average_precision = total_precision / total_samples
        average_recall = total_recall / total_samples
        # average_accuracy = total_accuracy / total_samples
        average_val_loss = total_val_loss / total_val_samples
        average_val_precision = total_val_precision / total_val_samples
        average_val_recall = total_val_recall / total_val_samples

        # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}, Accuracy: {average_accuracy}')
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}, Precision: {average_precision}, Recall: {average_recall}"
        )
        print(
            f"Validation Precision: {average_val_precision}, Validation Recall: {average_val_recall}"
        )

        result_dict["training_loss"].append(average_loss)
        # result_dict['validation_loss'].append(average_val_loss)
        result_dict["training_precision"].append(average_precision)
        result_dict["training_recall"].append(average_recall)
        result_dict["validation_precision"].append(average_val_precision)
        result_dict["validation_recall"].append(average_val_recall)
        result_dict["epochs"].append(epoch + 1)

    torch.save(model, f"{name}.pth")
    return result_dict


def calculate_precision_and_recall_multiclass(outputs, targets):
    predicted = np.argmax(outputs, axis=-1)
    predicted = predicted.cpu().numpy()
    targets = targets.cpu().numpy()

    precision = precision_score(
        targets, predicted.reshape(-1), average="macro", zero_division=0
    )
    recall = recall_score(
        targets, predicted.reshape(-1), average="macro", zero_division=0
    )
    f2 = fbeta_score(
        targets, predicted.reshape(-1), average="macro", zero_division=0, beta=2
    )

    return precision, recall, f2


def calc(output, targets):
    predicted = np.argmax(output, axis=-1)
    predicted = predicted.cpu().numpy()
    targets = targets.cpu().numpy()
    return precision_recall_fscore_support(
        targets, predicted.reshape(-1), average=None, zero_division=0, beta=2
    )


def validate_model_multiclass(
    model,
    num_w_points=200,
    sigma_max=0.15,
    logmag=False,
    scaled_logmag=False,
    method=1,
):
    """Validate a multiclass model on a validation set of data."""
    valX, valy, _, __ = generate_dat_extended(
        32 * 150,
        num_w_points,
        sigma_max=sigma_max,
        max_modes=6,
        neg_an=True,
        multiclass=True,
        logmag=logmag,
        scaled_logmag=scaled_logmag,
        method=method,
    )
    val_X = torch.from_numpy(valX).to(torch.float32)
    val_y = torch.from_numpy(valy).to(torch.long)

    total_val_precision = 0.0
    total_val_recall = 0.0
    total_val_f2 = 0.0
    total_val_samples = 0
    tot_p = np.zeros(3)
    tot_r = np.zeros(3)
    tot_f = np.zeros(3)

    dataset = TensorDataset(val_X, val_y)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model.eval()  # !!!
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            targets = targets.view(-1)
            # predicted = np.argmax(outputs, axis=-1)

            batch_precision, batch_recall, batch_f2 = (
                calculate_precision_and_recall_multiclass(outputs, targets)
            )
            total_val_precision += batch_precision * len(inputs)
            total_val_recall += batch_recall * len(inputs)
            total_val_f2 += batch_f2 * len(inputs)
            total_val_samples += len(inputs)

            p, r, f, _ = calc(outputs, targets)
            tot_p += p * len(inputs)
            tot_r += r * len(inputs)
            tot_f += f * len(inputs)

    average_val_precision = total_val_precision / total_val_samples
    average_val_recall = total_val_recall / total_val_samples
    average_val_f2 = total_val_f2 / total_val_samples

    print(f"Precision: {average_val_precision}")
    print(f"Recall: {average_val_recall}")
    # print(f"F_2: {5 * average_val_precision* average_val_recall / (average_val_recall + 4*average_val_precision)}")
    print(f"F_2: {average_val_f2}")
    avg_p = tot_p / total_val_samples
    avg_r = tot_r / total_val_samples
    avg_f = tot_f / total_val_samples

    print(f"Class 0: p: {avg_p[0]:.3f}, r: {avg_r[0]:.3f}, F2: {avg_f[0]:.3f}")
    print(f"Class 1: p: {avg_p[1]:.3f}, r: {avg_r[1]:.3f}, F2: {avg_f[1]:.3f}")
    print(f"Class 2: p: {avg_p[2]:.3f}, r: {avg_r[2]:.3f}, F2: {avg_f[2]:.3f}")


def train(
    num_batches,
    num_w_points,
    model,
    name,
    num_epochs=50,
    sigma_max=0.15,
    max_modes=7,
    multiclass=False,
    normalise=None,
    neg_an=False,
):

    X, y, _, __ = generate_data(
        num_batches * 32,
        num_w_points,
        sigma_max,
        max_modes,
        multiclass=multiclass,
        normalise=normalise,
        neg_an=neg_an,
    )
    valX, valy, _, __ = generate_data(
        num_batches * 8,
        num_w_points,
        multiclass=multiclass,
        normalise=normalise,
        neg_an=neg_an,
    )
    if multiclass is False:
        res = train_model_binary(model, X, y, valX, valy, name, num_epochs)
    return res
