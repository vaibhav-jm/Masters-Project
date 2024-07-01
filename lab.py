import pydvma as dvma
import numpy as np
import torch
import modal
import matplotlib.pyplot as plt


def load_data():
    d = dvma.load_data()
    tf_data = d.tf_data_list[0]
    tf_arr = np.array(tf_data.tf_data)
    return tf_arr


def extend_lab_tf(lab_tf, max_norm=False):
    """Extend lab_tf with phase and magnitude information."""
    real_imag = modal.split_real_imag(lab_tf)
    phase = np.arctan(np.imag(lab_tf) / np.real(lab_tf))
    mag = np.abs(lab_tf)
    if max_norm is True:
        max_mag = np.max(mag)
        mag = mag / (0.95 * max_mag)
        real_imag = real_imag / (0.95 * max_mag)
    extended_op = np.concatenate(
        (real_imag, phase.reshape(-1, 1), mag.reshape(-1, 1)), axis=1
    )
    return extended_op


def load_sampling_data(
    cutoff: int = -1, f_s: float = 3000, tf_type: str = "vel", num_tfs: int = 1
) -> tuple[np.ndarray, np.array]:
    """
    Function to import the TF and frequency axis from lab data.

    Args:
        cutoff: upper limit on sample number (only predict on certain number of measurements)
        f_s: sampling frequency
        tf_type: transfer function type (displacement, velocity or acceleration)
        num_tfs: the number of transfer functions to import
    """
    assert tf_type in ["disp", "vel", "acc"]

    d = dvma.load_data()
    # freqs = d.tf_data_list[0].freq_axis
    faxis = np.fft.rfftfreq(2 * d.tf_data_list[0].tf_data.shape[0] - 1, d=1 / f_s)
    waxis = 2 * np.pi * faxis[:cutoff]

    if num_tfs > 1:
        tf_dict = {}
        for i in range(num_tfs):
            if cutoff == -1:
                tf_dict[f"tf_{i}"] = d.tf_data_list[i].tf_data[1:].reshape(-1)
            else:
                tf_dict[f"tf_{i}"] = d.tf_data_list[i].tf_data[1:].reshape(-1)[:cutoff]
            if tf_type == "acc":
                waxis[0] = (
                    waxis[1] / 3
                )  # This is a bit dodgy, just to prevent dividing by zero
                tf_dict[f"tf_{i}"] = tf_dict[f"tf_{i}"] / (1j * waxis)
            elif tf_type == "disp":
                tf_dict[f"tf_{i}"] = tf_dict[f"tf_{i}"] * (1j * waxis)
        return tf_dict, waxis
    else:
        tf = d.tf_data_list[0].tf_data[:cutoff]
        if tf_type == "acc":
            tf = tf / (1j * waxis)
        elif tf_type == "disp":
            tf = tf * waxis
        return tf, waxis


def lab_predictions(
    model,
    tf_arr,
    multiclass=False,
    normalise=None,
    extended=False,
    w=None,
    max_norm=False,
    phase=False,
    logmag=False,
    scaled_logmag=False,
    plot_tf=True,
    name=None,
):
    """Predictions from a trained model on a given transfer function."""

    if extended:
        if normalise is not None:
            tf_arr2 = normalise(tf_arr)
            real_imag = modal.split_real_imag(tf_arr2)
            extended_tf = extend_lab_tf(tf_arr)
            extended_tf[:, 0] = real_imag[:, 0]
            extended_tf[:, 1] = real_imag[:, 1]
        else:
            if max_norm:
                real_imag = modal.split_real_imag(tf_arr)
                phase = np.arctan(np.imag(tf_arr) / np.real(tf_arr))
                if phase is True:
                    phase = np.mod(phase, 2 * np.pi)  # phase is between 0 and 2pi
                mag = np.abs(tf_arr)
                max_mag = np.max(mag)
                mag = mag / (0.95 * max_mag)
                real_imag = real_imag / (0.95 * max_mag)
                if logmag is True:
                    logmagnitude = np.log10(mag)
                    if scaled_logmag is True:
                        logmagnitude = (logmagnitude - np.mean(logmagnitude)) / np.std(
                            logmagnitude
                        )
                    extended_tf = np.concatenate(
                        (
                            real_imag,
                            phase.reshape(-1, 1),
                            mag.reshape(-1, 1),
                            logmagnitude.reshape(-1, 1),
                        ),
                        axis=1,
                    )
                else:
                    extended_tf = np.concatenate(
                        (real_imag, phase.reshape(-1, 1), mag.reshape(-1, 1)), axis=1
                    )

            else:
                extended_tf = extend_lab_tf(tf_arr)
        lab_tf_tensor = torch.from_numpy(extended_tf).to(torch.float32)
    else:
        lab_tf = modal.split_real_imag(tf_arr).reshape(1, -1, 2)
        if normalise is not None:
            lab_tf = normalise(lab_tf)
        lab_tf_tensor = torch.from_numpy(lab_tf).to(torch.float32)

    model.eval()
    with torch.no_grad():
        lab_tf_output = model(lab_tf_tensor)

    if multiclass:
        test_op = np.array(lab_tf_output.numpy())
        predictions = np.argmax(test_op, axis=-1).reshape(-1)
    else:
        test_op = lab_tf_output.numpy().reshape(-1)
        predictions = (test_op > 0.0).astype(int)

    input_tf = modal.split_real_imag(tf_arr)
    y = predictions
    # print(y)

    fig, ax = plt.subplots(figsize=(7, 4))
    if w is None:
        w = np.linspace(0, 1, len(y))
    if normalise is not None:
        tf = modal.to_db(normalise(input_tf))
    elif max_norm is True:
        tf = modal.to_db(real_imag)
    else:
        tf = modal.to_db(input_tf)

    if plot_tf is True:
        if multiclass:
            segment_start = None
            segment_color = None
            ax.plot(
                w, tf, label="Transfer Function", c="black", alpha=0.9, linewidth=1.15
            )
            for i in range(len(w)):
                if predictions[i] == 1 and (
                    segment_start is None or segment_color != "#e31a1c"
                ):  # red
                    if segment_start is not None:
                        ax.plot(
                            w[segment_start:i],
                            tf[segment_start:i],
                            c=segment_color,
                            alpha=0.7,
                            linewidth=5,
                            label=f"Predicted={segment_color[-1]}",
                        )
                    segment_start = i
                    segment_color = "#e31a1c"
                elif predictions[i] == 2 and (
                    segment_start is None or segment_color != "royalblue"
                ):  # blue #1f78b4
                    if segment_start is not None:
                        ax.plot(
                            w[segment_start:i],
                            tf[segment_start:i],
                            c=segment_color,
                            alpha=0.7,
                            linewidth=5,
                            label=f"Predicted={segment_color[-1]}",
                        )
                    segment_start = i
                    segment_color = "royalblue"
                elif (
                    predictions[i] != 1
                    and predictions[i] != 2
                    and segment_start is not None
                ):
                    if i == segment_start + 1:
                        ax.scatter(
                            w[segment_start],
                            tf[segment_start],
                            c=segment_color,
                            alpha=0.7,
                            s=50,
                        )
                    ax.plot(
                        w[segment_start:i],
                        tf[segment_start:i],
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
                    tf[segment_start:i],
                    c=segment_color,
                    alpha=0.7,
                    linewidth=5,
                    label=f"Predicted={segment_color[-1]}",
                )

            # legend_elements = []
            # legend_elements.append(plt.Line2D([0], [0], color='black', label='Transfer Function'))
            # legend_elements.append(plt.Line2D([0], [0], color='#e31a1c', alpha=0.7, linewidth=5, label='Model Predictions (Class 1)'))
            # legend_elements.append(plt.Line2D([0], [0], color='royalblue', alpha=0.7, linewidth=5, label='Model Predictions (Class 2)'))

            # fig.legend(handles=legend_elements, loc='upper center',
            #         bbox_to_anchor=(0.5, 1.),
            #         # bbox_to_anchor=(0.5, -0),
            #         bbox_transform=fig.transFigure,
            #         ncol=3,
            #         fancybox=True,
            #         #   shadow=True,
            #             # frameon=True,
            #             )

        else:
            segment_start = None
            for i in range(len(w)):
                if predictions[i] == 1:
                    if segment_start is None:
                        segment_start = i
                elif segment_start is not None:
                    ax.plot(
                        w[segment_start:i],
                        tf[segment_start:i],
                        c="red",
                        alpha=0.7,
                        linewidth=5,
                        label="Predicted=1",
                    )
                    segment_start = None
            ax.plot(
                w, tf, label="Transfer Function", c="blue", alpha=0.7, linewidth=1.15
            )

        # if multiclass:
        # ax.scatter(w[y == 2], tf[y == 2], c='orange', marker='o', label=r'Model Predictions (Class 2)')
        if w is not None:
            ax.set_xlabel("Frequency (rad/s)")
        else:
            ax.set_xlabel("Normalised Frequency")
        ax.set_ylabel("Magnitude (dB)")
        # ax.legend()
        if name is not None:
            plt.savefig(f"./Figs/{name}.pdf")
        plt.show()

    return test_op
