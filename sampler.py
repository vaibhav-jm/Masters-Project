import emcee
import lab
import corner
import modal
import numpy as np
import matplotlib.pyplot as plt
import pydvma as dvma
import pints
from numba import jit
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


class ModalSumDB(pints.ForwardModel):
    """
    Class to handle the forward model for the modal sum model. This model tries to
    match the predicted TF magnitudes to the true TF magnitudes (in dB).
    """

    def __init__(self, param, phase_flag):
        super().__init__()
        self.dimension = len(param) - 1
        self.phase_flag = phase_flag

    # @jit(nopython=True)
    def simulate(self, thetas, w):
        tf = np.zeros_like(w, dtype=np.complex64)
        if self.phase_flag:
            n = len(thetas) // 4
            w_n, a_n, z_n, p_n = (
                thetas[:n],
                thetas[n : 2 * n],
                thetas[2 * n : 3 * n],
                thetas[3 * n : 4 * n],
            )
            for i in range(len(a_n)):
                tf += (
                    a_n[i]
                    * 1j
                    * w
                    * np.exp(p_n[i] * 1j)
                    / (w_n[i] ** 2 - w**2 + 2j * z_n[i] * w * w_n[i])
                )
        else:
            n = len(thetas) // 3
            w_n, a_n, z_n = thetas[:n], thetas[n : 2 * n], thetas[2 * n : 3 * n]
            for i in range(len(a_n)):
                tf += a_n[i] * 1j * w / (w_n[i] ** 2 - w**2 + 2j * z_n[i] * w * w_n[i])

        # tf[0] += 0.01 + 0.01j
        tf[0] = tf[1]
        tf_db = 20 * np.log10(np.sqrt(tf.real**2 + tf.imag**2))

        return tf_db

    def n_parameters(self):
        return self.dimension


class ModalSumMultiOut(pints.ForwardModel):
    """
    Class to handle the forward model for the modal sum model. This model tries to
    match both the real and imaginary parts of the predicted TF to the true TF.
    """

    def __init__(self, param):
        super().__init__()
        self.dimension = len(param) - 2

    # @jit(nopython=True)
    def simulate(self, thetas, w):
        n = (len(thetas) - 1) // 3
        w_n, a_n, z_n = thetas[:n], thetas[n : 2 * n], thetas[2 * n : 3 * n]
        tf = np.zeros_like(w, dtype=np.complex64)
        for i in range(len(a_n)):
            tf += a_n[i] * 1j * w / (w_n[i] ** 2 - w**2 + 2j * z_n[i] * w * w_n[i])
        return modal.split_real_imag(tf)

    def n_parameters(self):
        return self.dimension

    def n_outputs(self):
        return 2


class Sampler:
    def __init__(self, model, raw_tf, w=None):
        """
        Initialize the Sampler class.

        Parameters
        ----------
        model : torch.nn.Module
            The trained model to be used for sampling.
        raw_tf : np.ndarray
            The raw transfer function to be used for sampling.
            Shape (n, ) where n is the number of frequency points.
            Each entry is a complex number.
        w : np.ndarray, optional
            The frequency points at which the transfer function is sampled.
            Shape (n, ) where n is the number of frequency points.
        """
        self.model = model
        self.raw_tf = raw_tf
        if w is None:
            self.w = np.linspace(0, 1, len(raw_tf))
            self.normalised_freq = True
        else:
            self.w = w
            self.normalised_freq = False
        # self.tf_db = modal.to_db(self.raw_tf)
        self.tf_db = 20 * np.log10(np.sqrt(raw_tf.real**2 + raw_tf.imag**2))
        self.real_tf = np.real(self.raw_tf)
        self.run_flag = False

    def run_sampler(
        self,
        nwalkers: int = 100,
        nsteps: int = 1000,
        plot_predictions: bool = True,
        reparamaterize: bool = False,
        prior: str = "uniform",
        sampler: str = "emcee",
        pints_sampler=pints.EmceeHammerMCMC,
        phase: bool = False,
        sample_cutoff: int = -1,
        return_results: bool = False,
        multi_output: bool = False,
        max_opt_its: int = 8000,
    ) -> None:
        """
        Runs the sampler.

        Parameters
        ----------
        nwalkers : int
            Number of walkers to use in the sampler.
        nsteps : int
            Number of steps to run the sampler for.
        plot_predictions : bool
            If True, plot the predictions of the model before sampling.
        reparamaterize : bool
            If True, reparamaterize the parameters. (This is not used.)
        prior : str
            The type of prior to use. Either 'uniform' or 'normal'.
        sampler : str
            The type of sampler to use. Either 'emcee' or 'pcn' or 'pints'.
            Note the 'emcee' implementation here is slow in high dimensions,
            instead use 'pints' with the approproate pints sampler.
        pints_sampler : pints.Sampler
            The pints sampler to use, if sampler is 'pints'.
        phase : bool
            If True, complex modes are allowed and modal phase is sampled.
        sample_cutoff : int
            The frequency index at which to cut off the TF's. Note that the
            *entire* TF is still passed to the model to predict modes.
        return_results : bool
            If True, return the results of the MCMC run.
        multi_output : bool
            If True, use the multi-output model. This matches the real and
            imaginary parts of the TF instead of the magnitude in dB.
        max_opt_its : int
            The maximum number of iterations for the ML optimisation. This
            needs to be adjusted appropriately for the problem.
        """

        assert prior in [
            "uniform",
            "normal",
        ], "Prior type must be 'uniform' or 'normal'."
        assert sampler in [
            "emcee",
            "pcn",
            "pints",
        ], "Sampler type must be 'emcee' or 'pcn' or 'pints'. "

        # assuming multiclass model
        raw_predictions = self.get_lab_predictions(plot_tf=plot_predictions)
        self.predictions = np.argmax(raw_predictions, axis=-1).reshape(-1)

        self.sample_cutoff = sample_cutoff
        self.predictions = self.predictions[:sample_cutoff]

        self.reparamaterize = reparamaterize
        self.prior_type = prior
        self.sampler_type = sampler
        self.phase_flag = phase

        log_prior = self.get_log_prior()  # obtains self.mins and self.maxs
        self.log_prior = log_prior
        log_posterior = self.get_log_posterior()

        self.burned_flag = False
        self.labels = self._get_labels()

        if self.sampler_type == "emcee":
            p0 = np.random.uniform(self.mins, self.maxs, size=(nwalkers, self.ndim))
            # with Pool() as pool:
            self.sampler = emcee.EnsembleSampler(
                nwalkers,
                self.ndim,
                log_posterior,
                # pool=pool,
                # args=(self.w, self.raw_tf),
            )
            self.sampler.run_mcmc(p0, nsteps, progress=True)
            self.run_flag = True
            self.samples = self.sampler.get_chain()
            print(self.samples.shape)
            self.flat_samples_all = self.sampler.get_chain(flat=True)

        elif self.sampler_type == "pcn":
            mu = (self.mins + self.maxs) / 2
            s = np.maximum(self.maxs - mu, mu - self.mins)
            std = s / 2
            K = np.diag(std**2)  # Prior covariance
            self.samples = np.zeros((nsteps, nwalkers, self.ndim))
            for walker in tqdm(range(nwalkers)):
                p0 = np.random.uniform(self.mins, self.maxs, size=self.ndim)
                self.samples[:, walker, :], acc = self.pcn(p0, K, nsteps, 0.008)
            self.run_flag = True
            self.flat_samples_all = self.samples.reshape(-1, self.ndim)
            print(self.samples.shape, acc)

        elif self.sampler_type == "pints":
            self.mins[-1] = 0  # Set lower bound for noise to 0
            self.maxs[-1] = 10  # Set upper bound for noise to 10

            if multi_output:
                problem = pints.MultiOutputProblem(
                    ModalSumMultiOut(self.mins),
                    self.w,
                    modal.split_real_imag(self.raw_tf),
                )
            else:
                problem = pints.MultiOutputProblem(
                    ModalSumDB(self.mins, self.phase_flag), self.w, self.tf_db
                )  # Matching dB setting
            log_likelihood = pints.GaussianLogLikelihood(problem)

            # Define boundaries for ML estimate
            wmin, amin, zmin = (
                self.mins[: self.n_modes],
                self.mins[self.n_modes : 2 * self.n_modes],
                self.mins[2 * self.n_modes : 3 * self.n_modes],
            )
            wmax, amax, zmax = (
                self.maxs[: self.n_modes],
                self.maxs[self.n_modes : 2 * self.n_modes],
                self.maxs[2 * self.n_modes : 3 * self.n_modes],
            )

            # factors used here to expand ML optimisation search space
            if not self.phase_flag:
                lb = np.concatenate(
                    (
                        wmin / 1.2,
                        -2 * np.maximum(np.abs(amin), np.abs(amax)),
                        [0] * self.n_modes,
                        [0],
                    )
                )
                ub = np.concatenate(
                    (
                        wmax * 1.2,
                        2 * np.maximum(np.abs(amin), np.abs(amax)),
                        [1.5] * self.n_modes,
                        [10],
                    )
                )
            else:
                lb = np.concatenate(
                    (
                        wmin / 1.2,
                        -2 * np.maximum(np.abs(amin), np.abs(amax)),
                        [0] * self.n_modes,
                        [-np.pi / 4] * self.n_modes,
                        [0],
                    )
                )
                ub = np.concatenate(
                    (
                        wmax * 1.2,
                        2 * np.maximum(np.abs(amin), np.abs(amax)),
                        [1.5] * self.n_modes,
                        [np.pi / 4] * self.n_modes,
                        [10],
                    )
                )
            boundaries = pints.RectangularBoundaries(lb, ub)

            # Compute maximum likelihood estimate
            opt = pints.OptimisationController(
                log_likelihood,
                x0=np.random.uniform(self.mins, self.maxs),
                boundaries=boundaries,
            )
            opt.set_max_iterations(max_opt_its)
            x0, _ = opt.run()
            # x0 = pints.optimise(log_likelihood, np.random.uniform(self.mins, self.maxs))[0]

            for i in range(len(x0)):
                # print(f"{self.mins[i]}, {x0[i]}, {self.maxs[i]}")
                if x0[i] > self.maxs[i]:
                    if np.sign(x0[i]) == 1:
                        self.maxs[i] = x0[i] * 2
                    else:
                        self.maxs[i] = x0[i] / 2
                elif x0[i] < self.mins[i]:
                    if np.sign(x0[i]) == 1:
                        self.mins[i] = x0[i] / 2
                    else:
                        self.mins[i] = x0[i] * 2

            if self.prior_type == "uniform":
                log_prior = pints.UniformLogPrior(self.mins, self.maxs)

            elif self.prior_type == "normal":
                mu = (self.mins + self.maxs) / 2
                s = np.minimum(self.maxs - mu, mu - self.mins)
                std = s / 2  # 2 standard deviations
                cov = np.diag(std**2)
                log_prior = pints.MultivariateGaussianLogPrior(mu, cov)

            log_posterior = pints.LogPosterior(log_likelihood, log_prior)

            # Add noise to the maximum likelihood estimate
            ml_noise_factor = 4.0  # decrease for wider initial spread of walkers
            sigma = np.minimum(x0 - self.mins, self.maxs - x0) / ml_noise_factor

            p0 = np.random.normal(loc=x0, scale=sigma, size=(nwalkers, len(x0)))
            p0 = np.clip(p0, self.mins, self.maxs)

            # Print log-likelihood for each initial point
            for i in range(nwalkers):
                # print(f"Log-likelihood for walker {i + 1}: {log_posterior(p0[i, :])}")
                while not np.isfinite(log_posterior(p0[i, :])):
                    p0[i, :] = np.random.normal(loc=x0, scale=sigma)

            mcmc = pints.MCMCController(
                log_posterior, nwalkers, p0, method=pints_sampler
            )
            mcmc.set_max_iterations(nsteps)
            mcmc.set_log_to_screen(True)
            mcmc.set_log_interval(1000)
            print("Running...")
            self.samples = mcmc.run()
            self.samples = np.transpose(self.samples, (1, 0, 2))
            self.flat_samples_all = self.samples.reshape(-1, self.ndim)
            self.run_flag = True
            # pints.plot.trace(self.samples, parameter_names=self.labels)
            # plt.show()
            if return_results:
                results = pints.MCMCSummary(
                    chains=self.samples, time=mcmc.time(), parameter_names=self.labels
                )
                print(results)

    def generate_samples(self, args):
        walker, nsteps, ndim, mins, maxs, K = args
        p0 = np.random.uniform(mins, maxs, size=ndim)
        samples, acc = self.pcn(p0, K, nsteps, 0.008)
        return samples, acc

    def pcn(self, theta0, K, n_iters, beta):
        """pCN MCMC method for sampling from pdf defined by log_prior and log_likelihood.
        Inputs:
            log_likelihood - log-likelihood function
            theta0 - initial sample
            K - prior covariance
            n_iters - number of samples
            beta - step-size parameter
        Returns:
            X - samples from target distribution
            acc/n_iters - the proportion of accepted samples"""
        X = []
        acc = 0
        u_prev = theta0
        N = theta0.shape[0]
        w = self.w
        tf = self.raw_tf
        ll_prev = self.log_likelihood(u_prev, w, tf)

        Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))

        for i in range(n_iters):
            u_new = np.sqrt(1 - beta**2) * u_prev + beta * Kc @ np.random.randn(
                N,
            )
            ll_new = self.log_likelihood(u_new, w, tf)

            log_alpha = min(ll_new - ll_prev, 0.0)
            log_u = np.log(np.random.random())

            # Accept/Reject
            accept = log_alpha >= log_u
            if accept:
                acc += 1
                X.append(u_new)
                u_prev = u_new
                ll_prev = ll_new
            else:
                X.append(u_prev)

        return np.array(X), acc / n_iters

    def get_samples(self) -> np.ndarray:
        """
        Getter for the samples from the sampler."""
        if self.run_flag:
            return self.samples
        else:
            raise ValueError("Sampler has not been run yet.")

    def plot_hist(self, num_modes, figsize=(15, 10), name=None):
        """
        Plot histograms of the samples.

        *Note this does not yet support complex amplitude sampling.*
        """

        fig, axs = plt.subplots(num_modes, 3, figsize=figsize, tight_layout=True)
        for i in range(num_modes):
            axs[0, i].hist(
                self.flat_samples[:, i], bins=100, color="k", histtype="step"
            )
            axs[1, i].hist(
                self.flat_samples[:, i + self.n_modes],
                bins=100,
                color="k",
                histtype="step",
            )
            axs[2, i].hist(
                self.flat_samples[:, i + 2 * self.n_modes],
                bins=100,
                color="k",
                histtype="step",
            )
            axs[0, i].yaxis.set_major_formatter(plt.NullFormatter())
            axs[1, i].yaxis.set_major_formatter(plt.NullFormatter())
            axs[2, i].yaxis.set_major_formatter(plt.NullFormatter())
            q1 = corner.quantile(self.flat_samples[:, i], [0.16, 0.5, 0.84])
            q2 = corner.quantile(
                self.flat_samples[:, i + self.n_modes], [0.16, 0.5, 0.84]
            )
            q3 = corner.quantile(
                self.flat_samples[:, i + 2 * self.n_modes], [0.16, 0.5, 0.84]
            )
            axs[0, i].set_title(
                rf"$\omega_{i+1} = {q1[1]:.2f}^{{+{q1[2]-q1[1]:.2f}}}_{{-{q1[1]-q1[0]:.2f}}}$"
            )
            axs[1, i].set_title(
                rf"$a_{i+1} = {q2[1]:.2f}^{{+{q2[2]-q2[1]:.3f}}}_{{-{q2[1]-q2[0]:.3f}}}$"
            )
            axs[2, i].set_title(
                rf"$\zeta_{i+1} = {q3[1]:.4f}^{{+{q3[2]-q3[1]:.4f}}}_{{-{q3[1]-q3[0]:.4f}}}$"
            )
        if name is not None:
            plt.savefig(f"./Figs/{name}.pdf")
        plt.show()

        if self.run_flag:
            fig, axs = plt.subplots(num_modes, 3, figsize=figsize)
            for i in range(num_modes):
                axs[i, 0].hist(
                    self.samples[:, :, i], bins=100, color="black", alpha=0.3
                )
                axs[i, 0].set_title(f"Mode {i+1} - $\omega$")
                axs[i, 1].hist(
                    self.samples[:, :, i + num_modes],
                    bins=100,
                    color="black",
                    alpha=0.3,
                )
                axs[i, 1].set_title(f"Mode {i+1} - $a$")
                axs[i, 2].hist(
                    self.samples[:, :, i + 2 * num_modes],
                    bins=100,
                    color="black",
                    alpha=0.3,
                )
                axs[i, 2].set_title(f"Mode {i+1} - $\zeta$")
            plt.show()
        else:
            raise ValueError("Sampler has not been run yet.")

    def plot_state_evolution(
        self, mode_num=None, figsize=None, ypos=-0.1, name=None, mode_truths=None
    ) -> None:
        if self.run_flag:
            if mode_num:
                if isinstance(mode_num, int):
                    if figsize is None:
                        figsize = (6, 3)
                    if self.phase_flag:
                        indices = [
                            mode_num - 1,
                            mode_num - 1 + self.n_modes,
                            mode_num - 1 + 2 * self.n_modes,
                            mode_num - 1 + 3 * self.n_modes,
                        ]
                        labels = [
                            rf"$\omega_{mode_num}$",
                            rf"$a_{mode_num}$",
                            rf"$\zeta_{mode_num}$",
                            rf"$\phi_{mode_num}$",
                        ]
                        max_range = 4
                    else:
                        labels = [
                            rf"$\omega_{mode_num}$",
                            rf"$a_{mode_num}$",
                            rf"$\zeta_{mode_num}$",
                        ]
                        indices = [
                            mode_num - 1,
                            mode_num - 1 + self.n_modes,
                            mode_num - 1 + 2 * self.n_modes,
                        ]
                        max_range = 3
                    fig, axs = plt.subplots(max_range, figsize=figsize, sharex=True)
                    for i in range(max_range):
                        ax = axs[i]
                        ax.plot(self.samples[:, :, indices[i]], "k", alpha=0.3)
                        ax.set_ylabel(labels[i])
                        if mode_truths:
                            ax.axhline(mode_truths[i], color="r", linestyle="--")
                        ax.yaxis.set_label_coords(ypos, 0.5)
                    axs[-1].set_xlabel("Iteration")
                    # plt.show()
            else:
                if figsize is None:
                    figsize = (10, self.ndim)
                if self.sampler_type == "emcee":
                    max_range = self.ndim - 1
                else:
                    max_range = self.ndim
                fig, axes = plt.subplots(
                    max_range, figsize=figsize, sharex=True
                )  # , gridspec_kw={'left': 0.2, 'bottom': 0.2}
                # fig.subplots_adjust(left=0.2, bottom=0.2)
                # fig.tight_layout(rect=(0.2,0.2,1,1))
                for i in range(max_range):
                    ax = axes[i]
                    ax.plot(self.samples[:, :, i], "k", alpha=0.3)
                    ax.set_xlim(0, len(self.samples))
                    ax.set_ylabel(self.labels[i])
                    ax.yaxis.set_label_coords(ypos, 0.5)
                axes[-1].set_xlabel("Iteration")
                # plt.subplots_adjust(left=0.1, bottom=0.3, right=1, top=0.7, wspace=0, hspace=0.05)
            if name:
                plt.savefig(f"./Figs/{name}.pdf")
            plt.show()
        else:
            raise ValueError("Sampler has not been run yet.")

    def burn_in(self, nburn: int) -> None:
        if self.run_flag:
            if self.sampler_type == "emcee":
                self.flat_samples = self.sampler.get_chain(discard=nburn, flat=True)
                self.log_probs = self.sampler.get_log_prob(discard=nburn, flat=True)
                self.burned_flag = True
                print(self.flat_samples.shape)
            else:
                self.samples = self.samples[nburn:, :, :]
                self.flat_samples = self.samples.reshape(-1, self.ndim)
                # self.flat_samples = self.samples[nburn:, :, :].reshape(-1, self.ndim)
                self.burned_flag = True

        else:
            raise ValueError("Sampler has not been run yet.")

    def plot_histograms(self) -> None:
        if self.run_flag:
            pints.plot.histogram(self.samples, parameter_names=self.labels)

    def plot_corner(
        self,
        truths=None,
        mode_num=None,
        figsize=(6, 6),
        tight=True,
        lpad=1,
        name=None,
        overlay=False,
    ) -> None:
        """
        Plot the corner plot of the samples.

        Parameters
        ----------
        truths : list, optional
            The true values of the parameters.
        mode_num : optional
            The number of the mode to plot, or a list of the numbers of the
            modes to plot.
        """
        if self.burned_flag:
            flat_samples = self.flat_samples
        else:
            flat_samples = self.flat_samples_all

        if self.run_flag:
            if overlay:
                indices_0 = [0, self.n_modes, 2 * self.n_modes]
                data_to_plot_0 = flat_samples[:, indices_0]
                labels = [rf"$\omega_{1}$", rf"$a_{1}$", rf"$\zeta_{1}$"]
                if truths is not None:
                    selected_truths = [truths[i] for i in indices_0]
                else:
                    selected_truths = None
                plt.figure(tight_layout=tight, figsize=figsize)
                fig = corner.corner(
                    data=data_to_plot_0,
                    labels=labels,
                    truths=selected_truths,
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True,
                    title_fmt=".4f",
                    title_kwargs={"fontsize": 16},
                    label_kwargs={"fontsize": 16, "labelpad": lpad},
                    truth_color="r",
                    plot_contours=True,
                )
                colors = ["r", "b"]
                for i in range(1, self.n_modes + 1):
                    indices = [i, i + self.n_modes, i + 2 * self.n_modes]
                    data_to_plot = flat_samples[:, indices]
                    labels = [rf"$\omega_{i}$", rf"$a_{i}$", rf"$\zeta_{i}$"]
                    corner.corner(
                        data=data_to_plot,
                        labels=labels,
                        truths=selected_truths,
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_fmt=".4f",
                        title_kwargs={"fontsize": 16},
                        label_kwargs={"fontsize": 16, "labelpad": lpad},
                        truth_color="r",
                        plot_contours=True,
                        fig=fig,
                        color=colors[i % 2],
                    )
                if name:
                    plt.savefig(f"./Figs/{name}_corner_overlay.pdf")
                plt.show()

            if mode_num:
                if isinstance(mode_num, list):
                    for mode in mode_num:
                        mode = int(mode)
                        indices = [
                            mode - 1,
                            mode - 1 + self.n_modes,
                            mode - 1 + 2 * self.n_modes,
                        ]
                        if truths is not None:
                            selected_truths = [truths[i] for i in indices]
                        else:
                            selected_truths = None

                        data_to_plot = flat_samples[:, indices]
                        labels = [
                            rf"$\omega_{mode}$",
                            rf"$a_{mode}$",
                            rf"$\zeta_{mode}$",
                        ]
                        plt.figure(tight_layout=tight, figsize=figsize)
                        fig = corner.corner(
                            data=data_to_plot,
                            labels=labels,
                            truths=selected_truths,
                            quantiles=[0.16, 0.5, 0.84],
                            show_titles=True,
                            title_fmt=".4f",
                            title_kwargs={"fontsize": 16},
                            label_kwargs={"fontsize": 16, "labelpad": lpad},
                            truth_color="r",
                            plot_contours=True,
                        )
                        if name:
                            plt.savefig(f"./Figs/{name}_corner_{mode}.pdf")
                        plt.show()
                elif isinstance(mode_num, int):
                    if self.phase_flag:
                        indices = [
                            mode_num - 1,
                            mode_num - 1 + self.n_modes,
                            mode_num - 1 + 2 * self.n_modes,
                            mode_num - 1 + 3 * self.n_modes,
                        ]
                        labels = [
                            rf"$\omega_{mode_num}$",
                            rf"$a_{mode_num}$",
                            rf"$\zeta_{mode_num}$",
                            rf"$\phi_{mode_num}$",
                        ]
                    else:
                        indices = [
                            mode_num - 1,
                            mode_num - 1 + self.n_modes,
                            mode_num - 1 + 2 * self.n_modes,
                        ]
                        labels = [
                            rf"$\omega_{mode_num}$",
                            rf"$a_{mode_num}$",
                            rf"$\zeta_{mode_num}$",
                        ]

                    if truths is not None:
                        selected_truths = [truths[i] for i in indices]
                    else:
                        selected_truths = None

                    data_to_plot = flat_samples[:, indices]

                    fig = corner.corner(
                        data=data_to_plot,
                        labels=labels,
                        truths=selected_truths,
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_fmt=".4f",
                        title_kwargs={"fontsize": 21},
                        label_kwargs={"fontsize": 21, "labelpad": lpad},
                        hist2d_kwargs={"data_kwargs": {"alpha": 0.1}},
                        truth_color="r",
                        plot_contours=True,
                    )
                    if name:
                        plt.savefig(f"./Figs/{name}_corner_{mode_num}.pdf")
                    plt.show()
            else:
                plt.figure(tight_layout=tight)
                fig = corner.corner(
                    data=flat_samples,
                    labels=self.labels,
                    truths=truths,
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True,
                    title_fmt=".4f",
                    title_kwargs={"fontsize": 14},
                    truth_color="r",
                    plot_contours=True,
                )
                plt.show()
        else:
            raise ValueError("Sampler has not been run yet.")

    def plot_posterior_samples(
        self,
        n_samples: int = 100,
        posterior_mean: bool = False,
        posterior_error: bool = False,  # implementation for this is wrong
        figsize=(7, 4),
        name=None,
        ws=None,
        y=None,
        markevery=5,
        tf_alpha=1.0,
        zoomed=False,
        xlims=None,  # x and y lims for zoomed box
        ylims=None,
    ) -> None:
        """
        Plot the posterior samples of the transfer function, alongside
        the original transfer function.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to plot.
        posterior_mean : bool, optional
            If True, plot the posterior mean of the transfer function.
        posterior_error : bool, optional
            If True, plot += 1 standard deviation error bars around the posterior mean.
            * This is not implemented correctly yet. *
        figsize : tuple, optional
        name : str, optional
            Name of the file to save the plot.
        ws : list, optional
            List of true natural frequencies to plot as vertical lines.
        y : np.ndarray, optional
            List of labels for the TF.
            * Multiclass not supported, these must be 0 or 1 *
        markevery : int, optional
            Marker every n points on the TF.
        tf_alpha : float, optional
            Opacity of the true TF.
        zoomed : bool, optional
            If True, overlay  a zoomed in version of the TF of the plot.
        """
        if self.burned_flag:
            flat_samples = self.flat_samples
            # log_probs = self.log_probs
        else:
            flat_samples = self.flat_samples_all
            # log_probs = self.get_log_probs()
        # flat_samples = flat_samples[
        #     log_probs > 0
        # ]  # only consider samples with positive log probs (converged)
        inds = np.random.randint(len(flat_samples), size=n_samples)
        n = len(self.mins - 1) // 3
        fig, ax = plt.subplots(figsize=figsize)
        # plt.plot(self.w[:self.sample_cutoff], self.tf_db[:self.sample_cutoff], color="blue", label="True TF")

        if zoomed:
            axins = zoomed_inset_axes(ax, zoom=2, loc="lower right")  # loc=4

        for ind in inds:
            sample = flat_samples[ind]
            if self.reparamaterize:
                sample = self._convert_theta(sample)
            ax.plot(
                self.w[: self.sample_cutoff],
                ModalSumDB(self.mins, self.phase_flag).simulate(
                    sample, self.w[: self.sample_cutoff]
                ),
                color="red",
                alpha=0.1,
            )
            if zoomed:
                axins.plot(
                    self.w[: self.sample_cutoff],
                    ModalSumDB(self.mins, self.phase_flag).simulate(
                        sample, self.w[: self.sample_cutoff]
                    ),
                    color="red",
                    alpha=0.1,
                )

        ax.plot(
            self.w[: self.sample_cutoff],
            self.tf_db[: self.sample_cutoff],
            color="blue",
            label="True TF",
            marker="o",
            ms=2,
            linewidth=0.75,
            markevery=markevery,
            alpha=tf_alpha,
        )
        if zoomed:
            axins.plot(
                self.w[: self.sample_cutoff],
                self.tf_db[: self.sample_cutoff],
                color="blue",
                label="True TF",
                marker="o",
                ms=2,
                linewidth=0.5,
                markevery=markevery,
                alpha=tf_alpha,
            )

        if y is not None:
            mask = np.zeros_like(self.w)
            mask[y == 1] = 1
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

        if zoomed:
            axins.set_xlim(xlims[0], xlims[1])  # Limit the x-axis range for the inset
            axins.set_ylim(ylims[0], ylims[1])  # Limit the y-axis range for the inset

            # Hide ticks on inset
            axins.xaxis.set_visible(False)
            axins.yaxis.set_visible(False)

            # Draw lines to indicate the zoomed area
            mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="0.5")

        if posterior_mean:
            mean = np.mean(flat_samples, axis=0)
            if self.reparamaterize:
                mean = self._convert_theta(mean)
            mean_tf = modal.modal_sum_fast(
                self.w, mean[n : 2 * n], mean[2 * n : 3 * n], mean[:n]
            )
            if posterior_error:
                plt.clf()  # clear the figure
                std = np.std(flat_samples, axis=0)
                if self.reparamaterize:
                    std = self._convert_theta(std)
                min_tf = modal.modal_sum_fast(
                    self.w,
                    (mean - std)[n : 2 * n],
                    (mean - std)[2 * n : 3 * n],
                    (mean - std)[:n],
                )
                max_tf = modal.modal_sum_fast(
                    self.w,
                    (mean + std)[n : 2 * n],
                    (mean + std)[2 * n : 3 * n],
                    (mean + std)[:n],
                )
                plt.fill_between(
                    self.w,
                    modal.to_db(min_tf),
                    modal.to_db(max_tf),
                    color="black",
                    alpha=0.2,
                    label=r"$\pm 1$ Standard deviation",
                )
                plt.plot(
                    self.w, modal.to_db(mean_tf), color="black", label="Posterior Mean"
                )
                plt.plot(self.w, self.tf_db, color="blue", label="True TF", alpha=0.5)
            else:
                plt.plot(
                    self.w, modal.to_db(mean_tf), color="green", label="Posterior Mean"
                )
        # plt.legend()
        if self.normalised_freq:
            ax.set_xlabel("Normalised Frequency")
        else:
            ax.set_xlabel("Frequency (rad/s)")
        ax.set_ylabel("Magnitude (dB)")
        if name:
            plt.savefig(f"./Figs/{name}.pdf")

        plt.show()

    def get_log_prior(self) -> callable:
        frequency_groups, tf_groups, real_tf_groups = self._extract_groups()
        priors, self.mins, self.maxs = self._generate_prior_groups(
            frequency_groups, tf_groups, real_tf_groups
        )
        mins = self.mins
        maxs = self.maxs
        self.ndim = len(self.mins)
        self.n_modes = (self.ndim - 1) // 4 if self.phase_flag else (self.ndim - 1) // 3

        if self.prior_type == "uniform":

            @jit(nopython=True)
            def log_prior(theta) -> float:
                if np.all(mins < theta) and np.all(theta < maxs):
                    return 0.0
                else:
                    return -np.inf

        elif self.prior_type == "normal":
            mu = (mins + maxs) / 2
            # s = np.maximum(maxs - mu, mu - mins)
            s = np.minimum(maxs - mu, mu - mins)
            std = s / 2  # 2 standard deviations

            @jit(nopython=True)
            def log_prior(theta) -> float:
                diff = theta - mu
                return -0.5 * np.dot(diff, 1 / std**2 * diff)
                # return -0.5 * diff.T @ (1 / std ** 2) @ diff

        return log_prior

    @staticmethod
    @jit(nopython=True)
    def log_likelihood(theta, w, tf) -> float:
        var = np.exp(theta[-1]) ** 2
        n = (len(theta) - 1) // 3
        # model = modal.modal_sum(
        #     w, theta[n : 2 * n], theta[2 * n : 3 * n], theta[:n], logsigma=None
        # )[0]
        model = modal.modal_sum_fast(
            w, theta[n : 2 * n], theta[2 * n : 3 * n], theta[:n]
        )
        # tf_db = 20 * np.log10(np.sqrt(tf.real ** 2 + tf.imag ** 2))
        # model_db = 20 * np.log10(np.sqrt(model.real ** 2 + model.imag ** 2))
        # diff = tf_db - model_db
        # squared_norm = np.sum(diff ** 2)
        # return -len(w) * np.log(2 * np.pi * var) - squared_norm / (2 * var)

        diff = modal.split_real_imag(tf - model)
        # squared_norm = np.sum(np.linalg.norm(diff, axis=1) ** 2)
        squared_norm = np.sum(diff.real**2 + diff.imag**2)
        return -len(w) * np.log(2 * np.pi * var) - squared_norm / (2 * var)

    def get_acceptance_fraction(self) -> float:
        if self.run_flag:
            return np.mean(self.sampler.acceptance_fraction)
        else:
            raise ValueError("Sampler has not been run yet.")

    def get_log_probs(self):
        if self.run_flag:
            return self.sampler.get_log_prob()
        else:
            raise ValueError("Sampler has not been run yet.")

    def log_posterior(self, theta) -> float:
        # w = self.w
        # tf = self.raw_tf
        lp = self.log_prior(theta)
        if lp == -np.inf:
            return -np.inf
        if self.reparamaterize:
            theta = self._convert_theta(theta)
        return lp + self.log_likelihood(theta, self.w, self.raw_tf)

    def get_log_posterior(self) -> callable:
        return self.log_posterior

    def _get_labels(self) -> list[str]:

        n = self.n_modes
        omega_labels = [rf"$\omega_{i}$" for i in range(1, n + 1)]
        if self.reparamaterize:
            a_labels = [rf"$a_{i}/(2\zeta_{i}\omega_{i})$" for i in range(1, n + 1)]
        else:
            a_labels = [rf"$a_{i}$" for i in range(1, n + 1)]
        zeta_labels = [rf"$\zeta_{i}$" for i in range(1, n + 1)]
        phase_lables = [rf"$p_{i}$" for i in range(1, n + 1)]
        if self.sampler_type == "pints":
            if self.phase_flag:
                labels = (
                    omega_labels + a_labels + zeta_labels + phase_lables + [r"$\sigma$"]
                )
            else:
                labels = omega_labels + a_labels + zeta_labels + [r"$\sigma$"]
        else:
            labels = omega_labels + a_labels + zeta_labels + [r"$\log(\sigma)$"]
        return labels

    def _extract_groups(self):
        """
        Extract the predicted frequency groups and their corresponding transfer function values.
        """
        frequency_groups = []
        tf_groups = []
        real_tf_groups = []
        group_length = 0

        for i, prediction in enumerate(self.predictions):
            if prediction == 1 or prediction == 2:  # TODO: fix this
                group_length += 1
            elif group_length > 0:  # changed 0 -> 1
                frequency_groups.append(self.w[i - group_length : i])
                tf_groups.append(self.tf_db[i - group_length : i])
                real_tf_groups.append(self.real_tf[i - group_length : i])
                group_length = 0
        # If the last element is part of a group, add it to the list
        if group_length > 0:
            frequency_groups.append(self.w[-group_length:])
            tf_groups.append(self.tf_db[-group_length:])
            real_tf_groups.append(self.real_tf[-group_length:])
        return frequency_groups, tf_groups, real_tf_groups

    def _generate_prior_groups(self, frequency_groups, tf_groups, real_tf_groups):
        """
        Use extracted frequency groups and their corresponding transfer function values to generate
        priors for each group (each mode).

        frequency_groups : list
            List of groups of frequency groups (natural frequency ranges).
        tf_groups : list
            List of groups of transfer function (in dB) values.
        real_tf_groups : list
            List of groups of real part of transfer function.
        """
        priors = []
        min_ws, max_ws = [], []
        min_as, max_as = [], []
        min_zs, max_zs = [], []
        for freq_group, tf_group, real_tf_group in zip(
            frequency_groups, tf_groups, real_tf_groups
        ):
            min_tf = np.min(tf_group)
            max_tf = np.max(tf_group)

            zeta = (freq_group[-1] - freq_group[0]) / (freq_group[-1] + freq_group[0])

            if self.reparamaterize:
                # parameters are now w_n, a_n / (2 * z_n * w_n), z_n
                # t2 is the second parameter
                zeta_min = 1e-4
                zeta_max = 0.8
                t2_min = 10 ** (min_tf / 20)
                t2_max = 10 ** (max_tf / 20)
                sign = np.sign(real_tf_group[np.argmax(np.abs(real_tf_group))])
                a_min, a_max = sign * t2_min, sign * t2_max
            else:
                # zeta_min = zeta / 2
                # zeta_max = zeta * 2
                zeta_min = zeta / 1.5
                zeta_max = zeta * 1.5
                # minimum and maximum permissible values for zeta
                if zeta_max > 0.8:
                    zeta_max = 0.8
                if zeta_min < 1e-4:
                    zeta_min = 1e-4

                if zeta_max < zeta_min:
                    zeta_max = 0.8

                if self.prior_type == "uniform":
                    scale_factor = 1
                elif self.prior_type == "normal":
                    scale_factor = 0.7
                # TODO: investigate scale factors
                a_min = 2 * zeta_min * freq_group[0] * 10 ** (min_tf / 20)
                a_max = 2 * zeta_max * freq_group[-1] * 10 ** (max_tf / 20)
                # Calculate sign of a_n using sign of real part of transfer function
                # sign = np.sign(real_tf_group[np.argmax(tf_group)])

                # pos_count = np.sum(real_tf_group > 0)
                # neg_count = np.sum(real_tf_group < 0)
                # sign = 1 if pos_count > neg_count else -1

                # sign as sign of maximum real part of transfer function in group
                sign = np.sign(real_tf_group[np.argmax(np.abs(real_tf_group))])

                if a_max < 3:
                    # if a_max < 1:
                    # include +ve and -ve values of a if a_max is less than 3
                    a_max *= scale_factor
                    a_min = -a_max
                elif sign == -1:
                    # swap a_min and a_max
                    temp = a_min
                    a_min = -1 * a_max
                    a_max = -1 * temp

                a_min *= scale_factor
                a_max *= scale_factor

                if a_max > 20:
                    a_max = 20
                if a_min < -20:
                    a_min = -20

            if a_min == 0 or a_max == 0:
                a_min = -0.1
                a_max = 0.1

            priors.append(
                [
                    freq_group[0],
                    freq_group[-1],
                    a_min,
                    a_max,
                    zeta_min,
                    zeta_max,
                ]
            )
            min_ws.append(freq_group[0] * 0.95)  # CHANGED
            max_ws.append(freq_group[-1] * 1.05)  # CHANGED
            min_as.append(a_min)
            max_as.append(a_max)
            min_zs.append(zeta_min)
            max_zs.append(zeta_max)

        sigma_min = [-5]
        sigma_max = [-1]

        if self.phase_flag:
            p_min = [-np.pi / 4] * int(len(min_ws))
            p_max = [np.pi / 4] * int(len(min_ws))
            mins = np.append(
                np.array([min_ws, min_as, min_zs, p_min]).flatten(), sigma_min
            )
            maxs = np.append(
                np.array([max_ws, max_as, max_zs, p_max]).flatten(), sigma_max
            )
        else:
            mins = np.append(np.array([min_ws, min_as, min_zs]).flatten(), sigma_min)
            maxs = np.append(np.array([max_ws, max_as, max_zs]).flatten(), sigma_max)

        print(mins)
        print(maxs)

        return priors, mins, maxs

    def get_lab_predictions(self, plot_tf=False):
        """
        Get the lab predictions for the given transfer function.

        Parameters
        ----------
        plot_tf : bool, optional
            If True, plot the transfer function.

        Returns
        -------
        np.ndarray
            The raw lab predictions for the given transfer function.
            Shape (n, 3) where n is the number of frequency points.

        """
        return lab.lab_predictions(
            model=self.model,
            tf_arr=self.raw_tf,
            multiclass=True,
            extended=True,
            w=self.w,
            max_norm=True,
            plot_tf=plot_tf,
        )

    @staticmethod
    @jit(nopython=True)
    def _convert_theta(theta):
        """
        Convert reparameterised theta values to the original parameter space.
        (w_n, a_n / (2 * z_n * w_n), z_n, logsigma) -> (w_n, a_n, z_n, logsigma)
        """
        theta_new = np.copy(theta)
        n = (len(theta) - 1) // 3
        theta_new[n : 2 * n] = theta[n : 2 * n] * 2 * theta[:n] * theta[2 * n : 3 * n]
        return theta_new
