import torch
import numpy as np
import copy

from torch import distributions, nn, optim
from pvi.models.sgp.kernels import RBFKernel, KernelList
from pvi.models.base import Model
from pvi.utils.psd_utils import psd_inverse, add_diagonal
from pvi.distributions import MultivariateGaussianDistributionWithZ

JITTER = 1e-4


class SparseGaussianProcessModel(Model, nn.Module):
    """
    Sparse Gaussian process model.
    """

    def __init__(self, **kwargs):
        Model.__init__(self, **kwargs)
        nn.Module.__init__(self)

        # Construct inducing points and kernel.
        if self.config["kernel_class"] is not None:
            self.kernel = self.config["kernel_class"](**self.config["kernel_params"])
        else:
            raise ValueError("Kernel class not specified.")

        self.hyperparameters = self.hyperparameters

    def get_default_nat_params(self):
        return {
            "np1": torch.tensor([0.0] * self.config["num_inducing"]),
            "np2": torch.tensor([-0.5] * self.config["num_inducing"]).diag_embed(),
        }

    def get_default_config(self):
        return {
            "D": None,
            "num_inducing": 50,
            "kernel_class": lambda **kwargs: RBFKernel(**kwargs),
            "kernel_params": {
                "ard_num_dims": None,  # No ARD.
                "outputscale": 1.0,
                "lengthscale": 1.0,
            },
            "optimiser_class": optim.Adam,
            "optimiser_params": {"lr": 1e-3},
            "reset_optimiser": True,
            "epochs": 100,
            "print_epochs": 10,
        }

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, hyperparameters):
        self._hyperparameters = {**self._hyperparameters, **hyperparameters}

        if hasattr(self, "kernel"):
            self.kernel.outputscale = self.hyperparameters["outputscale"]
            self.kernel.lengthscale = self.hyperparameters["lengthscale"]

    def get_default_hyperparameters(self):
        """
        :return: A default set of ε for the model.
        """
        print(self.config["kernel_params"])
        if self.config["kernel_params"]["ard_num_dims"] is not None:
            return {
                "outputscale": torch.tensor(1.0),
                "lengthscale": torch.ones(self.config["kernel_params"]["ard_num_dims"])
                * 1.0,
            }
        else:
            return {
                "outputscale": torch.tensor(1.0),
                "lengthscale": torch.tensor(1.0),
            }

    def prior(self, q=None, z=None, detach=False):
        """
        Returns the prior distribution p(u | z) at locations z.
        :param q: q(u), containing z = q.inducing_locations if not specified.
        :param z: Input locations to evaluate the prior.
        :param detach: Detach gradients before constructing q(u).
        :return: The prior p(u) = N(z; 0, Kzz).
        """
        assert not (q is not None and z is not None), "Must specify either " "q or z."
        if z is None:
            z = q.inducing_locations

        kzz = add_diagonal(self.kernel(z, z), JITTER)
        if hasattr(self.config, "P"):
            loc = torch.zeros((self.config["P"], z.shape[0]))
        else:
            loc = torch.zeros(z.shape[0])

        std = {
            "loc": loc,
            "covariance_matrix": kzz,
        }

        if detach:
            std = {k: v.detach() for k, v in std.items()}

        return MultivariateGaussianDistributionWithZ(
            std_params=std,
            inducing_locations=z,
            is_trainable=False,
            train_inducing=False,
        )

    def posterior(self, x, q, diag=True):
        """
        Returns the posterior distribution q(f) at locations x.
        :param x: The input locations to make predictions at.
        :param q: The approximate posterior distribution q(u).
        :param diag: Whether to return marginal posterior distribution.
        :return: ∫ p(f | u) q(u) df du.
        """
        qu_loc = q.std_params["loc"]
        qu_cov = q.std_params["covariance_matrix"]
        z = q.inducing_locations

        # Parameters of prior.
        kxz = self.kernel(x, z)
        kzx = kxz.transpose(-1, -2)
        kzz = add_diagonal(self.kernel(z, z), JITTER)
        ikzz = psd_inverse(kzz)

        if diag:
            kxx = self.kernel(x, x, diag=diag)

            # Predictive marginal posterior.
            kxz = kxz.unsqueeze(-2)
            kzx = kxz.transpose(-1, -2)
            qf_mu = (
                kxz.matmul(ikzz.unsqueeze(-3))
                .matmul(qu_loc.unsqueeze(-2).unsqueeze(-1))
                .squeeze(-1)
                .squeeze(-1)
            )
            qf_cov = kxx + kxz.matmul(ikzz.unsqueeze(-3)).matmul(
                (qu_cov - kzz).unsqueeze(-3)
            ).matmul(ikzz.unsqueeze(-3)).matmul(kzx).squeeze(-1).squeeze(-1)

            return distributions.Normal(qf_mu, qf_cov ** 0.5)

        else:
            kxx = add_diagonal(self.kernel(x, x, diag=diag), JITTER)

            # Predictive posterior.
            qf_mu = kxz.matmul(ikzz).matmul(qu_loc.unsqueeze(-1)).squeeze(-1)
            qf_cov = kxx + kxz.matmul(ikzz).matmul(qu_cov - kzz).matmul(ikzz).matmul(
                kzx
            )

            return distributions.MultivariateNormal(qf_mu, covariance_matrix=qf_cov)


class SparseGaussianProcessRegression(SparseGaussianProcessModel):
    """
    Sparse Gaussian process regression model using closed-form optimisation of
    q(u).
    """

    conjugate_family = MultivariateGaussianDistributionWithZ

    def __init__(self, train_sigma=True, **kwargs):
        super().__init__(**kwargs)

        self.train_sigma = train_sigma
        if self.train_sigma:
            self.register_parameter(
                "log_outputsigma",
                nn.Parameter(
                    torch.tensor(self.hyperparameters["outputsigma"]).log(),
                    requires_grad=True,
                ),
            )
        else:
            self.register_buffer(
                "log_outputsigma",
                torch.tensor(self.hyperparameters["outputsigma"]).log(),
            )

        # Set ε after model is constructed.
        self.hyperparameters = self.hyperparameters

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, hyperparameters):
        # TODO: why won't this work?
        # super().hyperparameters = hyperparameters

        self._hyperparameters = {**self._hyperparameters, **hyperparameters}

        if hasattr(self, "kernel"):
            self.kernel.outputscale = self.hyperparameters["outputscale"]
            self.kernel.lengthscale = self.hyperparameters["lengthscale"]

        if hasattr(self, "log_outputsigma"):
            if self.train_sigma:
                self.log_outputsigma.data = torch.as_tensor(
                    self.hyperparameters["outputsigma"]
                ).log()
            else:
                self.log_outputsigma = torch.as_tensor(
                    self.hyperparameters["outputsigma"]
                ).log()

    def get_default_hyperparameters(self):
        """
        :return: A default set of hyperparameters for the model.
        """
        return {
            **super().get_default_hyperparameters(),
            "outputsigma": 1.0,
        }

    def forward(self, x, q, diag=False):
        return self.posterior(x, q, diag=diag)

    def likelihood_forward(self, x, theta=None, **kwargs):
        """
        Returns the model's likelihood p(y | x).
        :param x: The input locations to make predictions at, (*, D).
        :param theta: The latent variables of the model.
        :return: p(y | x).
        """
        return distributions.Normal(x, self.outputsigma)

    def conjugate_update(self, data, q, t=None):
        """
        :param data: The local data to refine the model with.
        :param q: The current global posterior q(u).
        :param t: The local factor t(u).
        :return: q_new, t_new, the new global posterior and the new local
        contribution.
        """
        if t is not None:
            assert torch.equal(q.inducing_locations, t.inducing_locations), (
                "q and t must share the same inducing locations for "
                "conjugate update."
            )

        # Set up data etc.
        x = data["x"]
        y = data["y"]
        z = q.inducing_locations

        with torch.no_grad():
            # Parameters of prior.
            kxz = self.kernel(x, z)
            kzz = add_diagonal(self.kernel(z, z), JITTER)

            lzz = kzz.cholesky()
            ikzz = psd_inverse(chol=lzz)

            sigma = self.outputsigma
            a = kxz.matmul(ikzz)

            # Closed form solution for optimum q(u).
            # New local parameters.
            t_new_np2 = -0.5 * sigma.pow(-2) * a.T.matmul(a)
            t_new_np1 = sigma.pow(-2) * a.T.matmul(y).squeeze(-1)
            t_new_nps = {"np1": t_new_np1, "np2": t_new_np2}

        if t is None:
            # New model parameters.
            q_new_nps = {k: v + t_new_nps[k] for k, v in q.nat_params.items()}

            q_new = type(q)(
                inducing_locations=z, nat_params=q_new_nps, is_trainable=False
            )
            return q_new, None

        else:
            # New model parameters.
            q_new_nps = {
                k: v + t_new_nps[k] - t.nat_params[k] for k, v in q.nat_params.items()
            }

            q_new = type(q)(
                inducing_locations=z, nat_params=q_new_nps, is_trainable=False
            )
            t_new = type(t)(inducing_locations=z, nat_params=t_new_nps)
            return q_new, t_new

    def expected_log_likelihood(self, data, q, num_samples=None):
        # Set up data etc.
        x = data["x"]
        y = data["y"]
        z = q.inducing_locations

        kzz = add_diagonal(self.kernel(z, z), JITTER)
        ikzz = psd_inverse(kzz)
        kxz = self.kernel(x, z)
        kxx = add_diagonal(self.kernel(x, x), JITTER)

        a = kxz.matmul(ikzz)
        c = kxx - a.matmul(kxz.T)

        qf_loc = a.matmul(q.std_params["loc"])
        qf_cov = c + a.matmul(q.std_params["covariance_matrix"]).matmul(a.T)

        sigma = self.outputsigma
        dist = self.likelihood_forward(qf_loc)
        ll1 = dist.log_prob(y.squeeze())
        ll2 = -0.5 * sigma ** (-2) * qf_cov.diag()
        ll = ll1.sum() + ll2.sum()

        return ll

    def local_free_energy(self, data, q, t=None):
        """
        Returns the local variational free energy (up to an additive constant)
        of the sparse Gaussian process model under q(u), with local factor
        t(u).
        :param data: The local data.
        :param q: The current global posterior q(θ).
        :param t: The local factor t(θ) (could be cavity).
        :return: The evidence lower bound.
        """
        raise NotImplementedError

    @property
    def outputsigma(self):
        return self.log_outputsigma.exp()


class SparseGaussianProcessClassification(SparseGaussianProcessModel):
    """
    Sparse Gaussian process classification model.
    """

    conjugate_family = None

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "num_elbo_samples": 1,
            "num_predictive_samples": 1,
            "use_probit_approximation": True,
        }

    def forward(self, x, q, diag=False):
        """
        Returns the (approximate) predictive posterior distribution of a
        sparse Gaussian process.
        :param x: The input locations to make predictions at.
        :param q: The approximate posterior distribution q(u | z).
        :param diag: Whether to compute marginal posterior predictive
        distributions.
        :return: ∫ p(y | f, x) p(f | u) q(u) df du.
        """
        qf = self.posterior(x, q, diag=diag)

        if self.config["use_probit_approximation"]:
            # Use Probit approximation.
            qf_loc = qf.loc

            if str(type(qf)) == str(torch.distributions.MultivariateNormal):
                qf_scale = qf.covariance_matrix.diag() ** 0.5
            else:
                qf_scale = qf.scale

            denom = (1 + np.pi * qf_scale ** 2 / 8) ** 0.5
            logits = qf_loc / denom

            return distributions.Bernoulli(logits=logits)
        else:
            fs = qf.sample((self.config["num_predictive_samples"],))

            comp = distributions.Bernoulli(logits=fs.T)
            mix = distributions.Categorical(
                torch.ones(
                    len(fs),
                )
            )

            return distributions.MixtureSameFamily(mix, comp)

    def likelihood_forward(self, x, theta, **kwargs):
        """
        Returns the model's likelihood p(y | x).
        :param x: The input locations to make predictions at, (*, D).
        :param theta: The latent variables of the model.
        :return: p(y | x).
        """
        return distributions.Bernoulli(logits=theta)

    def conjugate_update(self, data, q, t=None):
        """
        :param data: The local data to refine the model with.
        :param q: The current global posterior q(θ).
        :param t: The local factor t(θ).
        :return: q_new, t_new, the new global posterior and the new local
        contribution.
        """
        raise NotImplementedError

    def expected_log_likelihood(self, data, q, num_samples=1):
        x = data["x"]

        qf = self.posterior(x, q, diag=True)
        fs = qf.rsample((num_samples,))
        return self.likelihood_log_prob(data, fs).mean(0)

    def local_free_energy(self, data, q, t=None):
        """
        Returns the local variational free energy (up to an additive constant)
        of the sparse Gaussian process model under q(u), with local factor
        t(u).
        :param data: The local data.
        :param q: The current global posterior q(θ).
        :param t: The local factor t(θ) (could be cavity).
        :return: The evidence lower bound.
        """
        raise NotImplementedError


class MOSparseGaussianProcess(SparseGaussianProcessModel):
    """
    Multi-output sparse Gaussian process model with independent dimensions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        kernel = self.kernel
        if not self.config["share_kernel"]:
            # Replace kernel with multiple kernels, one for each output
            # dimension.
            self.kernel = KernelList(
                [copy.deepcopy(kernel) for _ in range(self.config["P"])]
            )
        else:
            # Use sample kernel for each output dimension.
            self.kernel = KernelList(self.config["P"] * [kernel])

    def get_default_nat_params(self):
        return {
            "np1": torch.tensor([0.0] * self.config["num_inducing"])
            .unsqueeze(0)
            .repeat(self.config["K"], 1),
            "np2": torch.tensor([-0.5] * self.config["num_inducing"])
            .unsqueeze(0)
            .repeat(self.config["K"], 1)
            .diag_embed(),
        }

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "P": None,
            "share_kernel": False,
        }


class MOSparseGaussianProcessClassification(MOSparseGaussianProcess):
    """
    Sparse Gaussian process multi-output classification model.
    """

    conjugate_family = None

    def forward(self, x, q, diag=False):
        """
        Returns the (approximate) predictive posterior distribution of a
        multi-output sparse Gaussian process classification model.
        :param x: The input locations to make predictions at.
        :param q: The approximate posterior distribution q(u | z).
        :param diag: Whether to compute marginal posterior predictive
        distributions.
        :return: ∫ p(y | f, x) p(f | u) q(u) df du.
        """
        qf = self.posterior(x, q, diag=diag)
        # (num_samples, P, N).
        fs = qf.sample((self.config["num_predictive_samples"],))

        # (N, num_samples, P).
        fs = fs.permute(2, 0, 1)

        comp = distributions.Categorical(logits=fs)
        mix = distributions.Categorical(
            torch.ones(
                fs.shape[1],
            )
        )

        return distributions.MixtureSameFamily(mix, comp)

    def likelihood_forward(self, x, theta, **kwargs):
        """
        Returns the model's likelihood p(y | x).
        :param x: The input locations to make predictions at, (*, D).
        :param theta: The latent variables of the model.
        :return: p(y | x).
        """
        return distributions.Categorical(logits=theta)

    def conjugate_update(self, data, q, t=None):
        """
        :param data: The local data to refine the model with.
        :param q: The current global posterior q(θ).
        :param t: The local factor t(θ).
        :return: q_new, t_new, the new global posterior and the new local
        contribution.
        """
        raise NotImplementedError

    def expected_log_likelihood(self, data, q, num_samples=1):
        x = data["x"]
        qf = self.posterior(x, q, diag=True)
        fs = qf.rsample((num_samples,)).permute(0, 2, 1)
        return self.likelihood_log_prob(data, fs).mean(0)
