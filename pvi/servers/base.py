import logging
import time
import torch

from abc import ABC, abstractmethod
from collections import defaultdict
from pvi.utils.training_utils import Timer

logger = logging.getLogger(__name__)


class Server(ABC):
    """
    An abstract class for the server.
    """

    def __init__(
        self, model, p, clients, config=None, init_q=None, data=None, val_data=None
    ):

        if config is None:
            config = {}

        self._config = self.get_default_config()
        self.config = config

        # Shared probabilistic model.
        self.model = model

        # Global prior p(θ).
        self.p = p

        # Global posterior q(θ).
        if hasattr(p, "non_trainable_copy"):
            self.q = p.non_trainable_copy()
        else:
            self.q = p

        # Initial q(θ) for first client update.
        self.init_q = init_q

        # Clients.
        self.clients = clients

        # Union of clients data
        if data is None:
            # Third line down previously:for k in self.clients[0].data.keys()
            self.data = {
                k: torch.cat([client.data[k] for client in self.clients], dim=0)
                for k in self.clients[0].data.keys()
            }
        else:
            self.data = data

        # Validation dataset.
        self.val_data = val_data

        # Internal iteration counter.
        self.iterations = 0

        # Internal communication counter.
        self.communications = 0

        self.log = defaultdict(list)

        self.log["communications"].append(self.communications)

        # Evaluate performance of prior.
        # if self.q is not None:
        #    self.evaluate_performance()

        # Will initialise these in self.tick().
        self.timer = Timer()

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = {**self._config, **config}

    def get_default_config(self):
        return {
            "train_model": False,
            "model_update_freq": 1,
            "hyper_optimiser": "SGD",
            "hyper_optimiser_params": {"lr": 1},
            "hyper_updates": 1,
            "performance_metrics": lambda client, data: {},
            "track_q": False,
            "device": "cpu",
        }

    def tick(self):
        """ """
        if self.should_stop():
            return False

        if not self.timer.started:
            self.timer.start()

        self._tick()

        self.iterations += 1

        # Update hyperparameters.
        if (
            self.config["train_model"]
            and self.iterations % self.config["model_update_freq"] == 0
        ):
            self.update_hyperparameters()

        # Evaluate performance after every iterations.
        self.evaluate_performance()

    def _tick(self):
        """
        Defines what the server should do on each update round. Could be a
        synchronous update, asynchronous update etc.
        """
        raise NotImplementedError

    @abstractmethod
    def should_stop(self):
        """
        Defines when the server should stop running.
        """
        pass

    def evaluate_performance(self, default_metrics=None):
        metrics = {
            **self.timer.get(),
            "communications": self.communications,
            "iterations": self.iterations,
        }

        # Pause timer whilst getting performance metrics.
        self.timer.pause()

        if default_metrics is not None:
            metrics = {**default_metrics, **metrics}

        if self.config["performance_metrics"] is not None:
            train_metrics = self.config["performance_metrics"](self, self.data)
            for k, v in train_metrics.items():
                metrics["train_" + k] = v

            if self.val_data is not None:
                val_metrics = self.config["performance_metrics"](self, self.val_data)
                for k, v in val_metrics.items():
                    metrics["val_" + k] = v

        if self.config["track_q"]:
            # Store current q(θ) natural parameters.
            metrics["npq"] = {k: v.detach().cpu() for k, v in self.q.nat_params.items()}

        self.log["performance_metrics"].append(metrics)

        # Resume timer.
        self.timer.resume()

    def update_hyperparameters(self):
        """
        Updates the model hyperparameters according to
        dF / dε = (μ_q - μ_0)^T dη_0 / dε + Σ_m dF_m / dε.
        """
        # TODO: currently performs single optimisation step. Perform more?

        # Zero gradients as accumulated during client optimisation.
        for param in self.model.parameters():
            if param.grad is not None:
                if param.grad.grad_fn is not None:
                    param.grad.detach_()
                else:
                    param.grad.requires_grad_(False)

                param.grad.zero_()

        # Ensure clients have same model as server and get E[log p(y | x, θ)].
        vfe = 0
        for client in self.clients:
            vfe += self.model.expected_log_likelihood(
                client.data, self.q, client.config["num_elbo_samples"]
            ).sum()

        if hasattr(self.model, "prior"):
            # Compute prior using model's current hyperparameters.
            p_old = self.model.prior(q=self.q)
            mq = {k: v.detach() for k, v in self.q.mean_params.items()}
            mp = {k: v.detach() for k, v in p_old.mean_params.items()}

            vfe += sum(
                [
                    (mq[k] - mp[k]).flatten().dot(v.flatten())
                    for k, v in zip(mq.keys(), p_old.nat_params.values())
                ]
            )
        else:
            p_old = None

        # Make VFE per data-points. Improves stability.
        vfe /= sum([len(client.data["x"]) for client in self.clients])

        # Compute gradients.
        vfe.backward()

        # Update model parameters, and pass to clients.
        for param in self.model.parameters():
            param.data += self.config["hyper_optimiser_params"]["lr"] * param.grad

        for client in self.clients:
            client.model = self.model

        # Update prior term in maintained q.
        if p_old is not None:
            # Compute prior using model's current hyperparameters.
            p_new = self.model.prior(q=self.q)
            q_new_nps = {
                k: (v - p_old.nat_params[k].detach() + p_new.nat_params[k].detach())
                for k, v in self.q.nat_params.items()
            }

            self.q = self.q.create_new(nat_params=q_new_nps, is_trainable=False)

        parameters = {k: v.data for k, v in self.model.named_parameters()}
        logger.debug(
            f"Updated model hyperparameters."
            f"\nNew model hyperparameters:\n{parameters}\n."
        )
        print(
            f"Updated model hyperparameters."
            f"\nNew model hyperparameters:\n{parameters}\n."
        )

        return

    def model_predict(self, x, **kwargs):
        """
        Returns the current models predictive posterior distribution.
        :return: ∫ p(y | θ, x) q(θ) dθ.
        """
        return self.model(x, self.q, **kwargs)

    def add_client(self, client):
        self.clients.append(client)

    def get_compiled_log(self):
        """
        Get full log, including logs from each client.
        :return: full log.
        """
        final_log = {"server": self.log}

        client_logs = [client.log for client in self.clients]
        for i, log in enumerate(client_logs):
            final_log["client_" + str(i)] = log

        return final_log


class ServerBayesianHypers(Server):
    def __init__(
        self, model, p, peps, clients, config=None, init_q=None, init_qeps=None
    ):
        super().__init__(model, p, clients, config, init_q)

        # Global prior p(ε).
        self.peps = peps

        # Global posterior q(ε).
        self.qeps = peps.non_trainable_copy()

        # Initial q(ε) for first client.
        self.init_qeps = init_qeps

        self.log["qeps"].append(self.qeps.non_trainable_copy())

    @classmethod
    def get_default_config(cls):
        return {
            "num_eps_samples": 1,
        }

    def tick(self):
        """
        Wrapper for _tick method.
        """
        if self.should_stop():
            return False

        if not self.timer.started:
            self.timer.start()

        self._tick()

        self.iterations += 1

        # Evaluate performance after every iteration.
        #self.evaluate_performance()

    @abstractmethod
    def _tick(self):
        """
        Defines what the server should do on each update round. Could be a
        synchronous update, asynchronous update etc.
        """
        pass

    @abstractmethod
    def should_stop(self):
        """
        Defines when the server should stop running.
        """
        pass

    def model_predict(self, x, **kwargs):
        """
        Returns the current models predictive posterior distribution.
        :return: ∫ p(y | x, θ, ε) q(θ)q(ε) dθ dε.
        """
        neps = self.config["num_eps_samples"]
        dists = []
        for _ in range(neps):
            eps = self.qeps.sample()
            self.model.hyperparameters = eps
            dists.append(self.model(x, self.q, **kwargs))

        return dists
