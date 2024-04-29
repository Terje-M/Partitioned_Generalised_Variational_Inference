import logging
import time

from tqdm.auto import tqdm
from pvi.servers.base import Server, ServerBayesianHypers

logger = logging.getLogger(__name__)


class SynchronousServer(Server):
    def get_default_config(self):
        return {
            **super().get_default_config(),
            "max_iterations": 25,
            "shared_factor_iterations": 0,
            "init_q_always": False,
        }

    def _tick(self):
        logger.debug("Getting client updates.")
        t_olds = []
        t_news = []
        for i, client in tqdm(enumerate(self.clients), leave=False):
            if client.can_update():
                logger.debug(f"On client {i + 1} of {len(self.clients)}.")
                t_old = client.t

                if self.iterations == 0 or self.config["init_q_always"]:
                    # First iteration. Pass q_init(θ) to client.
                    _, t_new = client.fit(self.q, self.init_q)
                else:
                    _, t_new = client.fit(self.q)

                t_olds.append(t_old)
                t_news.append(t_new)

        # Single communication per iteration.
        self.communications += 1

        logger.debug("Received client updates. Updating global posterior.")

        # Update global posterior.
        for t_old, t_new in zip(t_olds, t_news):
            self.q = self.q.replace_factor(t_old, t_new, is_trainable=False)

        if self.iterations < self.config["shared_factor_iterations"]:
            # Update shared factor.
            n = len(self.clients)
            t_np = {k: v * (1 / n) for k, v in t_news[0].nat_params.items()}
            for t_new in t_news[1:]:
                for k, v in t_new.nat_params.items():
                    t_np[k] += v * (1 / n)

            # Set clients factor to shared factor.
            for client in self.clients:
                client.t.nat_params = t_np

        logger.debug(f"Iteration {self.iterations} complete.\n")

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1


class SynchronousServerBayesianHypers(ServerBayesianHypers):
    def get_default_config(self):
        return {
            **super().get_default_config(),
            "max_iterations": 5,
        }

    def _tick(self):
        logger.debug("Getting client updates.")

        clients_updated = 0

        t_olds, t_news = [], []
        teps_olds, teps_news = [], []
        for i, client in tqdm(enumerate(self.clients), leave=False):
            if client.can_update():
                logger.debug(f"On client {i + 1} of {len(self.clients)}.")
                t_old = client.t
                teps_old = client.teps

                if self.iterations == 0:
                    _, _, t_new, teps_new = client.fit(
                        self.q, self.qeps, self.init_q, self.init_qeps
                    )
                else:
                    _, _, t_new, teps_new = client.fit(self.q, self.qeps)

                t_olds.append(t_old)
                t_news.append(t_new)
                teps_olds.append(teps_old)
                teps_news.append(teps_new)

                clients_updated += 1
                self.communications += 1

        logger.debug("Received client updates. Updating global posterior.")

        # Update global posterior.
        for t_old, t_new, teps_old, teps_new in zip(
            t_olds, t_news, teps_olds, teps_news
        ):
            self.q = self.q.replace_factor(t_old, t_new, is_trainable=False)
            self.qeps = self.qeps.replace_factor(teps_old, teps_new, is_trainable=False)

        logger.debug(
            f"Iteration {self.iterations} complete."
            f"\nNew natural parameters:\n{self.q.nat_params}\n."
        )

        # Log progress.
        # self.log["q"].append(self.q.non_trainable_copy())
        # self.log["qeps"].append(self.qeps.non_trainable_copy())
        self.log["clients_updated"].append(clients_updated)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1
