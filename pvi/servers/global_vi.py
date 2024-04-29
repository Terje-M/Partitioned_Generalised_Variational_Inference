import logging
import time
import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
from pvi.servers.base import Server
from pvi.utils.dataset import ListDataset
from pvi.utils.training_utils import EarlyStopping
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# =============================================================================
# Global VI class
# =============================================================================


class GlobalVIServer(Server):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Tracks number of epochs.
        self.epochs = 0

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "train_model": False,
            "model_optimiser_params": {"lr": 1e-2},
            "max_iterations": 1,
            "epochs": 1,
            "batch_size": 100,
            "optimiser": "Adam",
            "optimiser_params": {"lr": 0.05},
            "lr_scheduler": "MultiplicativeLR",
            "lr_scheduler_params": {"lr_lambda": lambda epoch: 1.0},
            "early_stopping": EarlyStopping(np.inf),
            "num_elbo_samples": 10,
            "print_epochs": np.pi,
            "homogenous_split": True,
        }

    def tick(self):
        """
        Trains the model using the entire dataset.
        """
        if self.should_stop():
            return False

        # Start timer.
        self.timer.start()

        p = self.p.non_trainable_copy()

        if self.iterations == 0 and self.init_q is not None:
            q = self.init_q.trainable_copy()
        else:
            q = self.q.trainable_copy()

        # Parameters are those of q(θ) and self.model.
        if self.config["train_model"]:
            parameters = [
                {"params": q.parameters()},
                {
                    "params": self.model.parameters(),
                    **self.config["model_optimiser_params"],
                },
            ]
        else:
            parameters = q.parameters()

        # TODO: currently assumes Gaussian distribution.
        # Parameters are those of q(θ) and self.model.
        # Try using different learning rate for σ than μ.
        # q_parameters = list(q.parameters())
        # if self.config["train_model"]:
        #     parameters = [
        #         {"params": q_parameters[0]},
        #         {"params": q_parameters[1],
        #          **self.config["sigma_optimiser_params"]},
        #         {"params": self.model.parameters(),
        #          **self.config["model_optimiser_params"]}
        #     ]
        # else:
        #     parameters = [
        #         {"params": q_parameters[0]},
        #         {"params": q_parameters[1],
        #          **self.config["sigma_optimiser_params"]},
        #     ]

        logging.info("Resetting optimiser")
        optimiser = getattr(torch.optim, self.config["optimiser"])(
            parameters, **self.config["optimiser_params"]
        )
        lr_scheduler = getattr(torch.optim.lr_scheduler, self.config["lr_scheduler"])(
            optimiser, **self.config["lr_scheduler_params"]
        )

        # Set up data: global VI server can access the entire dataset.
        if self.config["homogenous_split"]:
            x = self.data["x"]
            y = self.data["y"]

            # Shuffle data.
            tensor_dataset = TensorDataset(x, y)
            loader = DataLoader(
                tensor_dataset, batch_size=self.config["batch_size"], shuffle=True
            )
        else:
            # Inhomogenous split: order matters.
            m = self.config["batch_size"]
            data = defaultdict(list)
            for client in self.clients:
                # Chunk clients data into batch size.
                data = {
                    k: data[k]
                    + [v[i : i + m] for i in range(0, len(client.data["x"]), m)]
                    for k, v in client.data.items()
                }

            # Lists of tensors size (batch_size, *).
            x = data["x"]
            y = data["y"]
            list_dataset = ListDataset(x, y)
            loader = DataLoader(list_dataset, batch_size=None, shuffle=True)

        if self.config["device"] == "cuda":
            loader.pin_memory = True

        # Dict for logging optimisation progress.
        training_metrics = defaultdict(list)

        # Reset early stopping.
        self.config["early_stopping"](scores=None, model=q.non_trainable_copy())

        # Gradient-based optimisation loop -- loop over epochs
        epoch_iter = tqdm(range(self.config["epochs"]), desc="Epochs")
        # for i in range(self.config["epochs"]):
        for i in epoch_iter:
            epoch = defaultdict(lambda: 0.0)

            # Loop over batches in current epoch
            for (x_batch, y_batch) in iter(loader):
                x_batch = x_batch.to(self.config["device"])
                y_batch = y_batch.to(self.config["device"])
                self.communications += 1

                optimiser.zero_grad()

                batch = {
                    "x": x_batch,
                    "y": y_batch,
                }

                # Compute KL divergence between q and p.
                kl = q.kl_divergence(p).sum() / len(self.data["x"])

                # Compute E_q[log p(y | x, θ)].
                ll = self.model.expected_log_likelihood(
                    batch, q, self.config["num_elbo_samples"]
                ).sum()
                ll /= len(x_batch)

                # Negative local Free Energy is KL minus log-probability
                loss = kl - ll
                loss.backward()
                optimiser.step()

                # Keep track of quantities for current batch
                # Will be very slow if training on GPUs.
                epoch["elbo"] += -loss.item() / len(loader)
                epoch["kl"] += kl.item() / len(loader)
                epoch["ll"] += ll.item() / len(loader)

            epoch_iter.set_postfix(
                elbo=epoch["elbo"],
                kl=epoch["kl"],
                ll=epoch["ll"],
                lr=optimiser.param_groups[0]["lr"],
            )

            # Log progress for current epoch
            training_metrics["elbo"].append(epoch["elbo"])
            training_metrics["kl"].append(epoch["kl"])
            training_metrics["ll"].append(epoch["ll"])

            stop_early = self.config["early_stopping"](
                training_metrics, model=q.non_trainable_copy()
            )

            if (
                i % self.config["print_epochs"] == 0
                or i == (self.config["epochs"] - 1)
                or stop_early
            ):
                # Update global posterior before evaluating performance.
                self.q = q.non_trainable_copy()

                self.evaluate_performance(
                    {
                        "epochs": i,
                        "elbo": epoch["elbo"],
                        "kl": epoch["kl"],
                        "ll": epoch["ll"],
                    }
                )

                # Report performance.
                report = ""
                metrics = self.log["performance_metrics"][-1]
                report += f"epochs: {metrics['epochs']} "
                report += f"elbo: {metrics['elbo']:.3f} "
                report += f"ll: {metrics['ll']:.3f} "
                report += f"kl: {metrics['kl']:.3f} \n"
                for k, v in self.log["performance_metrics"][-1].items():
                    if "mll" in k or "acc" in k:
                        report += f"{k}: {v:.3f} "

                tqdm.write(report)

            # Log q so we can track performance each epoch.
            self.log["communications"].append(self.communications)
            self.epochs += 1

            # Update learning rate.
            lr_scheduler.step()

            # Check whether to stop early.
            if stop_early:
                break

        # Create non-trainable copy to send back to server.
        if self.config["early_stopping"].stash_model:
            self.q = self.config["early_stopping"].best_model
        else:
            self.q = q.non_trainable_copy()

        self.iterations += 1

        # Log training.
        self.log["training_curves"].append(training_metrics)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1
