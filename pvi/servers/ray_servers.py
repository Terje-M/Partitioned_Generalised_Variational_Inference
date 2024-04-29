import logging
import time
import numpy as np
import ray

from tqdm.auto import tqdm
from pvi.servers.base import Server
from pvi.clients.base import Client

logger = logging.getLogger(__name__)


class SynchronousRayServer(Server):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.server_worker = ServerWorker.options(**self.config["ray_options"]).remote(
            self.q
        )

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "ray_options": {},
        }

    def tick(self):
        q = self.server_worker.get_current_q.remote()

        while not self.should_stop():
            # Pass current q to clients.
            working_clients = [
                client.update_client.remote(q, self.init_q) for client in self.clients
            ]

            # Apply change in factors.
            q = self.server_worker.update_q.remote(*working_clients)
            self.communications += len(self.clients)
            self.iterations += 1

            # Evaluate current posterior.
            self.q = ray.get(q)
            self.evaluate_performance()
            self.log["communications"].append(self.communications)

            metrics = self.log["performance_metrics"][-1]
            print("Communications: {}.".format(self.communications))
            print(
                "Test mll: {:.3f}. Test acc: {:.3f}.".format(
                    metrics["val_mll"], metrics["val_acc"]
                )
            )
            print(
                "Train mll: {:.3f}. Train acc: {:.3f}.\n".format(
                    metrics["train_mll"], metrics["train_acc"]
                )
            )

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1


class AsynchronousRayServer(Server):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.server_worker = ServerWorker.options(**self.config["ray_options"]).remote(
            self.q
        )

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "ray_options": {},
        }

    def tick(self):
        working_clients = {}
        q = self.server_worker.get_current_q.remote()
        for client in self.clients:
            working_clients[client.update_client.remote(q, self.init_q)] = client

        while not self.should_stop():
            ready_clients, _ = ray.wait(list(working_clients))
            ready_client_id = ready_clients[0]
            client = working_clients.pop(ready_client_id)

            # Apply change in factors.
            q = self.server_worker.update_q.remote(*[ready_client_id])

            # Get client training again.
            working_clients[client.update_client.remote(q)] = client

            self.communications += 1
            if self.communications % len(self.clients) == 0:
                # Evaluate current posterior.
                self.q = ray.get(q)
                self.evaluate_performance()
                self.log["communications"].append(self.communications)

                metrics = self.log["performance_metrics"][-1]
                print("Communications: {}.".format(self.communications))
                print(
                    "Test mll: {:.3f}. Test acc: {:.3f}.".format(
                        metrics["val_mll"], metrics["val_acc"]
                    )
                )
                print(
                    "Train mll: {:.3f}. Train acc: {:.3f}.\n".format(
                        metrics["train_mll"], metrics["train_acc"]
                    )
                )

    def should_stop(self):
        return self.communications > self.config["max_communications"] - 1


@ray.remote
class ServerWorker:
    def __init__(self, q):
        self.q = q

    def update_q(self, *ts):
        for t in ts:
            t_old, t_new = t

            # Update global posterior.
            self.q = self.q.replace_factor(t_old, t_new, is_trainable=False)

        return self.q

    def get_current_q(self):
        return self.q


@ray.remote
class RayClient(Client):
    def get_log(self):
        return self.log

    def update_client(self, q, init_q=None):
        print(f"Updating client {str(id(self))[-3:]}.")

        t_old = self.t
        _, t_new = self.fit(q, init_q)

        return t_old, t_new


class AsynchronousRayFactory(Server):
    """
    This acts as both the server and clients to enable scalable distributed
    learning
    """

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "init_q_always": False,
            "ray_options": {},
        }

    def tick(self):

        if not self.timer.started:
            self.timer.start()

        working_clients = []
        for i, client in enumerate(self.clients):
            working_clients.append(
                update_client.options(**self.config["ray_options"]).remote(
                    client, self.q, self.init_q, i
                )
            )

        # Stores ray object refs returned by performance_metrics.
        performance_metrics = []
        while not self.should_stop():
            ready_clients, _ = ray.wait(list(working_clients))
            client_id = ready_clients[0]
            client_idx = working_clients.index(client_id)

            # Apply change in factors.
            self.clients[client_idx], _, t_old, t_new = ray.get(client_id)
            self.q = update_q.options(**self.config["ray_options"]).remote(
                self.q, *[client_id]
            )

            # Get client training again.
            if self.config["init_q_always"]:
                working_clients[client_idx] = update_client.options(
                    **self.config["ray_options"]
                ).remote(
                    self.clients[client_idx], self.q, self.init_q, client_idx=client_idx
                )
            else:
                working_clients[client_idx] = update_client.options(
                    **self.config["ray_options"]
                ).remote(self.clients[client_idx], self.q, client_idx=client_idx)

            self.communications += 1

            # Log time and which client was updated.
            updated_client_times = {**self.timer.get()}
            updated_client_times[client_idx] = self.clients[client_idx].log[
                "update_time"
            ][-1]
            self.log["updated_client_times"].append(updated_client_times)

            if (self.communications % len(self.clients) == 0) or (
                self.communications < len(self.clients)
            ):
                # Evaluate current posterior.
                self.q = ray.get(self.q)
                performance_metrics.append(
                    evaluate_performance.options(**self.config["ray_options"]).remote(
                        self
                    )
                )

        self.log["performance_metrics"] = ray.get(performance_metrics)

    def should_stop(self):
        com_test = self.communications > self.config["max_communications"] - 1

        if len(self.log["performance_metrics"]) > 0:
            perf_test = self.log["performance_metrics"][-1]["val_mll"] < -10
        else:
            perf_test = False

        return com_test or perf_test


class SynchronousRayFactory(Server):
    """
    This acts as both the server and clients to enable scalable distributed
    learning
    """

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "init_q_always": False,
            "ray_options": {},
        }

    def tick(self):

        if not self.timer.started:
            self.timer.start()

        # Stores ray object refs returned by performance_metrics.
        performance_metrics = []
        while not self.should_stop():
            # Pass current q to clients.
            if self.iterations == 0 or self.config["init_q_always"]:
                working_clients = [
                    update_client.options(**self.config["ray_options"]).remote(
                        client, self.q, self.init_q, i
                    )
                    for i, client in enumerate(self.clients)
                ]
            else:
                working_clients = [
                    update_client.options(**self.config["ray_options"]).remote(
                        client, self.q, client_idx=i
                    )
                    for i, client in enumerate(self.clients)
                ]

            # Apply change in factors.
            self.q = update_q.options(**self.config["ray_options"]).remote(
                self.q, *working_clients
            )

            self.communications += 1
            self.iterations += 1

            # Update stored clients.
            for i, working_client in enumerate(working_clients):
                self.clients[i] = ray.get(working_client)[0]

            # Log time taken for each client to update.
            updated_client_times = {**self.timer.get()}
            for client_idx, client in enumerate(self.clients):
                updated_client_times[client_idx] = client.log["update_time"][-1]

            self.log["updated_client_times"].append(updated_client_times)

            # Evaluate current posterior.
            self.q = ray.get(self.q)
            performance_metrics.append(
                evaluate_performance.options(**self.config["ray_options"]).remote(self)
            )

        self.log["performance_metrics"] = ray.get(performance_metrics)

    def should_stop(self):
        iter_test = self.iterations > self.config["max_iterations"] - 1

        if len(self.log["performance_metrics"]) > 0:
            perf_test = self.log["performance_metrics"][-1]["val_mll"] < -10
        else:
            perf_test = False

        return iter_test or perf_test


class BCMSameRayFactory(Server):
    def get_default_config(self):
        return {
            **super().get_default_config(),
            "max_iterations": 1,
        }

    def tick(self):

        if not self.timer.started:
            self.timer.start()

        working_clients = []
        for i, client in enumerate(self.clients):
            working_clients.append(
                update_client.options(**self.config["ray_options"]).remote(
                    client, self.q, self.init_q, i
                )
            )

        performance_metrics = []
        while not self.should_stop():
            # Pass current q to clients.
            if self.iterations == 0:
                working_clients = [
                    update_client.options(**self.config["ray_options"]).remote(
                        client, self.q, self.init_q, i
                    )
                    for i, client in enumerate(self.clients)
                ]
            else:
                working_clients = [
                    update_client.options(**self.config["ray_options"]).remote(
                        client, self.q, client_idx=i
                    )
                    for i, client in enumerate(self.clients)
                ]

            # Get natural parameters of each clients posterior.
            nps = []
            for i, working_client in enumerate(working_clients):
                self.clients[i], q_i, _, _ = ray.get(working_client)
                nps.append({k: v.detach().clone() for k, v in q_i.nat_params.items()})

            # Update global posterior.
            q_nps = {
                k: sum([x[k] for x in nps]) - (len(self.clients) - 1) * v
                for k, v in self.q.nat_params.items()
            }

            self.q = self.q.create_new(nat_params=q_nps, is_trainable=False)

            self.communications += 1
            self.iterations += 1

            # Evaluate current posterior.
            performance_metrics.append(
                evaluate_performance.options(**self.config["ray_options"]).remote(self)
            )

        self.log["performance_metrics"] = ray.get(performance_metrics)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1


class BCMSplitRayFactory(Server):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        nk = [client.data["x"].shape[0] for client in self.clients]
        client_props = [n / sum(nk) for n in nk]
        self.client_props = client_props

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "max_iterations": 1,
        }

    def tick(self):

        if not self.timer.started:
            self.timer.start()

        working_clients = []
        for i, client in enumerate(self.clients):
            working_clients.append(
                update_client.options(**self.config["ray_options"]).remote(
                    client, self.q, self.init_q, i
                )
            )

        performance_metrics = []
        while not self.should_stop():
            # Pass current q to clients.
            working_clients = []
            for i, client in enumerate(self.clients):
                p_i_nps = {
                    k: v * self.client_props[i] for k, v in self.q.nat_params.items()
                }
                p_i = self.q.create_new(nat_params=p_i_nps, is_trainable=False)

                if self.iterations == 0:
                    working_clients.append(
                        update_client.options(**self.config["ray_options"]).remote(
                            client, p_i, self.init_q, i
                        )
                    )
                else:
                    working_clients.append(
                        update_client.options(**self.config["ray_options"]).remote(
                            client, p_i, client_indx=i
                        )
                    )

            # Get natural parameters of each clients posterior.
            nps = []
            for i, working_client in enumerate(working_clients):
                self.clients[i], q_i, _, _ = ray.get(working_client)
                nps.append({k: v.detach().clone() for k, v in q_i.nat_params.items()})

            # Update global posterior.
            q_nps = {k: sum([x[k] for x in nps]) for k, v in self.q.nat_params.items()}

            self.q = self.q.create_new(nat_params=q_nps, is_trainable=False)

            self.communications += 1
            self.iterations += 1

            # Evaluate current posterior.
            performance_metrics.append(
                evaluate_performance.options(**self.config["ray_options"]).remote(self)
            )

        self.log["performance_metrics"] = ray.get(performance_metrics)

    def should_stop(self):
        return self.iterations > self.config["max_iterations"] - 1


class SequentialRayFactory(Server):
    """
    This acts as both the server and clients.
    """

    def get_default_config(self):
        return {
            **super.get_default_config(),
            "init_q_to_all": False,
            "init_q_always": False,
            "ray_options": {},
        }

    def tick(self):

        if not self.timer.started:
            self.timer.start()

        # Stores ray object refs returned by performance metrics.
        performance_metrics = []
        client_idx = 0
        while not self.should_stop():
            client_idx = client_idx % len(self.clients)
            if (
                (not self.config["init_q_to_all"] and self.communications == 0)
                or (
                    self.config["init_q_to_all"]
                    and self.communications < len(self.clients)
                )
                or self.config["init_q_always"]
            ):
                working_client = update_client.options(
                    **self.config["ray_options"]
                ).remote(self.clients[client_idx], self.q, self.init_q, client_idx)
            else:
                working_client = update_client.options(
                    **self.config["ray_options"]
                ).remote(self.clients[client_idx], self.q, client_idx=client_idx)

            # Wait for current client to finish and apply change in factors.
            self.clients[client_idx], _, t_old, t_new = ray.get(working_client)
            self.q = update_q.options(**self.config["ray_options"]).remote(
                self.q, *[working_client]
            )

            self.communications += 1

            # Log time and which client was updated.
            updated_client_times = {**self.timer.get()}
            updated_client_times[client_idx] = self.clients[client_idx].log[
                "update_time"
            ][-1]
            self.log["updated_client_times"].append(updated_client_times)

            if (self.communications % len(self.clients) == 0) or (
                self.communications < len(self.clients)
            ):
                # Evaluate current posterior.
                self.q = ray.get(self.q)
                performance_metrics.append(
                    evaluate_performance.options(**self.config["ray_options"]).remote(
                        self
                    )
                )

        self.log["performance_metrics"] = ray.get(performance_metrics)


@ray.remote
def update_client(client, q, init_q=None, client_idx=None):
    if client_idx is not None:
        print(f"Updating client {client_idx}.")

    t_old = client.t
    q_new, t_new = client.fit(q, init_q)

    # Add IP address of whichever node it was ran on.
    ip_address = ray._private.services.get_node_ip_address()
    client.log["update_time"][-1]["ip_address"] = ip_address

    return client, q_new, t_old, t_new


@ray.remote
def update_q(q, *ts):
    for t in ts:
        _, _, t_old, t_new = t

        # Update posterior.
        q = q.replace_factor(t_old, t_new, is_trainable=False)

    return q


@ray.remote
def evaluate_performance(server):
    server.evaluate_performance()

    metrics = server.log["performance_metrics"][-1]
    print("Communications: {}.".format(server.communications))
    print(
        "Test mll: {:.3f}. Test acc: {:.3f}.".format(
            metrics["val_mll"], metrics["val_acc"]
        )
    )
    print(
        "Train mll: {:.3f}. Train acc: {:.3f}.\n".format(
            metrics["train_mll"], metrics["train_acc"]
        )
    )

    # Add IP address of whichever node it was ran on.
    metrics["ip_address"] = ray._private.services.get_node_ip_address()

    return metrics
