import time
import numpy as np


class EarlyStopping:
    """
    Returns False if the score doesn't improve after a given patience.
    https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """

    def __init__(
        self, patience=5, score_name="elbo", verbose=False, delta=0, stash_model=False
    ):
        """
        :param patience: How long to wait after last time score improved.
        :param verbose: Whether to print message informing of early stopping.
        :param delta: Minimum change to qualify as an improvement.
        :param stash_model: Whether to update the best model with the score.
        """
        self.patience = patience
        self.score_name = score_name
        self.verbose = verbose
        self.delta = delta
        self.stash_model = stash_model
        self.best_model = None
        self.best_score = None

    def __call__(self, scores=None, model=None):
        """
        :param scores: A dict of scores.
        :param model: Current model producing latest score.
        :return: Whether to stop early.
        """
        if scores is None:
            self.best_score = None
            if self.stash_model:
                self.best_model = model

            return

        else:
            vals = scores[self.score_name]

            # Check whether best score has been beaten.
            new_val = vals[-1]
            if self.best_score is None or new_val > self.best_score:
                self.best_score = new_val
                if self.stash_model and model is not None:
                    self.best_model = model

            # Check whether to stop.
            if len(vals) > self.patience:
                prev_vals = np.array(vals[-self.patience :])
                ref_val = np.array(vals[-self.patience - 1]) + self.delta

                if np.all(prev_vals < ref_val):
                    return True
                else:
                    return False
            else:
                return False


class Timer:
    def __init__(self):
        self.tstart = None
        self.pcstart = None
        self.ptstart = None

        self.tpaused = None
        self.pcpaused = None
        self.ptpaused = None
        self.paused = False

    def start(self):
        """
        Starts an internal timer by recording the current time.
        """
        self.tstart = time.time()
        self.pcstart = time.perf_counter()
        self.ptstart = time.process_time()

    def pause(self):
        """
        Pauses the timer.
        """
        if self.tstart is None:
            raise ValueError("Timer not started.")
        if self.paused:
            raise ValueError("Timer is already paused.")

        self.tpaused = time.time()
        self.pcpaused = time.perf_counter()
        self.ptpaused = time.process_time()
        self.paused = True

    def resume(self):
        """
        Resumes the timer by adding the pause time to the start time.
        """
        if self.tstart is None:
            raise ValueError("Timer not started.")
        if not self.paused:
            raise ValueError("Timer is not paused.")

        pauset = time.time() - self.tpaused
        pausepc = time.perf_counter() - self.pcpaused
        pausept = time.process_time() - self.ptpaused

        self.tstart = self.tstart + pauset
        self.pcstart = self.pcstart + pausepc
        self.ptstart = self.ptstart + pausept
        self.paused = False

    def get(self):
        """
        Returns a dict showing the amount of time elapsed since the start time,
        minus any pauses.
        """
        if self.tstart is None:
            raise ValueError("Timer not started.")
        if self.paused:
            return {
                "time": self.tpaused - self.tstart,
                "perf_counter": self.pcpaused - self.pcstart,
                "process_time": self.ptpaused - self.ptstart,
            }
        else:
            return {
                "time": time.time() - self.tstart,
                "perf_counter": time.perf_counter() - self.pcstart,
                "process_time": time.process_time() - self.ptstart,
            }

    @property
    def started(self):
        return self.tstart is not None
