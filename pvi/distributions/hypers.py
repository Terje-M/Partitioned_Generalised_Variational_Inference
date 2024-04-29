class HyperparameterDistribution:
    """
    Maintains the distributions over hyperparameters.
    """

    def __init__(self, distributions=None):
        """
        :param distributions: A dictionary of (hyperparameter, distribution)
        pairs.
        """
        self.distributions = distributions

    @property
    def nat_params(self):
        return {k: v.nat_params for k, v in self.distributions.items()}

    def non_trainable_copy(self):
        return type(self)(
            distributions={
                k: v.non_trainable_copy() for k, v in self.distributions.items()
            }
        )

    def trainable_copy(self):
        return type(self)(
            distributions={k: v.trainable_copy() for k, v in self.distributions.items()}
        )

    def replace_factor(self, t_old=None, t_new=None, **kwargs):
        """
        Forms a new distribution by replacing the natural parameters of
        t_old(ε) with t_new(ε).
        :param t_old: The factor to remove.
        :param t_new: The factor to add.
        :param kwargs: Passed to self.create_new()
        :return: Updated distribution.
        """
        if t_old is not None and t_new is not None:
            new_distributions = {
                k: self.distributions[k].replace_factor(
                    t_old.factors[k], t_new.factors[k], **kwargs
                )
                for k in self.distributions.keys()
            }
        elif t_old is None and t_new is not None:
            new_distributions = {
                k: self.distributions[k].replace_factor(
                    None, t_new.factors[k], **kwargs
                )
                for k in self.distributions.keys()
            }
        elif t_old is not None and t_new is None:
            new_distributions = {
                k: self.distributions[k].replace_factor(
                    t_old.factors[k], None, **kwargs
                )
                for k in self.distributions.keys()
            }
        else:
            raise ValueError("Both t_old and t_new are None")

        return self.create_new(distributions=new_distributions, **kwargs)

    def kl_divergence(self, other, **kwargs):
        return {
            k: v.kl_divergence(other.distributions[k], **kwargs)
            for k, v in self.distributions.items()
        }

    def log_prob(self, args_dict, kwargs_dict):
        return {
            k: v.log_prob(**args_dict[k], **kwargs_dict[k])
            for k, v in self.distributions.items()
        }

    def sample(self, *args, **kwargs):
        return {k: v.sample(*args, **kwargs) for k, v in self.distributions.items()}

    def rsample(self, *args, **kwargs):
        return {k: v.rsample(*args, **kwargs) for k, v in self.distributions.items()}

    def parameters(self):
        parameters = [list(v.parameters()) for v in self.distributions.values()]
        return [item for sublist in parameters for item in sublist]

    @classmethod
    def create_new(cls, **kwargs):
        return cls(**kwargs)


class HyperparameterFactor:
    """
    Maintains the factors over hyperparameters.
    """

    def __init__(self, factors=None):
        """
        :param factors: A dictionary of (hyperparameter, factor)
        pairs.
        """
        self.factors = factors

    def compute_refined_factor(self, q1, q2, **kwargs):
        return type(self)(
            factors={
                k: v.compute_refined_factor(
                    q1.distributions[k],
                    q2.distributions[k],
                    **kwargs
                )
                for k, v in self.factors.items()
            }
        )

    def __call__(self, thetas):
        return {k: v(thetas[k]) for k, v in self.factors.items()}

    @property
    def nat_params(self):
        return {k: v.nat_params for k, v in self.factors.items()}

    def log_h(self, thetas):
        return {k: v.log_h(thetas[k]) for k, v in self.factors.items()}

    def npf(self, thetas):
        return {k: v.npf(thetas[k]) for k, v in self.factors.items()}

    def eqlogt(self, q, num_samples=1):
        return {
            k: v.eqlogt(q.distributions[k], num_samples)
            for k, v in self.factors.items()
        }

    def nat_from_dist(self, q):
        return {k: v.nat_from_dist(q.distributions[k]) for k, v in self.factors.items()}

    def dist_from_nat(self, np):
        return {k: v.dist_from_nat(np[k]) for k, v in self.factors.items()}
