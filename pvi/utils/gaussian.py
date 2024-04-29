import torch

from pvi.utils.psd_utils import psd_inverse


def mvstandard2natural(mu, cov):
    """
    Converts standard parameterisation of a multivariate Gaussian to natural
    parameterisation.
    :param mu: The mean, (*, D).
    :param cov: The covariance matrix, (*, D, D).
    :return: np1, np2, (*, D), (*, D, D).
    """
    assert mu.shape[-1] == cov.shape[-1], "mu (*, D), cov (*, D, D)."
    assert cov.shape[-2] == cov.shape[-1], "cov (*, D, D)."

    prec = psd_inverse(cov)
    np2 = -0.5 * prec
    np1 = prec.matmul(mu.unsqueeze(-1)).squeeze(-1)

    return np1, np2


def mvnatural2standard(np1, np2):
    """
    Converts natural parameterisation of a multivariate Gaussian to standard
    parameterisation.
    :param np1: Natural parameter 1, (*, D).
    :param np2: Natural parameter 2, (*, D, D).
    :return: mu, cov, (*, D), (*, D, D).
    """
    assert np1.shape[-1] == np2.shape[-1], "np1 (*, D), np2 (*, D, D)."
    assert np2.shape[-2] == np2.shape[-1], "np2 (*, D, D)."

    prec = -2 * np2
    cov = psd_inverse(prec)
    mu = cov.matmul(np1.unsqueeze(-1)).squeeze(-1)

    return mu, cov


def standard2natural(mu, sigma):
    """
    Converts standard parameterisation of a 1-D Gaussian to natural
    parameterisation.
    :param mu: The mean, (*, D).
    :param sigma: The standard deviation, (*, D).
    :return: np1, np2, (*, D), (*, D).
    """
    assert mu.shape[-1] == sigma.shape[-1], "mu (*, D), sigma (*, D)."
    assert len(mu.shape) == len(sigma.shape), "mu (*, D), sigma (*, D)."

    np2 = -0.5 * sigma ** (-2)
    np1 = sigma ** (-2) * mu

    return np1, np2


def natural2standard(np1, np2):
    """
    Converts natural parameterisation of a multivariate Gaussian to standard
    parameterisation.
    :param np1: Natural parameter 1, (*, D).
    :param np2: Natural parameter 2, (*, D).
    :return: mu, sigma, (*, D), (*, D).
    """
    assert np1.shape[-1] == np2.shape[-1], "np1 (*, D), np2 (*, D)."
    assert len(np1.shape) == len(np2.shape), "np1 (*, D), np2 (*, D)."

    sigma = (-2 * np2) ** (-0.5)
    mu = sigma ** 2 * np1

    return mu, sigma


def joint_from_marginal(qa, kab, kbb, kaa=None, ikaa=None, b_then_a=False):
    """
    Computes the joint distribution q(a, b) = p(b | a) q(a).
    :param qa: The marginal distribution, q(a).
    :param kab: K(za, zb).
    :param kbb: K(zb, zb).
    :param kaa: K(za, za).
    :param ikaa: K(za, za)^{-1}.
    :param b_then_a: Order as q(b, a), rather than q(a, b).
    :return: q(a, b) = p(b | a) q(a).
    """
    dima = kab.shape[-2]
    dimb = kab.shape[-1]
    dimab = dima + dimb
    qa_loc = qa.std_params["loc"]
    qa_cov = qa.std_params["covariance_matrix"]

    if ikaa is None:
        if kaa is None:
            raise ValueError("Must specifiy either kaa or ikaa.")
        else:
            ikaa = psd_inverse(kaa)

    a = kab.transpose(-1, -2).matmul(ikaa)
    at = a.transpose(-1, -2)
    qab_loc = torch.zeros(*kbb.shape[:-2], dimab)
    qab_cov = torch.zeros(*kbb.shape[:-2], dimab, dimab)

    if b_then_a:
        qab_loc[..., dimb:] = qa_loc
        qab_loc[..., :dimb] = a.matmul(qa_loc.unsqueeze(-1)).squeeze(-1)
        qab_cov[..., dimb:, dimb:] = qa_cov
        qab_cov[..., dimb:, :dimb] = qa_cov.matmul(at)
        qab_cov[..., :dimb, dimb:] = a.matmul(qa_cov)
        qab_cov[..., :dimb, :dimb] = (
                kbb + a.matmul(qa_cov).matmul(at) - a.matmul(kab))
    else:
        qab_loc[..., :dima] = qa_loc
        qab_loc[..., dima:] = a.matmul(qa_loc.unsqueeze(-1)).squeeze(-1)
        qab_cov[..., :dima, :dima] = qa_cov
        qab_cov[..., :dima, dima:] = qa_cov.matmul(at)
        qab_cov[..., dima:, :dima] = a.matmul(qa_cov)
        qab_cov[..., dima:, dima:] = (
                kbb + a.matmul(qa_cov).matmul(at) - a.matmul(kab))

    assert torch.isclose(qab_cov, qab_cov.transpose(-1, -2)).all()

    return qab_loc, qab_cov


def nat_from_std(std_params):
    loc = std_params["loc"]
    cov = std_params["covariance_matrix"]
    # prec = psd_inverse(cov)
    prec = cov.inverse()

    nat = {
        "np1": prec.matmul(loc.unsqueeze(-1)).squeeze(-1),
        "np2": -0.5 * prec
    }

    return nat


def std_from_nat(nat_params):
    np1 = nat_params["np1"]
    np2 = nat_params["np2"]

    prec = -2. * np2
    # cov = psd_inverse(prec)
    cov = prec.inverse()

    std = {
        "loc": cov.matmul(np1.unsqueeze(-1)).squeeze(-1),
        "covariance_matrix": cov
    }

    return std


def slow_kl(std_params1, std_params2):
    """
    Compute the KL-divergence KL(q1 || q2) whilst avoiding Cholesky
    decompositions to allow for invalid distributions.
    :param std_params1: Standard parameters of q1.
    :param std_params2: Standard parameters of q2.
    :return: KL(q1 || q2).
    """
    mu1, cov1 = std_params1["loc"], std_params1["covariance_matrix"]
    mu2, cov2 = std_params2["loc"], std_params2["covariance_matrix"]
    prec2 = cov2.inverse()

    assert mu1.shape == mu2.shape and cov1.shape == cov2.shape

    kl = 0.5 * (cov2.logdet() - cov1.logdet() - len(mu1)
                + prec2.matmul(cov1).trace()
                + (mu2 - mu1).dot(prec2.matmul(mu2 - mu1)))

    return kl
