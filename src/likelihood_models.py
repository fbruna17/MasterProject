"""
Likelihood Models
-----------------
The likelihood models contain all the logic needed to train and use Darts' neural network models in
a probabilistic way. This essentially means computing an appropriate training loss and sample from the
distribution, given the parameters of the distribution.
By default, all versions will be trained using their negative log likelihood as a loss function
(hence performing maximum likelihood estimation when training the model).
However, most likelihoods also optionally support specifying time-independent "prior"
beliefs about the distribution parameters.
In such cases, the a KL-divergence term is added to the loss in order to regularise it in the
direction of the specified prior distribution. (Note that this is technically not purely
a Bayesian approach as the priors are actual parameters values, and not distributions).
The parameter `prior_strength` controls the strength of the "prior" regularisation on the loss.
Some distributions (such as ``GaussianLikelihood``, and ``PoissonLikelihood``) are univariate,
in which case they are applied to model each component of multivariate series independently.
Some other distributions (such as ``DirichletLikelihood``) are multivariate,
in which case they will model all components of multivariate time series jointly.
Univariate likelihoods accept either scalar or array-like values for the optional prior parameters.
If a scalar is provided, it is used as a prior for all components of the series. If an array-like is provided,
the i-th value will be used as a prior for the i-th component of the series. Multivariate likelihoods
require array-like objects when specifying priors.
The target series used for training must always lie within the distribution's support, otherwise
errors will be raised during training. You can refer to the individual likelihoods' documentation
to see what is the support. Similarly, the prior parameters also have to lie in some pre-defined domains.
"""

import collections
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta as _Beta
from torch.distributions import Exponential as _Exponential
from torch.distributions import Gamma as _Gamma
from torch.distributions import Gumbel as _Gumbel
from torch.distributions import HalfNormal as _HalfNormal
from torch.distributions import LogNormal as _LogNormal
from torch.distributions import Normal as _Normal
from torch.distributions import Weibull as _Weibull
from torch.distributions.kl import kl_divergence

# TODO: Table on README listing distribution, possible priors and wiki article
from darts.utils.utils import raise_if_not

MIN_CAUCHY_GAMMA_SAMPLING = 1e-100


# Some utils for checking parameters' domains
def _check(param, predicate, param_name, condition_str):
    if param is None:
        return
    if isinstance(param, (collections.Sequence, np.ndarray)):
        raise_if_not(
            all(predicate(p) for p in param),
            f"All provided parameters {param_name} must be {condition_str}.",
        )
    else:
        raise_if_not(
            predicate(param),
            f"The parameter {param_name} must be {condition_str}.",
        )


def _check_strict_positive(param, param_name=""):
    _check(param, lambda p: p > 0, param_name, "strictly positive")


def _check_in_open_0_1_intvl(param, param_name=""):
    _check(param, lambda p: 0 < p < 1, param_name, "in the open interval (0, 1)")


class Likelihood(ABC):
    def __init__(self, prior_strength=1.0):
        """
        Abstract class for a likelihood model.
        """
        self.prior_strength = prior_strength

    def compute_loss(self, model_output: torch.Tensor, target: torch.Tensor):
        """
        Computes a loss from a `model_output`, which represents the parameters of a given probability
        distribution for every ground truth value in `target`, and the `target` itself.
        """
        params_out = self._params_from_output(model_output)
        loss = self._nllloss(params_out, target)

        prior_params = self._prior_params
        use_prior = prior_params is not None and any(
            p is not None for p in prior_params
        )
        if use_prior:
            out_distr = self._distr_from_params(params_out)
            device = params_out[0].device
            prior_params = tuple(
                # use model output as "prior" for parameters not specified as prior
                torch.tensor(prior_params[i]).to(device)
                if prior_params[i] is not None
                else params_out[i]
                for i in range(len(prior_params))
            )
            prior_distr = self._distr_from_params(prior_params)

            # Loss regularization using the prior distribution
            loss += self.prior_strength * torch.mean(
                kl_divergence(prior_distr, out_distr)
            )

        return loss

    def _nllloss(self, params_out, target):
        """
        This is the basic way to compute the NLL loss. It can be overwritten by likelihoods for which
        PyTorch proposes a numerically better NLL loss.
        """
        out_distr = self._distr_from_params(params_out)
        return -out_distr.log_prob(target).mean()

    @property
    def _prior_params(self):
        """
        Has to be overwritten by the Likelihood objects supporting specifying a prior distribution on the
        outputs. If it returns None, no prior will be used and the model will be trained with plain maximum likelihood.
        """
        return None

    @abstractmethod
    def _distr_from_params(self, params: Tuple) -> torch.distributions.Distribution:
        """
        Returns a torch distribution built with the specified params
        """
        pass

    @abstractmethod
    def _params_from_output(self, model_output: torch.Tensor):
        """
        Returns the distribution parameters, obtained from the raw model outputs
        (e.g. applies softplus or sigmoids to get parameters in the expected domains).
        """
        pass

    @abstractmethod
    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        """
        Samples a prediction from the probability distributions defined by the specific likelihood model
        and the parameters given in `model_output`.
        """
        pass

    @property
    @abstractmethod
    def num_parameters(self) -> int:
        """
        Returns the number of parameters that define the probability distribution for one single
        target value.
        """
        pass


class GaussianLikelihood(Likelihood):
    def __init__(self, prior_mu=None, prior_sigma=None, prior_strength=1.0):
        """
        Univariate Gaussian distribution.
        https://en.wikipedia.org/wiki/Normal_distribution
        - Univariate continuous distribution.
        - Support: :math:`\\mathbb{R}`.
        - Parameters: mean :math:`\\mu \\in \\mathbb{R}`, standard deviation :math:`\\sigma > 0`.
        Parameters
        ----------
        prior_mu
            mean of the prior Gaussian distribution (default: None).
        prior_sigma
            standard deviation (or scale) of the prior Gaussian distribution (default: None)
        prior_strength
            strength of the loss regularisation induced by the prior
        """
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        _check_strict_positive(self.prior_sigma, "sigma")

        self.nllloss = nn.GaussianNLLLoss(reduction="mean", full=True)
        self.softplus = nn.Softplus()

        super().__init__(prior_strength)

    def _nllloss(self, params_out, target):
        means_out, sigmas_out = params_out
        return self.nllloss(
            means_out.contiguous(), target.contiguous(), sigmas_out.contiguous()
        )

    @property
    def _prior_params(self):
        return self.prior_mu, self.prior_sigma

    def _distr_from_params(self, params):
        mu, sigma = params
        return _Normal(mu, sigma)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        mu, sigma = self._params_from_output(model_output)
        return torch.normal(mu, sigma)

    @property
    def num_parameters(self) -> int:
        return 2

    def _params_from_output(self, model_output):
        mu = model_output[:, :, :, 0]
        sigma = self.softplus(model_output[:, :, :, 1])
        return mu, sigma


class BetaLikelihood(Likelihood):
    def __init__(self, prior_alpha=None, prior_beta=None, prior_strength=1.0):
        """
        Beta distribution.
        https://en.wikipedia.org/wiki/Beta_distribution
        - Univariate continuous distribution.
        - Support: open interval :math:`(0,1)`
        - Parameters: shape parameters :math:`\\alpha > 0` and :math:`\\beta > 0`.
        Parameters
        ----------
        prior_alpha
            shape parameter :math:`\\alpha` of the Beta distribution, strictly positive (default: None)
        prior_beta
            shape parameter :math:`\\beta` distribution, strictly positive (default: None)
        prior_strength
            strength of the loss regularisation induced by the prior
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        _check_strict_positive(self.prior_alpha, "alpha")
        _check_strict_positive(self.prior_beta, "beta")

        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        return self.prior_alpha, self.prior_beta

    def _distr_from_params(self, params):
        alpha, beta = params
        return _Beta(alpha, beta)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        alpha, beta = self._params_from_output(model_output)
        distr = _Beta(alpha, beta)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        return 2

    def _params_from_output(self, model_output):
        alpha = self.softplus(model_output[:, :, :, 0])
        beta = self.softplus(model_output[:, :, :, 1])
        return alpha, beta


class ExponentialLikelihood(Likelihood):
    def __init__(self, prior_lambda=None, prior_strength=1.0):
        """
        Exponential distribution.
        https://en.wikipedia.org/wiki/Exponential_distribution
        - Univariate continuous distribution.
        - Support: :math:`\\mathbb{R}_{>0}`.
        - Parameter: rate :math:`\\lambda > 0`.
        Parameters
        ----------
        prior_lambda
            rate :math:`\\lambda` of the prior exponential distribution (default: None).
        prior_strength
            strength of the loss regularisation induced by the prior
        """
        self.prior_lambda = prior_lambda
        _check_strict_positive(self.prior_lambda, "lambda")
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        return (self.prior_lambda,)

    def _distr_from_params(self, params: Tuple):
        lmbda = params[0]
        return _Exponential(lmbda)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        lmbda = self._params_from_output(model_output)
        distr = _Exponential(lmbda)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        return 1

    def _params_from_output(self, model_output: torch.Tensor):
        lmbda = self.softplus(model_output.squeeze(dim=-1))
        return lmbda


class GammaLikelihood(Likelihood):
    def __init__(self, prior_alpha=None, prior_beta=None, prior_strength=1.0):
        """
        Gamma distribution.
        https://en.wikipedia.org/wiki/Gamma_distribution
        - Univariate continuous distribution
        - Support: :math:`\\mathbb{R}_{>0}`.
        - Parameters: shape :math:`\\alpha > 0` and rate :math:`\\beta > 0`.
        Parameters
        ----------
        prior_alpha
            shape :math:`\\alpha` of the prior gamma distribution (default: None).
        prior_beta
            rate :math:`\\beta` of the prior gamma distribution (default: None).
        prior_strength
            strength of the loss regularisation induced by the prior
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        _check_strict_positive(self.prior_alpha, "alpha")
        _check_strict_positive(self.prior_beta, "beta")
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        return self.prior_alpha, self.prior_beta

    def _distr_from_params(self, params: Tuple):
        alpha, beta = params
        return _Gamma(alpha, beta)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        alpha, beta = self._params_from_output(model_output)
        distr = _Gamma(alpha, beta)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        return 2

    def _params_from_output(self, model_output: torch.Tensor):
        alpha = self.softplus(model_output[:, :, :, 0])
        beta = self.softplus(model_output[:, :, :, 1])
        return alpha, beta


class GumbelLikelihood(Likelihood):
    def __init__(self, prior_mu=None, prior_beta=None, prior_strength=1.0):
        """
        Gumbel distribution.
        https://en.wikipedia.org/wiki/Gumbel_distribution
        - Univariate continuous distribution
        - Support: :math:`\\mathbb{R}`.
        - Parameters: location :math:`\\mu \\in \\mathbb{R}` and scale :math:`\\beta > 0`.
        Parameters
        ----------
        prior_mu
            location :math:`\\mu` of the prior Gumbel distribution (default: None).
        prior_beta
            scale :math:`\\beta` of the prior Gumbel distribution (default: None).
        prior_strength
            strength of the loss regularisation induced by the prior
        """
        self.prior_mu = prior_mu
        self.prior_beta = prior_beta
        _check_strict_positive(self.prior_beta)
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        return self.prior_mu, self.prior_beta

    def _distr_from_params(self, params: Tuple):
        mu, beta = params
        return _Gumbel(mu, beta)

    def sample(self, model_output):
        mu, beta = self._params_from_output(model_output)
        distr = _Gumbel(mu, beta)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        return 2

    def _params_from_output(self, model_output: torch.Tensor):
        mu = model_output[:, :, :, 0]
        beta = self.softplus(model_output[:, :, :, 1])
        return mu, beta


class HalfNormalLikelihood(Likelihood):
    def __init__(self, prior_sigma=None, prior_strength=1.0):
        """
        Half-normal distribution.
        https://en.wikipedia.org/wiki/Half-normal_distribution
        - Univariate continuous distribution.
        - Support: :math:`\\mathbb{R}_{>0}`.
        - Parameter: rate :math:`\\sigma > 0`.
        Parameters
        ----------
        prior_sigma
            standard deviation :math:`\\sigma` of the prior half-normal distribution (default: None).
        prior_strength
            strength of the loss regularisation induced by the prior
        """
        self.prior_sigma = prior_sigma
        _check_strict_positive(self.prior_sigma, "sigma")
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        return (self.prior_sigma,)

    def _distr_from_params(self, params: Tuple):
        sigma = params[0]
        return _HalfNormal(sigma)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        sigma = self._params_from_output(model_output)
        distr = _HalfNormal(sigma)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        return 1

    def _params_from_output(self, model_output: torch.Tensor):
        sigma = self.softplus(model_output.squeeze(dim=-1))
        return sigma


class LogNormalLikelihood(Likelihood):
    def __init__(self, prior_mu=None, prior_sigma=None, prior_strength=1.0):
        """
        Log-normal distribution.
        https://en.wikipedia.org/wiki/Log-normal_distribution
        - Univariate continuous distribution.
        - Support: :math:`\\mathbb{R}_{>0}`.
        - Parameters: :math:`\\mu \\in \\mathbb{R}` and :math:`\\sigma > 0`.
        Parameters
        ----------
        prior_mu
            parameter :math:`\\mu` of the prior log-normal distribution (default: None).
        prior_sigma
            parameter :math:`\\sigma` of the prior log-normal distribution (default: None)
        prior_strength
            strength of the loss regularisation induced by the prior
        """
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        _check_strict_positive(self.prior_sigma, "sigma")
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        return self.prior_mu, self.prior_sigma

    def _distr_from_params(self, params):
        mu, sigma = params
        return _LogNormal(mu, sigma)

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        mu, sigma = self._params_from_output(model_output)
        distr = _LogNormal(mu, sigma)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        return 2

    def _params_from_output(self, model_output):
        mu = model_output[:, :, :, 0]
        sigma = self.softplus(model_output[:, :, :, 1])
        return mu, sigma


class WeibullLikelihood(Likelihood):
    def __init__(self, prior_strength=1.0):
        """
        Weibull distribution.
        https://en.wikipedia.org/wiki/Weibull_distribution
        - Univariate continuous distribution
        - Support: :math:`\\mathbb{R}_{>0}`.
        - Parameters: scale :math:`\\lambda > 0` and concentration :math:`k > 0`.
        It does not support priors.
        Parameters
        ----------
        prior_strength
            strength of the loss regularisation induced by the prior
        """
        self.softplus = nn.Softplus()
        super().__init__(prior_strength)

    @property
    def _prior_params(self):
        return None

    def _distr_from_params(self, params: Tuple):
        lmba, k = params
        return _Weibull(lmba, k)

    def sample(self, model_output):
        lmbda, k = self._params_from_output(model_output)
        distr = _Weibull(lmbda, k)
        return distr.sample()

    @property
    def num_parameters(self) -> int:
        return 2

    def _params_from_output(self, model_output: torch.Tensor):
        lmbda = self.softplus(model_output[:, :, :, 0])
        k = self.softplus(model_output[:, :, :, 1])
        return lmbda, k


class QuantileRegression(Likelihood):
    def __init__(self, quantiles: Optional[List[float]] = None):
        """
        The "likelihood" corresponding to quantile regression.
        It uses the Quantile Loss Metric for custom quantiles centered around q=0.5.
        This class can be used as any other Likelihood objects even though it is not
        representing the likelihood of a well defined distribution.
        Parameters:
        -----------
        quantiles
            list of quantiles
        """

        super().__init__()
        if quantiles is None:
            self.quantiles = [
                0.01,
                0.05,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,
                0.99,
            ]
        else:
            self.quantiles = sorted(quantiles)
        _check_quantiles(self.quantiles)
        self._median_idx = self.quantiles.index(0.5)
        self.first = True
        self.quantiles_tensor = None

    def sample(self, model_output: torch.Tensor) -> torch.Tensor:
        """
        Sample uniformly between [0, 1] (for each batch example) and return the linear interpolation between the fitted
        quantiles closest to the sampled value.
        model_output is of shape (batch_size, n_timesteps, n_components, n_quantiles)
        """
        device = model_output.device
        num_samples, n_timesteps, n_components, n_quantiles = model_output.shape

        # obtain samples
        probs = torch.rand(
            size=(
                num_samples,
                n_timesteps,
                n_components,
                1,
            )
        ).to(device)
        # add dummy dim
        probas = probs.unsqueeze(-2)

        # tile and transpose
        p = torch.tile(probas, (1, 1, 1, n_quantiles, 1)).transpose(4, 3)

        # prepare quantiles
        tquantiles = torch.tensor(self.quantiles).reshape((1, 1, 1, -1)).to(device)

        # calculate index of biggest quantile smaller than the sampled value
        left_idx = torch.sum(p > tquantiles, dim=-1)

        # obtain index of smallest quantile bigger than sampled value
        right_idx = left_idx + 1

        # repeat the model output on the edges
        repeat_count = [1] * n_quantiles
        repeat_count[0] = 2
        repeat_count[-1] = 2
        repeat_count = torch.tensor(repeat_count).to(device)
        shifted_output = torch.repeat_interleave(model_output, repeat_count, dim=-1)

        # obtain model output values corresponding to the quantiles left and right of the sampled value
        left_value = torch.gather(shifted_output, index=left_idx, dim=-1)
        right_value = torch.gather(shifted_output, index=right_idx, dim=-1)

        # add 0 and 1 to quantiles
        ext_quantiles = [0.0] + self.quantiles + [1.0]
        expanded_q = torch.tile(torch.tensor(ext_quantiles), left_idx.shape).to(device)

        # calculate closest quantiles to the sampled value
        left_q = torch.gather(expanded_q, index=left_idx, dim=-1)
        right_q = torch.gather(expanded_q, index=right_idx, dim=-1)

        # linear interpolation
        weights = (probs - left_q) / (right_q - left_q)
        inter = left_value + weights * (right_value - left_value)

        return inter.squeeze(-1)

    @property
    def num_parameters(self) -> int:
        return len(self.quantiles)

    def compute_loss(self, model_output: torch.Tensor, target: torch.Tensor):
        """
        We are re-defining a custom loss (which is not a likelihood loss) compared to Likelihood
        Parameters
        ----------
        model_output
            must be of shape (batch_size, n_timesteps, n_target_variables, n_quantiles)
        target
            must be of shape (n_samples, n_timesteps, n_target_variables)
        """

        dim_q = 3

        batch_size, length = model_output.shape[:2]
        device = model_output.device

        # test if torch model forward produces correct output and store quantiles tensor
        if self.first:
            raise_if_not(
                len(model_output.shape) == 4
                and len(target.shape) == 3
                and model_output.shape[:2] == target.shape[:2],
                "mismatch between predicted and target shape",
            )
            raise_if_not(
                model_output.shape[dim_q] == len(self.quantiles),
                "mismatch between number of predicted quantiles and target quantiles",
            )
            self.quantiles_tensor = torch.tensor(self.quantiles).to(device)
            self.first = False

        errors = target.unsqueeze(-1) - model_output
        losses = torch.max(
            (self.quantiles_tensor - 1) * errors, self.quantiles_tensor * errors
        )

        return losses.sum(dim=dim_q).mean()

    def _distr_from_params(self, params: Tuple) -> None:
        # This should not be called in this class (we are abusing Likelihood)
        return None

    def _params_from_output(self, model_output: torch.Tensor) -> None:
        # This should not be called in this class (we are abusing Likelihood)
        return None



def _check_quantiles(quantiles):
    raise_if_not(
        all([0 < q < 1 for q in quantiles]),
        "All provided quantiles must be between 0 and 1.",
    )

    # we require the median to be present and the quantiles to be symmetric around it,
    # for correctness of sampling.
    median_q = 0.5
    raise_if_not(
        median_q in quantiles, "median quantile `q=0.5` must be in `quantiles`"
    )
    is_centered = [
        -1e-6 < (median_q - left_q) + (median_q - right_q) < 1e-6
        for left_q, right_q in zip(quantiles, quantiles[::-1])
    ]
    raise_if_not(
        all(is_centered),
        "quantiles lower than `q=0.5` need to share same difference to `0.5` as quantiles "
        "higher than `q=0.5`",
    )