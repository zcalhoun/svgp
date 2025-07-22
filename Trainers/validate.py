"""
This code contains the validation logic for a Gaussian Process model.

"""

import numpy as np
import torch
import gpytorch


def generate_maps(model, likelihood, test_loader):
    """
    This model assumes that we care most about generating maps of the mean
    predictions with lower and upper bounds.
    """

    model.eval()
    likelihood.eval()

    mean_preds = []
    upper95 = []
    lower95 = []
    upper90 = []
    lower90 = []
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(1000):

        for (X,) in test_loader:
            if torch.cuda.is_available():
                X = X.cuda()

            preds = likelihood(model(X))

            if isinstance(likelihood, gpytorch.likelihoods.GaussianLikelihood):
                # For Gaussian likelihood, we can directly use the mean and stddev
                mean_preds.extend(preds.mean.cpu().numpy())
                upper95.extend((preds.mean + 1.96 * preds.stddev).cpu().numpy())
                lower95.extend((preds.mean - 1.96 * preds.stddev).cpu().numpy())
                upper90.extend((preds.mean + 1.645 * preds.stddev).cpu().numpy())
                lower90.extend((preds.mean - 1.645 * preds.stddev).cpu().numpy())
            else:
                mean_preds.extend(preds.mean.mean(dim=0).cpu().numpy())

                sample = preds.sample()
                upper95.extend(torch.quantile(sample, 0.975, dim=0).cpu().numpy())
                lower95.extend(torch.quantile(sample, 0.025, dim=0).cpu().numpy())
                upper90.extend(torch.quantile(sample, 0.95, dim=0).cpu().numpy())
                lower90.extend(torch.quantile(sample, 0.05, dim=0).cpu().numpy())

    return {
        "pred": np.array(mean_preds),
        "upper95": np.array(upper95),
        "lower95": np.array(lower95),
        "upper90": np.array(upper90),
        "lower90": np.array(lower90),
    }


def validate_model(model, likelihood, test_loader):
    """
    Validate the model
    """

    model.eval()
    likelihood.eval()
    mae = 0
    wmae = 0
    mse = 0
    wmse = 0
    nlpd = 0
    wnlpd = 0
    qce50 = 0
    qce75 = 0
    qce90 = 0
    qce95 = 0
    count = 0
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(100):

        for X, y, w in test_loader:
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
                w = w.cuda()

            preds = likelihood(model(X))

            mean_preds = preds.mean
            if mean_preds.dim() == 1:
                mae += torch.sum(torch.abs(mean_preds - y))
                wmae += torch.abs(mean_preds - y) @ w

                mse += torch.sum((mean_preds - y) ** 2)
                wmse += ((mean_preds - y) ** 2) @ w

                nlpd += torch.sum(-preds.log_prob(y))
                wnlpd += -preds.log_prob(y) @ w

                qce50 += quantile_coverage_error(preds, y, 50.0).item() * y.shape[0]

                qce75 += quantile_coverage_error(preds, y, 75.0).item() * y.shape[0]

                qce90 += quantile_coverage_error(preds, y, 90.0).item() * y.shape[0]

                qce95 += quantile_coverage_error(preds, y, 95.0).item() * y.shape[0]
            else:
                mae += torch.sum(torch.abs(mean_preds.mean(axis=0) - y))
                wmae += torch.abs(mean_preds.mean(axis=0) - y) @ w

                mse += torch.sum((mean_preds.mean(axis=0) - y) ** 2)
                wmse += ((mean_preds.mean(axis=0) - y) ** 2) @ w

                S, _ = mean_preds.shape
                nlpd += torch.sum(
                    -torch.logsumexp(preds.log_prob(y), dim=0)
                    + torch.log(torch.tensor(S))
                )
                wnlpd += (
                    -torch.logsumexp(preds.log_prob(y), dim=0)
                    + torch.log(torch.tensor(S))
                ) @ w

                samples = preds.sample()
                qce50 += t_qce_coverage(samples, y, alpha=50.0) * y.shape[0]

                qce75 += t_qce_coverage(samples, y, alpha=75.0) * y.shape[0]

                qce90 += t_qce_coverage(samples, y, alpha=90.0) * y.shape[0]

                qce95 += t_qce_coverage(samples, y, alpha=95.0) * y.shape[0]
            count += y.size(0)
    mae /= count
    mse /= count
    nlpd /= count
    qce50 /= count
    qce75 /= count
    qce90 /= count
    qce95 /= count

    wmae /= count
    wmse /= count
    wnlpd /= count

    return {
        "mae": mae.item(),
        "mse": mse.item(),
        "nlpd": nlpd.item(),
        "qce50": qce50,
        "qce75": qce75,
        "qce90": qce90,
        "qce95": qce95,
        "wmae": wmae.item(),
        "wmse": wmse.item(),
        "wnlpd": wnlpd.item(),
    }


def quantile_coverage_error(pred_dist, test_y, quantile):
    """
    Quantile coverage error for normal distributions
    """
    if quantile <= 0 or quantile >= 100:
        raise NotImplementedError("Quantile must be between 0 and 100")
    # combine_dim = -2 if isinstance(pred_dist, MultitaskMultivariateNormal) else -1
    standard_normal = torch.distributions.Normal(loc=0.0, scale=1.0)
    deviation = standard_normal.icdf(torch.as_tensor(0.5 + 0.5 * (quantile / 100)))
    lower = pred_dist.mean - deviation * pred_dist.stddev
    upper = pred_dist.mean + deviation * pred_dist.stddev
    n_samples_within_bounds = ((test_y > lower) * (test_y < upper)).sum(-1)
    fraction = n_samples_within_bounds / test_y.shape[-1]
    return fraction - quantile / 100


def t_qce_coverage(y_samples, y_true, alpha=95.0):
    """
    Compute empirical coverage of central prediction intervals using PyTorch.

    Parameters
    ----------
    y_samples : torch.Tensor
        Shape (S, N) — predictive samples from the posterior
    y_true : torch.Tensor
        Shape (N,) — true labels
    alpha : float
        Desired coverage level (e.g., 90. for 90%)

    Returns
    -------
    coverage error: float
        Distance from the desired coverage level
    """
    # Compute quantiles across samples (dim=0 → across S samples per point)
    lower = torch.quantile(y_samples, q=(1 - alpha / 100) / 2, dim=0)
    upper = torch.quantile(y_samples, q=(1 + alpha / 100) / 2, dim=0)

    # Check if true values are inside the intervals
    inside = (y_true >= lower) & (y_true <= upper)

    fraction = inside.float().mean()

    return (fraction - alpha / 100).item()
