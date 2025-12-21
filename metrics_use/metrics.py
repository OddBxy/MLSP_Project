import torch
import torchaudio.functional as F


def compute_covariance(x):
    N = x.shape[0]
    x_centered = x - x.mean(dim=0)
    cov = (x_centered.T @ x_centered) / (N - 1)
    return cov


def compute_FAD(embedding_group1, embedding_group2):

  mu_group1 = embedding_group1.mean(dim=0)
  sigma_group1 = compute_covariance(embedding_group1)

  mu_group2 = embedding_group2.mean(dim=0)
  sigma_group2 = compute_covariance(embedding_group2)

  # Régularisation
  eps = 1e-6
  sigma_group1 += eps * torch.eye(sigma_group1.shape[0])
  sigma_group2 += eps * torch.eye(sigma_group2.shape[0])

  # FAD
  return F.frechet_distance(mu_group1, sigma_group1, mu_group2, sigma_group2)


def compute_mmd(x, y, sigma=1.0):
    xx = torch.cdist(x, x) ** 2
    yy = torch.cdist(y, y) ** 2
    xy = torch.cdist(x, y) ** 2

    k_xx = torch.exp(-xx / (2 * sigma**2))
    k_yy = torch.exp(-yy / (2 * sigma**2))
    k_xy = torch.exp(-xy / (2 * sigma**2))

    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()


