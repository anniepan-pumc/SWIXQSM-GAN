import numpy as np
import torch
from scipy.linalg import sqrtm
import torch.nn.functional as F
from torchmetrics.image.inception import InceptionScore


def cal_fid(fig1, fig2):
    # calculate mean and covariance statistics
    mu1, sigma1 = fig1.mean(axis=0), np.cov(fig1, rowvar=False)
    mu2, sigma2 = fig2.mean(axis=0), np.cov(fig2, rowvar=False)
    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'
    # calculate sum squared difference between means
    sqr_mu = (mu1 - mu2) ** 2.0
    ssdiff = torch.sum(sqr_mu,dim=0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def cal_IS(real,fake):
    p_yx = fake / fake.sum(dim=1, keepdim=True)
    p_y = real.mean(dim=0, keepdim=True)
    kl_divergence = F.kl_div(p_yx.log(), p_y.expand_as(p_yx), reduction='none').sum(dim=1)
    # Calculate the Inception Score
    kl_divergence_mean = kl_divergence.mean().item()
    return torch.exp(kl_divergence_mean)


if __name__ == '__main__':
    import torch

    qsm = torch.randn((1, 256, 256)).to('mps')
    swi = torch.randn((1, 256, 256)).to('mps')
    print(cal_fid(qsm.cpu().squeeze(), swi.cpu().squeeze()))
    print(cal_IS(qsm,swi),cal_IS(swi,qsm))

