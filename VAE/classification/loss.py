import torch.nn.functional as F
import torch
def VaeLoss(y,yPerd,q_mu,q_logvar,kld_weight=0.00025):
    """
    This loss have two part
    1. mse loss
    2. KLD loss
    """
    return F.mse_loss(yPerd,y)+ kld_weight* kld_gauss(q_mu,q_logvar)
def kld_gauss(mu, logvar):
    """
    KL divergence between two diagonal Gaussians
    in standard VAEs, the prior p(z) is a standard Gaussian.
    :param q_mu: posterior mean
    :param q_logvar: posterior log-variance
    :param mu: prior mean
    :param logvar: prior log-variance
    """
    # set prior to a standard Gaussian


    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
    return kld_loss