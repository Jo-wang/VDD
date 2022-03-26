import torch
import abc
from torch import nn, autograd
import torch.nn.functional as F
import math
import torch.nn.init as init


from .utils import log_importance_weight_matrix, matrix_log_density_gaussian, log_density_gaussian
LOSSES = ["VAE", "betaH", "betaB", "factor", "btcvae"]
RECON_DIST = ["bernoulli", "laplace", "gaussian"]


def euclidean(x1, x2):
    return ((x1 - x2) ** 2).sum().sqrt()

def cross_entropy_soft(pred):
    softmax = nn.Softmax(dim=1)
    logsoftmax = nn.LogSoftmax(dim=1)
    loss = torch.mean(torch.sum(-softmax(pred) * logsoftmax(pred), 1))
    return loss

def osbp_loss(output, target):
    output_neg = 1 - output
    target_neg = 1 - target
    result = torch.mean(target * torch.log(output + 1e-6))
    result += torch.mean(target_neg * torch.log(output_neg + 1e-6))
    return -torch.mean(result)

def k_moment(output_s1, output_s2, output_s3, output_s4, output_t, k):
    output_s1 = (output_s1 ** k).mean(0)
    output_s2 = (output_s2 ** k).mean(0)
    output_s3 = (output_s3 ** k).mean(0)
    output_t = (output_t ** k).mean(0)
    return euclidean(output_s1, output_t) + euclidean(output_s2, output_t) + euclidean(output_s3, output_t) + \
           euclidean(output_s1, output_s2) + euclidean(output_s2, output_s3) + euclidean(output_s3, output_s1) + \
           euclidean(output_s4, output_s1) + euclidean(output_s4, output_s2) + euclidean(output_s4, output_s2) + \
           euclidean(output_s4, output_t)


def msda_regulizer(output_s1, output_s2, output_s3, output_s4, output_t, belta_moment):
    # print('s1:{}, s2:{}, s3:{}, s4:{}'.format(output_s1.shape, output_s2.shape, output_s3.shape, output_t.shape))
    s1_mean = output_s1.mean(0)
    s2_mean = output_s2.mean(0)
    s3_mean = output_s3.mean(0)
    t_mean = output_t.mean(0)
    output_s1 = output_s1 - s1_mean
    output_s2 = output_s2 - s2_mean
    output_s3 = output_s3 - s3_mean
    output_t = output_t - t_mean
    moment1 = euclidean(output_s1, output_t) + euclidean(output_s2, output_t) + euclidean(output_s3, output_t) + \
              euclidean(output_s1, output_s2) + euclidean(output_s2, output_s3) + euclidean(output_s3, output_s1) + \
              euclidean(output_s4, output_s1) + euclidean(output_s4, output_s2) + euclidean(output_s4, output_s2) + \
              euclidean(output_s4, output_t)
    reg_info = moment1
    # print(reg_info)
    for i in range(belta_moment - 1):
        reg_info += k_moment(output_s1, output_s2, output_s3, output_s4, output_t, i + 2)

    return reg_info / 6

class FocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, output, target):

        input_soft = (output + 1e-6).float()
        batch_size, num_class = output.shape
        # @note change the lower bound of input to avoid gradient issues
        min_pro = torch.Tensor(0.001, dtype=torch.float32)
        input_soft = torch.max(input_soft, min_pro)

        # create the labels one hot tensor
        target_one_hot = torch.FloatTensor(batch_size, num_class).cuda()
        target_one_hot.zero_()
        target_one_hot.scatter_(1, target.view(-1, 1), 1)

        # compute the actual focal loss
        weight = torch.pow(torch.tensor(1.) - input_soft, self.gamma).float()

        focal = -self.alpha * weight * torch.log(input_soft)
        loss_tmp = torch.sum(target_one_hot * focal, dim=1)

        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError("Invalid reduction mode: {}"
                                      .format(self.reduction))
        return loss



class Exclusive(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, V, momentum):
        ctx.mark_dirty(inputs)
        ctx.mark_dirty(V)
        ctx.save_for_backward(inputs, targets, V)

        outputs = inputs.mm(V.t())
        for x, y in zip(inputs, targets):
            V[y] = momentum * V[y] + (1 - momentum) * x
        return outputs, V

    @staticmethod
    def backward(ctx, grad_outputs, grad_V):
        inputs, targets, V = ctx.saved_tensors
        grad_inputs = grad_outputs.clone().mm(V) if ctx.needs_input_grad[0] else None

        return grad_inputs, None, None, None


class ExLoss(nn.Module):
    def __init__(self, num_features, num_classes, t=1.0,
                 weight=None):
        super(ExLoss, self).__init__()
        self.num_features = num_features
        self.t = t
        self.weight = weight
        self.delta_known = 0.9
        self.delta_unk = 0.3
        self.momentum = 0.9
        self.register_buffer('V', torch.zeros(num_classes, num_features))
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        init.kaiming_uniform_(self.V, a=math.sqrt(5))


    def forward(self, inputs, targets=None):
        # pseudo labeling for the target domain
        if targets is None:
            try_outputs = F.softmax(inputs.mm(self.V.t()), dim=-1)
            idx_known = try_outputs[:, :-1].max(dim=-1)[0] > self.delta_known
            idx_unk = try_outputs[:, :-1].max(dim=-1)[0] < self.delta_unk
            inputs_list = []
            targets_list = []
            if idx_known.any():
                inputs_list.append(inputs[idx_known])
                targets_list.append(try_outputs[idx_known, :-1].max(dim=-1)[1].long())
            if idx_unk.any():
                inputs_list.append(inputs[idx_unk])
                targets_list.append((self.V.size()[0] - 1) * torch.ones(inputs[idx_unk].size()[0], ).long().cuda())
            # nothing to learn for target
            if len(inputs_list) == 0:
                return 0
            else:
                inputs = torch.cat(inputs_list, dim=0)
                targets = torch.cat(targets_list, dim=0).long()
            self.momentum = 0.5

        # for the source domains

        outputs, self.V = Exclusive.apply(inputs, targets, self.V.clone(), self.momentum)
        if outputs is None or targets is None:
            print("check")
        loss = self.criterion(outputs, targets)

        return loss


def get_loss_f(loss_name, **kwargs_parse):
    """Return the correct loss function given the argparse arguments."""
    kwargs_all = dict(rec_dist=kwargs_parse["rec_dist"],
                      steps_anneal=kwargs_parse["reg_anneal"])
    if loss_name == "betaH":
        return BetaHLoss(beta=kwargs_parse["btcvae_B"], **kwargs_all)
    elif loss_name == "VAE":
        return BetaHLoss(beta=1, **kwargs_all)
    elif loss_name == "betaB":
        return BetaBLoss(C_init=kwargs_parse["betaB_initC"],
                         C_fin=kwargs_parse["betaB_finC"],
                         gamma=kwargs_parse["betaB_G"], **kwargs_all)
    elif loss_name == "factor":
        return FactorKLoss(kwargs_parse["device"],
                           gamma=kwargs_parse["factor_G"],
                           disc_kwargs=dict(latent_dim=kwargs_parse["latent_dim"]),
                           optim_kwargs=dict(lr=kwargs_parse["lr_disc"], betas=(0.5, 0.9)), **kwargs_all)
    elif loss_name == "btcvae":
        return BtcvaeLoss(kwargs_parse["n_data"],
                          alpha=kwargs_parse["btcvae_A"],
                          beta=kwargs_parse["btcvae_B"],
                          gamma=kwargs_parse["btcvae_G"], **kwargs_all)
    else:
        assert loss_name not in LOSSES
        raise ValueError("Uknown loss : {}".format(loss_name))


class BaseLoss(abc.ABC):
    """
    Base class for losses.
    Parameters
    ----------
    record_loss_every: int, optional
        Every how many steps to recorsd the loss.
    rec_dist: {"bernoulli", "gaussian", "laplace"}, optional
        Reconstruction distribution istribution of the likelihood on the each pixel.
        Implicitely defines the reconstruction loss. Bernoulli corresponds to a
        binary cross entropy (bse), Gaussian corresponds to MSE, Laplace
        corresponds to L1.
    steps_anneal: nool, optional
        Number of annealing steps where gradually adding the regularisation.
    """

    def __init__(self, record_loss_every=50, rec_dist="bernoulli", steps_anneal=0):
        self.n_train_steps = 0
        self.record_loss_every = record_loss_every
        self.rec_dist = rec_dist
        self.steps_anneal = steps_anneal

    @abc.abstractmethod
    def __call__(self, data, recon_data, latent_dist, is_train, storer, **kwargs):
        """
        Calculates loss for a batch of data.
        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images). Shape : (batch_size, n_chan,
            height, width).
        recon_data : torch.Tensor
            Reconstructed data. Shape : (batch_size, n_chan, height, width).
        latent_dist : tuple of torch.tensor
            sufficient statistics of the latent dimension. E.g. for gaussian
            (mean, log_var) each of shape : (batch_size, latent_dim).
        is_train : bool
            Whether currently in train mode.
        storer : dict
            Dictionary in which to store important variables for vizualisation.
        kwargs:
            Loss specific arguments
        """




class BetaHLoss(BaseLoss):
    """
    Compute the Beta-VAE loss as in [1]
    Parameters
    ----------
    beta : float, optional
        Weight of the kl divergence.
    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.
    References
    ----------
        [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
        a constrained variational framework." (2016).
    """

    def __init__(self, beta=4, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def __call__(self,  data, recon_batch, d_dist, d_samp, z_dist, z_samp, **kwargs):


        rec_loss = _reconstruction_loss(data, recon_batch,
                                        distribution=self.rec_dist)
        kl_loss_d = _kl_normal_loss(*d_dist)
        kl_loss_z = _kl_normal_loss(*z_dist)

        anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal))
        loss = rec_loss + anneal_reg * (self.beta * (kl_loss_d + kl_loss_z))

        return loss


class BetaBLoss(BaseLoss):
    """
    Compute the Beta-VAE loss as in [1]
    Parameters
    ----------
    C_init : float, optional
        Starting annealed capacity C.
    C_fin : float, optional
        Final annealed capacity C.
    gamma : float, optional
        Weight of the KL divergence term.
    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.
    References
    ----------
        [1] Burgess, Christopher P., et al. "Understanding disentangling in
        $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
    """

    def __init__(self, C_init=0., C_fin=20., gamma=100., **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.C_init = C_init
        self.C_fin = C_fin

    def __call__(self, data, recon_data, latent_dist, is_train, storer, **kwargs):


        rec_loss = _reconstruction_loss(data, recon_data,
                                        storer=storer,
                                        distribution=self.rec_dist)
        kl_loss = _kl_normal_loss(*latent_dist, storer)

        C = (linear_annealing(self.C_init, self.C_fin, self.n_train_steps, self.steps_anneal)
             if is_train else self.C_fin)

        loss = rec_loss + self.gamma * (kl_loss - C).abs()

        if storer is not None:
            storer['loss'].append(loss.item())

        return loss


class FactorKLoss(BaseLoss):
    """
    Compute the Factor-VAE loss as per Algorithm 2 of [1]
    Parameters
    ----------
    device : torch.device
    gamma : float, optional
        Weight of the TC loss term. `gamma` in the paper.
    discriminator : disvae.discriminator.Discriminator
    optimizer_d : torch.optim
    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.
    References
    ----------
        [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
        arXiv preprint arXiv:1802.05983 (2018).
    """

    def __init__(self, device,
                 gamma=10.,
                 disc_kwargs={},
                 optim_kwargs=dict(lr=5e-5, betas=(0.5, 0.9)),
                 **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.device = device
        self.discriminator = Discriminator(**disc_kwargs).to(self.device)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), **optim_kwargs)

    def __call__(self, *args, **kwargs):
        raise ValueError("Use `call_optimize` to also train the discriminator")

    def call_optimize(self, data, model, optimizer, storer):

        # factor-vae split data into two batches. In the paper they sample 2 batches
        batch_size = data.size(dim=0)
        half_batch_size = batch_size // 2
        data = data.split(half_batch_size)
        data1 = data[0]
        data2 = data[1]

        # Factor VAE Loss
        recon_batch, latent_dist, latent_sample1 = model(data1)
        rec_loss = _reconstruction_loss(data1, recon_batch,
                                        storer=storer,
                                        distribution=self.rec_dist)

        kl_loss = _kl_normal_loss(*latent_dist, storer)

        d_z = self.discriminator(latent_sample1)
        # We want log(p_true/p_false). If not using logisitc regression but softmax
        # then p_true = exp(logit_true) / Z; p_false = exp(logit_false) / Z
        # so log(p_true/p_false) = logit_true - logit_false
        tc_loss = (d_z[:, 0] - d_z[:, 1]).mean()
        # with sigmoid (not good results) should be `tc_loss = (2 * d_z.flatten()).mean()`

        anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if model.training else 1)
        vae_loss = rec_loss + kl_loss + anneal_reg * self.gamma * tc_loss

        if storer is not None:
            storer['loss'].append(vae_loss.item())
            storer['tc_loss'].append(tc_loss.item())

        if not model.training:
            # don't backprop if evaluating
            return vae_loss

        # Run VAE optimizer
        optimizer.zero_grad()
        vae_loss.backward(retain_graph=True)
        optimizer.step()

        # Discriminator Loss
        # Get second sample of latent distribution
        latent_sample2 = model.sample_latent(data2)
        z_perm = _permute_dims(latent_sample2).detach()
        d_z_perm = self.discriminator(z_perm)

        # Calculate total correlation loss
        # for cross entropy the target is the index => need to be long and says
        # that it's first output for d_z and second for perm
        ones = torch.ones(half_batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros_like(ones)
        d_tc_loss = 0.5 * (F.cross_entropy(d_z, zeros) + F.cross_entropy(d_z_perm, ones))
        # with sigmoid would be :
        # d_tc_loss = 0.5 * (self.bce(d_z.flatten(), ones) + self.bce(d_z_perm.flatten(), 1 - ones))

        # TO-DO: check ifshould also anneals discriminator if not becomes too good ???
        #d_tc_loss = anneal_reg * d_tc_loss

        # Run discriminator optimizer
        self.optimizer_d.zero_grad()
        d_tc_loss.backward()
        self.optimizer_d.step()

        if storer is not None:
            storer['discrim_loss'].append(d_tc_loss.item())

        return vae_loss


class BtcvaeLoss(BaseLoss):
    """
    Compute the decomposed KL loss with either minibatch weighted sampling or
    minibatch stratified sampling according to [1]
    Parameters
    ----------
    n_data: int
        Number of data in the training set
    alpha : float
        Weight of the mutual information term.
    beta : float
        Weight of the total correlation term.
    gamma : float
        Weight of the dimension-wise KL term.
    is_mss : bool
        Whether to use minibatch stratified sampling instead of minibatch
        weighted sampling.
    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.
    References
    ----------
       [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
       autoencoders." Advances in Neural Information Processing Systems. 2018.
    """

    def __init__(self, n_data, alpha=1., beta=2., gamma=1., is_mss=True, **kwargs):
        super().__init__(**kwargs)
        self.n_data = n_data
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.is_mss = is_mss  # minibatch stratified sampling

    def __call__(self, data, recon_batch, d_dist, d_samp, z_dist, z_samp):

        rec_loss = _reconstruction_loss(data, recon_batch,
                                        distribution=self.rec_dist)
        log_pz_d, log_qz_d, log_prod_qzi_d, log_q_zCx_d = _get_log_pz_qz_prodzi_qzCx(d_samp, d_dist,
                                                                             self.n_data,
                                                                             is_mss=self.is_mss)

        # log_pz_z, log_qz_z, log_prod_qzi_z, log_q_zCx_z = _get_log_pz_qz_prodzi_qzCx(z_samp, z_dist,
        #                                                                              self.n_data,
        #                                                                              is_mss=self.is_mss)
        # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi_loss_d = (log_q_zCx_d - log_qz_d).mean()
        # mi_loss_z = (log_q_zCx_z - log_qz_z).mean()
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss_d = (log_qz_d - log_prod_qzi_d).mean()
        # tc_loss_z = (log_qz_z - log_prod_qzi_z).mean()
        # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        dw_kl_loss_d = (log_prod_qzi_d - log_pz_d).mean()
        # dw_kl_loss_z = (log_prod_qzi_z - log_pz_z).mean()
        kl_loss = _kl_normal_loss(*z_dist)
        anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal))

        # total loss
        # loss = rec_loss + (self.alpha * (mi_loss_d + mi_loss_z) +
        #                    self.beta * (tc_loss_d + tc_loss_z) +
        #                    anneal_reg * self.gamma * (dw_kl_loss_d + dw_kl_loss_z))
        loss = rec_loss + self.alpha * (mi_loss_d) + self.beta * (tc_loss_d) + \
               anneal_reg * self.gamma * (dw_kl_loss_d + kl_loss)

        return loss


def _reconstruction_loss(data, recon_data, distribution="bernoulli", storer=None):
    """
    Calculates the per image reconstruction loss for a batch of data. I.e. negative
    log likelihood.
    Parameters
    ----------
    data : torch.Tensor
        Input data (e.g. batch of images). Shape : (batch_size, n_chan,
        height, width).
    recon_data : torch.Tensor
        Reconstructed data. Shape : (batch_size, n_chan, height, width).
    distribution : {"bernoulli", "gaussian", "laplace"}
        Distribution of the likelihood on the each pixel. Implicitely defines the
        loss Bernoulli corresponds to a binary cross entropy (bse) loss and is the
        most commonly used. It has the issue that it doesn't penalize the same
        way (0.1,0.2) and (0.4,0.5), which might not be optimal. Gaussian
        distribution corresponds to MSE, and is sometimes used, but hard to train
        ecause it ends up focusing only a few pixels that are very wrong. Laplace
        distribution corresponds to L1 solves partially the issue of MSE.
    storer : dict
        Dictionary in which to store important variables for vizualisation.
    Returns
    -------
    loss : torch.Tensor
        Per image cross entropy (i.e. normalized per batch but not pixel and
        channel)
    """
    batch_size, n_domain, n_chan, height, width = recon_data.size()
    _, _, _, height_d, width_d = data.size()
    recon_data = recon_data.view(batch_size*n_domain, n_chan, height, width)
    data = data.view(batch_size*n_domain, n_chan, height_d, width_d)
    if height_d is not height or width_d is not width:
        data = F.interpolate(data, [height, width], mode='bilinear')
    is_colored = n_chan == 3

    if distribution == "bernoulli":
        loss = F.binary_cross_entropy(recon_data, data, reduction="mean")
    elif distribution == "gaussian":
        # loss in [0,255] space but normalized by 255 to not be too big
        loss = F.mse_loss(255 * recon_data, 255 * data, reduction="sum") / 255
    elif distribution == "laplace":
        # loss in [0,255] space but normalized by 255 to not be too big but
        # multiply by 255 and divide 255, is the same as not doing anything for L1
        loss = F.l1_loss(recon_data, data, reduction="mean")
        loss = loss * 3  # emperical value to give similar values than bernoulli => use same hyperparam
        loss = loss * (loss != 0)  # masking to avoid nan
    else:
        assert distribution not in RECON_DIST
        raise ValueError("Unkown distribution: {}".format(distribution))

    loss = loss / (batch_size * n_domain)

    if storer is not None:
        storer['recon_loss'].append(loss.item())

    return loss


def _kl_normal_loss(mean, logvar, storer=None):
    """
    Calculates the KL divergence between a normal distribution
    with diagonal covariance and a unit normal distribution.
    Parameters
    ----------
    mean : torch.Tensor
        Mean of the normal distribution. Shape (batch_size, latent_dim) where
        D is dimension of distribution.
    logvar : torch.Tensor
        Diagonal log variance of the normal distribution. Shape (batch_size,
        latent_dim)
    storer : dict
        Dictionary in which to store important variables for vizualisation.
    """
    latent_dim = mean.size(1)
    # batch mean of kl for each latent dimension
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
    total_kl = latent_kl.sum()

    if storer is not None:
        storer['kl_loss'].append(total_kl.item())
        for i in range(latent_dim):
            storer['kl_loss_' + str(i)].append(latent_kl[i].item())

    return total_kl


def _permute_dims(latent_sample):
    """
    Implementation of Algorithm 1 in ref [1]. Randomly permutes the sample from
    q(z) (latent_dist) across the batch for each of the latent dimensions (mean
    and log_var).
    Parameters
    ----------
    latent_sample: torch.Tensor
        sample from the latent dimension using the reparameterisation trick
        shape : (batch_size, latent_dim).
    References
    ----------
        [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
        arXiv preprint arXiv:1802.05983 (2018).
    """
    perm = torch.zeros_like(latent_sample)
    batch_size, dim_z = perm.size()

    for z in range(dim_z):
        pi = torch.randperm(batch_size).to(latent_sample.device)
        perm[:, z] = latent_sample[pi, z]

    return perm


def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed


# Batch TC specific
# TO-DO: test if mss is better!
def _get_log_pz_qz_prodzi_qzCx(samp, dist, n_data, is_mss=True):
    batch_size, n_domain, hidden_dim = samp.shape
    batch_size = batch_size * n_domain
    samp = samp.view(batch_size, hidden_dim)
    dist = (dist[0].view(batch_size, -1), dist[1].view(batch_size, -1))
    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(samp, *dist).sum(dim=1)
    _log_qz = log_density_gaussian(samp.unsqueeze(1),
                                   dist[0].unsqueeze(0), dist[1].unsqueeze(0))
    # calculate log p(z)
    # mean and log var is 0
    zeros = torch.zeros_like(samp)
    log_pz = log_density_gaussian(samp, zeros, zeros).sum(1)

    # mat_log_qz = matrix_log_density_gaussian(samp, *dist)

    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(samp.device)
        log_qz = torch.logsumexp(log_iw_mat + _log_qz.sum(2), dim=1, keepdim=False)
        log_prod_qzi = torch.logsumexp(log_iw_mat.view(batch_size, batch_size, 1) + _log_qz, dim=1, keepdim=False).sum(1)
        # mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)
    else:
        log_prod_qzi = (torch.logsumexp(_log_qz, dim=1, keepdim=False) - math.log(batch_size * n_data)).sum(1)
        log_qz = (torch.logsumexp(_log_qz.sum(2), dim=1, keepdim=False) - math.log(batch_size * n_data))
    # log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    # log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx


class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.eps = eps

    def forward(self, x, labels, logit=False):
        '''
        input shape (N, in_features)
        '''
        # assert len(x) == len(labels)
        # assert torch.min(labels) >= 0
        # assert torch.max(labels) < self.out_features

        wf = x
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        if logit:
            return L
        else:
            return -torch.mean(L)