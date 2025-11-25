import torch
import torch.nn as nn

def reconstruction_loss(x_true, x_pred, input_shape):
    return (torch.norm(x_pred - x_true, p=2, dim=(1, 2, 3)) ** 2).mean() / (input_shape * input_shape)

def recon_loss_with_kl(beta, x_true, x_pred, input_shape, mean, log_var, latent_dim):
    recon_loss = reconstruction_loss(x_true, x_pred, input_shape)
    kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / latent_dim
    return recon_loss + beta * kl_divergence

def recon_loss_with_decentralization(x_true, x_pred, input_shape, z, lambda_, latent_dim):
    recon_loss = reconstruction_loss(x_true, x_pred, input_shape)
    decentralization_loss = latent_dim - torch.sum(torch.bmm(z.unsqueeze(1), z.unsqueeze(2))) / z.size(0)
    return recon_loss + decentralization_loss * lambda_

def KLD(mu, std):
    # numerical value for stability of log computation
    eps = 1e-8

    # Compute the KL divergence, normalized by latent dimensionality
    kld = -0.5 * torch.sum(1 + torch.log(eps + std ** 2) - (mu ** 2) - (std ** 2)) / torch.numel(mu)

    return kld

def joint_loss_function(output_samples_classification, target, output_samples_recon, inp, mu, std, device, args):
    """
    Computes the model's joint loss function consisting of a term for reconstruction, a KL term between
    approximate posterior and prior and the loss for the generative classifier. The number of samples
    is one per default, as specified in the command line parser and typically is how VAE models are trained.
    We have added the option to flexibly work with an arbitrary amount of samples.

    Parameters:
        output_samples_classification (torch.Tensor): Mini-batch of var_sample many classification prediction values.
        target (torch.Tensor): Classification targets for each element in the mini-batch.
        output_samples_recon (torch.Tensor): Mini-batch of var_sample many reconstructions.
        inp (torch.Tensor): The input mini-batch (before noise), aka the reconstruction loss' target.
        mu (torch.Tensor): Encoder (recognition model's) mini-batch of mean vectors.
        std (torch.Tensor): Encoder (recognition model's) mini-batch of standard deviation vectors.
        device (str): Device for computation.
        args (dict): Command line parameters. Needs to contain autoregression (bool).

    Returns:
        float: normalized classification loss
        float: normalized reconstruction loss
        float: normalized KL divergence
    """

    # for autoregressive models the decoder loss term corresponds to a classification based on 256 classes (for each
    # pixel value), i.e. a 256-way Softmax and thus a cross-entropy loss.
    # For regular decoders the loss is the reconstruction negative-log likelihood.
    
    recon_loss = nn.BCEWithLogitsLoss(reduction='sum')

    class_loss = nn.CrossEntropyLoss(reduction='sum')

    # Place-holders for the final loss values over all latent space samples
    recon_losses = torch.zeros(output_samples_recon.size(0)).to(device)
    cl_losses = torch.zeros(output_samples_classification.size(0)).to(device)

    # loop through each sample for each input and calculate the correspond loss. Normalize the losses.
    for i in range(output_samples_classification.size(0)):
        cl_losses[i] = class_loss(output_samples_classification[i], target) / torch.numel(target)
        recon_losses[i] = recon_loss(output_samples_recon[i], inp) / torch.numel(inp)

    # average the loss over all samples per input
    cl = torch.mean(cl_losses, dim=0)
    rl = torch.mean(recon_losses, dim=0)

    # Compute the KL divergence, normalized by latent dimensionality
    kld = KLD(mu, std)

    return cl, rl, kld