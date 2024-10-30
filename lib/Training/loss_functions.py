import torch

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
