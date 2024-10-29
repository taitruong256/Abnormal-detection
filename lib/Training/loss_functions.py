import torch
import torch.nn.functional as F

def central_custom_loss(beta, x_true, x_pred, mean, log_var, z, lambda_, latent_dim, input_shape):
    reconstruction_loss = (torch.norm(x_pred - x_true, p=2, dim=(1, 2, 3)) ** 2).mean() / (input_shape * input_shape)
    kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / latent_dim
    centralization_loss = torch.mean(z.pow(2))

    total_loss = reconstruction_loss 
    return total_loss, reconstruction_loss, beta * kl_divergence, centralization_loss

def decentral_custom_loss(beta, x_true, x_pred, mean, log_var, z, lambda_, latent_dim, input_shape):
    reconstruction_loss = (torch.norm(x_pred - x_true, p=2, dim=(1, 2, 3)) ** 2).mean() / (input_shape * input_shape)
    kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / latent_dim
    decentralization_loss = latent_dim - torch.sum(torch.bmm(z.unsqueeze(1), z.unsqueeze(2))) / z.size(0)

    total_loss = reconstruction_loss.mean() + beta * kl_divergence + decentralization_loss * lambda_
    return total_loss, reconstruction_loss.mean(), beta * kl_divergence, decentralization_loss * lambda_