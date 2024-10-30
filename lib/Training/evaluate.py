import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from lib.Training.loss_functions import reconstruction_loss 
from lib.Openset.meta_recognition import build_weibull_model, calculate_outlier_probability

def evaluate_vae(model, device, val_close_loader, val_open_loader, latent_dim, beta, lambda_, input_shape, loss_type):
    model.eval()

    total_loss, total_reconstruction_loss, total_kl_divergence = 0.0, 0.0, 0.0
    total_samples, total_samples_close, total_samples_open = 0, 0, 0
    total_decentralization_loss_open = 0.0

    # Đánh giá trên tập close (central loss)
    with torch.no_grad():
        for data, labels in tqdm(val_close_loader, desc="Evaluating VAE (closed dataset)", leave=False):
            data, labels = data.to(device), labels.to(device)
            mean, log_var, z, data_reconstructions = model(data)
            
            # Tính các hàm loss tách biệt để đánh giá
            recon_loss = reconstruction_loss(data, data_reconstructions, input_shape)
            kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / latent_dim
            if loss_type == "recon":
                loss = recon_loss
            elif loss_type == 'recon_kld' or loss_type=='recon_kld_decentral':
                loss = recon_loss + kl_divergence

            total_reconstruction_loss += recon_loss.item()
            total_kl_divergence += kl_divergence.item()
            total_loss += loss.item()
            total_samples += len(labels)
            total_samples_close += len(labels)

    # Đánh giá trên tập open (decentral loss)
    if loss_type == "recon_kld_decentral":
        with torch.no_grad():
            for data, labels in tqdm(val_open_loader, desc="Evaluating VAE (open dataset)", leave=False):
                data, labels = data.to(device), labels.to(device)
                mean, log_var, z, data_reconstructions = model(data)
                
                recon_loss = reconstruction_loss(data, data_reconstructions, input_shape)
                decentral_loss = latent_dim - torch.sum(torch.bmm(z.unsqueeze(1), z.unsqueeze(2))) / z.size(0)
                loss = decentral_loss * lambda_
                
                total_reconstruction_loss += recon_loss.item()
                total_loss += loss.item()
                total_decentralization_loss_open += decentral_loss.item()
                total_samples += len(labels)
                total_samples_open += len(labels)
    
    avg_reconstruction_loss = total_reconstruction_loss / total_samples if total_samples > 0 else 0
    avg_kl_divergence = total_kl_divergence / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_decentralization_loss_open = total_decentralization_loss_open / total_samples_open if total_samples_open > 0 else 0

    return avg_loss, avg_reconstruction_loss, avg_kl_divergence, avg_decentralization_loss_open


def evaluate_weibull(vae, device, all_latent_vectors, mean_vector, open_loader, num_classes, latent_dim, tail_size):
    vae.eval()
    true_labels = []
    pred_labels = []
    outlier_probs = []
    weibull_model = build_weibull_model(mean_vector, all_latent_vectors, num_classes, tail_size)

    with torch.no_grad():
        for x_batch, y_batch in tqdm(open_loader, desc="Evaluating Weibull", leave=False):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            _, _, z_batch, _ = vae(x_batch)

            for z, y in zip(z_batch, y_batch):
                z = z.cpu().detach().numpy()
                outlier_probability = calculate_outlier_probability(z, mean_vector, weibull_model, num_classes)
                true_labels.append(0 if y.item() < num_classes else 1)
                outlier_probs.append(max(0, min(outlier_probability)))  # Lưu lại xác suất outlier nhỏ nhất

    true_labels = np.array(true_labels)
    outlier_probs = np.array(outlier_probs)
    
    best_f1 = 0
    best_threshold = 0.05
    best_pred_labels = np.zeros_like(true_labels)

    # Tìm ngưỡng với F1 score tốt nhất
    threshold_list = np.arange(0.05, 1, 0.05)
    for threshold in threshold_list:
        pred_labels = (outlier_probs >= threshold).astype(int)
        f1 = f1_score(true_labels, pred_labels, average=None).mean()
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_pred_labels = pred_labels

    # Tính toán accuracy và confusion matrix với ngưỡng tốt nhất
    accuracy = accuracy_score(true_labels, best_pred_labels)
    confusion = confusion_matrix(true_labels, best_pred_labels)

    return accuracy, best_f1, confusion, best_threshold