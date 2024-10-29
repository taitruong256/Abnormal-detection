import torch 
from tqdm import tqdm
from lib.Training.loss_functions import central_custom_loss, decentral_custom_loss

def train_vae(model, device, train_close_loader, train_open_loader, optimizer, beta, lambda_, latent_dim, input_shape):
    model.train()

    # Vector tổng cho các latent vector
    total_z_sum_close = torch.zeros(latent_dim).to(device)  
    total_samples_close = 0 
    all_latent_vectors_close = []

    # Huấn luyện trên tập close (central loss)
    progress_bar_close = tqdm(train_close_loader, desc="Training (closed dataset)", leave=False)
    for batch_idx, (data, labels) in enumerate(progress_bar_close):
        
        data, labels = data.to(device), labels.to(device)
        mean, log_var, z, data_reconstructions = model(data)
        total_loss, recon_loss, kl_loss, central_loss = central_custom_loss(beta, data, data_reconstructions, mean, log_var, z, lambda_, latent_dim, input_shape)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
        total_z_sum_close += z.sum(dim=0)  
        total_samples_close += z.size(0)  
        all_latent_vectors_close.append(z.detach().cpu())
        
        progress_bar_close.set_postfix(total_loss=total_loss.item(), batch=batch_idx+1)
        break 

    # Tính vector trung bình của z cho tập đóng
    mean_vector_close = total_z_sum_close / total_samples_close 
    mean_vector_close = mean_vector_close.detach().cpu().numpy()

#     # Huấn luyện trên tập open (decentral loss)
#     progress_bar_open = tqdm(train_open_loader, desc="Training (opened dataset)", leave=False)
#     for batch_idx, (data, labels) in enumerate(progress_bar_open):
#         data, labels = data.to(device), labels.to(device)
#         mean, log_var, z, data_reconstructions = model(data)
#         total_loss, recon_loss, kl_loss, decentral_loss = decentral_custom_loss(beta, data, data_reconstructions, mean, log_var, z, lambda_, latent_dim, input_shape)
#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()
    
#         progress_bar_open.set_postfix(total_loss=total_loss.item(), batch=batch_idx+1)
    
    all_latent_vectors_close = torch.cat(all_latent_vectors_close, dim=0).cpu().numpy()
    return all_latent_vectors_close, mean_vector_close
        
