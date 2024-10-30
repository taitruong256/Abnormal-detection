import os
import torch
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from lib.Openset.meta_recognition import calculate_outlier_probability 
from lib.Data.datasets import BreastCancerDataset


def plot_metrics(val_history, NUM_EPOCHS, output_dir, filename='training_metrics_plot.jpg'):
    plt.figure(figsize=(12, 42))
    epochs = range(1, NUM_EPOCHS + 1)
    
    metrics = [
        ('Total Loss', 'Test Total Loss', 0, 'blue'),
        ('Reconstruction Loss', 'Test Reconstruction Loss', 1, 'blue'),
        ('KL Divergence', 'Test KL Divergence', 2, 'blue'),
        ('Decentralization Loss (Cancer)', 'Test Decentralization Loss (Cancer)', 3, 'red'),
        ('Open-set Recognition Accuracy', 'Test Open-set Recognition Accuracy', 4, 'green'),
        ('F1 Score', 'Test F1 Score', 5, 'green')
    ]
    
    for i, (title, label, idx, color) in enumerate(metrics):
        plt.subplot(6, 1, i+1)
        plt.plot(epochs, [val_results[idx] for val_results in val_history], label=label, color=color)
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(title)
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)  # Lưu biểu đồ thành file 
    print(f'Metrics plot saved at: {output_path}')
    plt.close()  
    
    
def reduce_and_visualize_latent_space(val_loader, vae, INPUT_SHAPE, device, output_dir):
    labels = []
    latent_vectors = []
    
    with torch.no_grad():
        progress_bar_close = tqdm(val_loader, desc="Visualizing latent space", leave=False)
        for batch_idx, (x_batch, y_batch) in enumerate(progress_bar_close):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            _, _, z_batch, x_recon = vae(x_batch)
            labels.append(y_batch.cpu().detach().numpy())
            latent_vectors.append(z_batch.cpu().detach().numpy())
            progress_bar_close.set_postfix(batch=batch_idx+1)
    
    labels = np.concatenate(labels, axis=0)
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    
    # Giảm chiều vector ẩn từ LATENT_DIM xuống 2D với t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    latent_vectors_2d = tsne.fit_transform(latent_vectors)
    plot_latent_space(latent_vectors_2d, labels, output_dir)
    
    
def plot_latent_space(latent_vectors_2d, labels, output_dir, filename='latent_space_plot.jpg'):

    plt.figure(figsize=(8, 24))
    colors = plt.cm.tab10(np.linspace(0, 1, 11))

    plt.subplot(3, 1, 1)
    plt.scatter(latent_vectors_2d[labels == 0, 0], latent_vectors_2d[labels == 0, 1], color=colors[0], label=f'Closed set', marker='o')  # Use circle marker
    plt.title(f'Latent Representation for closed set')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.scatter(latent_vectors_2d[labels == 1, 0], latent_vectors_2d[labels == 1, 1], color='black', label=f'Open Set', marker='x')  # Use "x" marker
    plt.title(f'Latent Representation for Opened set')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.scatter(latent_vectors_2d[labels == 0, 0], latent_vectors_2d[labels == 0, 1], color=colors[0], label=f'Closed set', marker='o')  # Use circle marker
    plt.scatter(latent_vectors_2d[labels == 1, 0], latent_vectors_2d[labels == 1, 1], color='black', label=f'Open Set', marker='x')  # Use "x" marker
    plt.title(f'Latent Representation for closed set and opened set')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Legend', labels=['Close set', 'Open Set'])
    plt.grid(True)

    plt.tight_layout()

    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)  # Lưu biểu đồ thành file 
    print(f'Latent space plot saved at: {output_path}')
    plt.close()  
    
    
def calculate_and_plot_outlier_probabilities(val_loader, vae, mean_vector, weibull_model, NUM_CLASSES, device, output_dir):
    closed_set_probabilities = []
    open_set_probabilities = []

    with torch.no_grad():
        progress_bar_close = tqdm(val_loader, desc="Calculating Outlier Probabilities", leave=False)
        for batch_idx, (x_batch, y_batch) in enumerate(progress_bar_close):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            _, _, z_batch, _ = vae(x_batch)
            
            for z, y in zip(z_batch, y_batch):
                z = z.cpu().detach().numpy()
                outlier_probability = calculate_outlier_probability(z, mean_vector, weibull_model, NUM_CLASSES)
                min_probability = min(outlier_probability)
                if y.item() < NUM_CLASSES:
                    closed_set_probabilities.append(min_probability)
                else:
                    open_set_probabilities.append(min_probability)
            progress_bar_close.set_postfix(batch=batch_idx+1)
    
    plot_outlier_probability_histogram(closed_set_probabilities, open_set_probabilities, output_dir)


def plot_outlier_probability_histogram(closed_set_probabilities, open_set_probabilities, output_dir, filename='outlier_probability_histogram_plot.jpg'):
    plt.figure(figsize=(15, 10))
    
    plt.hist(closed_set_probabilities, bins=100, range=(0, 1), color='green', alpha=0.5, label='Closed-set')
    plt.hist(open_set_probabilities, bins=100, range=(0, 1), color='red', alpha=0.5, label='Open-set')
    
    plt.title('Outlier Probability Histogram')
    plt.xlabel('Outlier Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    print(f'Outlier probability histogram saved at: {output_path}')
    plt.close()
    
    
def evaluate_and_plot_samples(df, vae, mean_vector, weibull_model, NUM_CLASSES, OMEGA_T, device, output_dir, input_shape):
    # Lấy mẫu dữ liệu: 10 ảnh không bị ung thư và 10 ảnh bị ung thư
    non_cancer_sample = df[df.cancer == 0].sample(min(10, len(df[df.cancer == 0])), replace=False)
    cancer_sample = df[df.cancer == 1].sample(min(10, len(df[df.cancer == 1])), replace=False)

    sample_df = pd.concat([non_cancer_sample, cancer_sample])

    sample_dataset = BreastCancerDataset(sample_df, input_shape)
    sample_loader = DataLoader(sample_dataset, batch_size=16, shuffle=False)

    with torch.no_grad():
        for idx, (x_batch, y_batch) in enumerate(sample_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            _, _, z_batch, reconstructed_img = vae(x_batch)
            plot_image_comparison(x_batch, y_batch, z_batch, reconstructed_img, mean_vector, weibull_model, NUM_CLASSES, OMEGA_T, output_dir)



def plot_image_comparison(x_batch, y_batch, z_batch, reconstructed_img, mean_vector, weibull_model, NUM_CLASSES, OMEGA_T, output_dir):
    index = 1 
    for x, y, z, recon_img in zip(x_batch, y_batch, z_batch, reconstructed_img):
        z = z.cpu().detach().numpy()
        reconstruction_loss = torch.norm(recon_img - x, p=2) ** 2 / (x.size(-1) * x.size(-2))

        # Tính xác suất outlier cho từng ảnh x với mô hình Weibull
        outlier_probability = calculate_outlier_probability(z, mean_vector, weibull_model, NUM_CLASSES)
        min_outlier_prob = min(outlier_probability)

        # Kiểm tra xác suất outlier với ngưỡng OMEGA_T
        if min_outlier_prob <= OMEGA_T:
            if y.item() < NUM_CLASSES:
                marker = '$\u2714$'
                color = 'green'
            else:
                marker = 'x'
                color = 'red'
        else:
            if y.item() >= NUM_CLASSES:
                marker = '$\u2714$'
                color = 'green'
            else:
                marker = 'x'
                color = 'red'

        # Vẽ ảnh gốc, ảnh tái tạo và ảnh khác biệt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].imshow(x[0].cpu().detach().numpy(), cmap='gray')
        axes[0].set_title(f'Original Image. Label: {y.item()}')

        axes[1].imshow(recon_img[0].cpu().detach().numpy(), cmap='gray')
        axes[1].scatter(20, 20, color=color, marker=marker, s=1000)
        axes[1].set_title(f'Reconstructed Image\nReconstruction Loss: {reconstruction_loss.item():.4f}\nMin Outlier Prob: {min_outlier_prob:.4f}')

        difference = np.abs(x[0].cpu().detach().numpy() - recon_img[0].cpu().detach().numpy())
        im = axes[2].imshow(difference, cmap='hot')
        axes[2].set_title(f'Difference Image\nWeibull Scores: {["{:.4f}".format(x) for x in np.round(outlier_probability, 4)]}')

        cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.set_label('Difference Intensity')

        # Lưu ảnh vào output_dir
        save_path = os.path.join(output_dir, f'sample_result_{index}.png')
        plt.savefig(save_path)
        print(f'Image saved at: {save_path}')
        index+=1 

        plt.close()  