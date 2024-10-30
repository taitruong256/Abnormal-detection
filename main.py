import os 
import torch
import argparse

from lib.Data.load_data import load_and_concatenate_datasets, download_kaggle_dataset, split_data
from lib.Model.model import VAE_Unet
from lib.Training.train import train_vae
from lib.Openset.meta_recognition import build_weibull_model
from lib.Training.evaluate import evaluate_vae, evaluate_weibull
from lib.Utils.visualization import plot_metrics, reduce_and_visualize_latent_space, calculate_and_plot_outlier_probabilities, evaluate_and_plot_samples

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate VAE model with open set recognition")
    parser.add_argument('--is_train', type=bool, default=True, help="If true, train the model, else evaluate only.")
    parser.add_argument('--latent_dim', type=int, default=640, help="Dimension of latent space.")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--beta', type=float, default=0.1, help="Beta for KL divergence loss.")
    parser.add_argument('--lambda_', type=float, default=0.01, help="Lambda for centralization/decentralization loss.")
    parser.add_argument('--learning_rate', type=float, default=0.005, help="Learning rate for the optimizer.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training.")
    parser.add_argument('--variance', type=float, default=0.25, help="Variance for the VAE.")
    parser.add_argument('--num_classes', type=int, default=1, help="Number of classes for open set evaluation.")
    parser.add_argument('--input_shape', type=int, default=32, help="Input image shape.")
    parser.add_argument('--num_examples', type=int, default=100, help="Number of examples to load from dataset.")
    parser.add_argument('--tail_size', type=float, default=0.05, help="Tail size for Weibull model.")
    parser.add_argument("--loss_type", type=str, default="recon", choices=["recon", "recon_kld", "recon_kld_decentral"], help="Type of loss to use during training.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    IS_TRAIN = args.is_train
    LATENT_DIM = args.latent_dim
    NUM_EPOCHS = args.num_epochs
    BETA = args.beta
    LAMBDA_ = args.lambda_
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    VARIANCE = args.variance
    NUM_CLASSES = args.num_classes
    INPUT_SHAPE = args.input_shape
    NUM_EXAMPLES = args.num_examples
    TAIL_SIZE = args.tail_size
    LOSS_TYPE = args.loss_type 
    
    dataset_names = ["taitruong270/cancer1", "taitruong270/cancer2", "taitruong270/cancer3", 
                 "taitruong270/cancer4", "taitruong270/cancer5", "taitruong270/cancer6"]
    dataset_paths = [download_kaggle_dataset(name) for name in dataset_names]
    df = load_and_concatenate_datasets(dataset_paths)
    df = df.head(NUM_EXAMPLES)
    print("Total records in combined dataset:", len(df))
    
    train_loader, train_open_loader, train_close_loader, val_loader, val_open_loader, val_close_loader = split_data(df, BATCH_SIZE, INPUT_SHAPE)

    print('Training set size (non-cancer): ', len(train_close_loader))
    print('Training set size (cancer): ', len(train_open_loader))
    print('Validation set size (non-cancer): ', len(val_close_loader))
    print('Validation set size (cancer): ', len(val_open_loader))
    
    if torch.cuda.is_available():
        print("Model is running on GPU")
    else:
        print("Model is running on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE_Unet(INPUT_SHAPE, VARIANCE, LATENT_DIM).to(device) 
    optimizer = torch.optim.SGD(vae.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
    best_f1_open = 0.0
    train_history = []
    val_history = []
    best_model_path = os.path.join(os.getcwd(), 'lib', 'Data', 'best_model.pth')

    if IS_TRAIN:
        for epoch in range(NUM_EPOCHS):
            print(f'Starting training epoch {epoch + 1} / {NUM_EPOCHS}.')
            
            all_latent_vectors, mean_vector = train_vae(vae, device, train_close_loader, train_open_loader, optimizer, BETA, LAMBDA_, LATENT_DIM, INPUT_SHAPE, LOSS_TYPE)
            
            val_losses = evaluate_vae(vae, device, val_close_loader, val_open_loader, LATENT_DIM, BETA, LAMBDA_, INPUT_SHAPE, LOSS_TYPE)
            val_openset_metrics = evaluate_weibull(vae, device, all_latent_vectors, mean_vector, val_loader, NUM_CLASSES, LATENT_DIM, TAIL_SIZE)
            
            val_history.append(val_losses + val_openset_metrics)
            
            print(f'Val  Total Loss: {val_losses[0]:.4f}, Reconstruction Loss: {val_losses[1]:.4f}, KL Divergence: {val_losses[2]:.4f}, Decentralization Loss (Cancer): {val_losses[3]:.4f}')
            print(f'Val  Openset Accuracy: {val_openset_metrics[0]:.4f}, F1 Open: {val_openset_metrics[1]:.4f}')
            print(f'Confusion Matrix:\n{val_openset_metrics[2]}\n')
            
            current_f1_open = val_openset_metrics[1]
            if current_f1_open >= best_f1_open:
                best_f1_open = current_f1_open
                torch.save(vae.state_dict(), best_model_path)  # Lưu mô hình tốt nhất
                print(f'Saved new best model with F1 Open: {best_f1_open:.4f}')
                
        print("Training finished!")
        
        # Visualize result 
        output_dir = os.path.join('lib', 'Utils', 'result')
        os.makedirs(output_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
        plot_metrics(val_history, NUM_EPOCHS, output_dir)
        
    # Load best model         
    print("Loading best model")
    vae.load_state_dict(torch.load(best_model_path, weights_only=True, map_location=device))
    vae = vae.to(device)

    all_latent_vectors, mean_vector = train_vae(vae, device, train_close_loader, train_open_loader, optimizer, BETA, LAMBDA_, LATENT_DIM, INPUT_SHAPE, LOSS_TYPE)
    accuracy, best_f1, confusion, best_threshold = evaluate_weibull(vae, device, all_latent_vectors, mean_vector, val_loader, NUM_CLASSES, LATENT_DIM, TAIL_SIZE)
    OMEGA_T = best_threshold
    print(f"Best Accuracy: {accuracy}")
    print(f"Best F1 Score: {best_f1}")
    print(f"Best Threshold: {best_threshold}")
    print(f"Confusion Matrix:\n{confusion}")
        
    weibull_model = build_weibull_model(mean_vector, all_latent_vectors, NUM_CLASSES, TAIL_SIZE)
    reduce_and_visualize_latent_space(val_loader, vae, INPUT_SHAPE, device, output_dir)
    calculate_and_plot_outlier_probabilities(val_loader, vae, mean_vector, weibull_model, NUM_CLASSES, device, output_dir)
    evaluate_and_plot_samples(df, vae, mean_vector, weibull_model, NUM_CLASSES, OMEGA_T, device, output_dir, INPUT_SHAPE)
        
    print("Running finished.")
