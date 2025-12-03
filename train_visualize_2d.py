"""
Script train VAE với latent_dim=2 và visualize sử dụng CÁC HÀM CÓ SẴN.

Sử dụng:
    python train_visualize_2d.py --dataset bloodmnist --epochs 50

Các hàm có sẵn:
    - visualize_dataset_in_2d_embedding: Visualize 2D latent space (lib/Utility/visualization.py)
    - visualize_image_grid: Visualize images (lib/Utility/visualization.py)
    - train: Training loop (lib/Training/train.py)
    - validate: Validation loop (lib/Training/validate.py)
"""

import os
import torch
import argparse
import time
import random
import logging

from lib.Utils.utils import setup_logging, MetricsLogger, save_checkpoint
from lib.Data import datasets
from lib.Model import architectures
from lib.Training.loss_functions import joint_loss_function as criterion
from lib.Training.train import train
from lib.Training.validate import validate
from lib.Utility.visualization import visualize_dataset_in_2d_embedding, visualize_image_grid
from torch.utils.data import Subset


def collect_latent_codes_by_class(model, dataset, device):
    """
    Thu thập latent codes (mu) theo từng class.
    
    Returns:
        encoding_list: List of tensors cho visualize_dataset_in_2d_embedding
    """
    model.eval()
    logger = logging.getLogger()
    
    # Dictionary to store encodings per class
    class_encodings = {}
    
    with torch.no_grad():
        for images, labels in dataset.train_loader:
            images = images.to(device)
            labels_cpu = labels.cpu().numpy()
            
            # Encode to get latent mu
            z_mean, _ = model.encode(images)
            z_mean_cpu = z_mean.cpu()
            
            # Group by class
            for i, label in enumerate(labels_cpu):
                label = int(label)
                if label not in class_encodings:
                    class_encodings[label] = []
                class_encodings[label].append(z_mean_cpu[i])
    
    # Convert to list of tensors - ensure all tensors have same batch dimension
    # encoding_list[i] should be tensor of shape [N_i, 2] where N_i is number of samples for class i
    encoding_list = []
    class_ids = sorted(class_encodings.keys())
    
    for class_id in class_ids:
        if len(class_encodings[class_id]) > 0:
            class_tensor = torch.stack(class_encodings[class_id])
            encoding_list.append(class_tensor)
        else:
            # Add empty tensor with shape [0, 2] to maintain consistency
            encoding_list.append(torch.empty(0, 2))
    
    logger.info(f"Collected latent codes for {len(encoding_list)} classes (IDs: {class_ids})")
    logger.info(f"Shapes: {[enc.shape for enc in encoding_list]}")
    return encoding_list


def visualize_latent_space(model, dataset, device, save_path, epoch, dataset_name):
    """
    Sử dụng hàm có sẵn visualize_dataset_in_2d_embedding để visualize 2D latent space.
    """
    logger = logging.getLogger()
    
    # Check if latent dim is 2D
    if model.latent_dim != 2:
        logger.warning(f"Latent dim is {model.latent_dim}, not 2. Skipping 2D visualization.")
        return
    
    logger.info(f"Creating 2D latent space visualization (Epoch {epoch})...")
    
    # Collect latent codes by class
    encoding_list = collect_latent_codes_by_class(model, dataset, device)
    
    # Filter out empty tensors before calling visualization
    encoding_list_filtered = [enc for enc in encoding_list if enc.size(0) > 0]
    
    if len(encoding_list_filtered) == 0:
        logger.warning("No latent codes collected. Skipping visualization.")
        return
    
    # Use built-in function (writer can be None)
    visualize_dataset_in_2d_embedding(
        writer=None,
        encoding_list=encoding_list_filtered,
        dataset_name=dataset_name,
        save_path=save_path,
        task=epoch
    )
    
    logger.info(f"✓ 2D visualization saved")


def generate_and_visualize_samples(model, device, save_path, epoch):
    """
    Generate samples và sử dụng hàm có sẵn visualize_image_grid.
    """
    logger = logging.getLogger()
    logger.info(f"Generating samples (Epoch {epoch})...")
    
    model.eval()
    with torch.no_grad():
        # Use model's built-in generate method
        generated = model.generate()
        generated = torch.sigmoid(generated)
        
        # Use built-in visualization function
        visualize_image_grid(
            images=generated,
            writer=None,
            count=epoch,
            name='generated_samples',
            save_path=save_path
        )
    
    logger.info(f"✓ Generated samples saved")


def visualize_reconstructions(model, dataset, device, save_path, epoch):
    """
    Visualize reconstructions sử dụng hàm có sẵn visualize_image_grid.
    """
    logger = logging.getLogger()
    logger.info(f"Creating reconstruction visualization (Epoch {epoch})...")
    
    model.eval()
    with torch.no_grad():
        # Get one batch from validation set
        images, _ = next(iter(dataset.val_loader))
        images = images.to(device)
        
        # Get reconstructions
        _, recon_samples, _, _ = model(images)
        recon = torch.mean(recon_samples, dim=0)
        recon = torch.sigmoid(recon)
        
        # Combine original and reconstructed
        comparison = torch.cat([images[:16], recon[:16]])
        
        # Use built-in visualization function
        visualize_image_grid(
            images=comparison,
            writer=None,
            count=epoch,
            name='reconstructions',
            save_path=save_path
        )
    
    logger.info(f"✓ Reconstruction visualization saved")


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE with 2D latent visualization")
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='bloodmnist',
                       choices=['bloodmnist', 'octmnist', 'dermamnist', 'tissuemnist'],
                       help='Dataset name (default: bloodmnist)')
    parser.add_argument('--dataroot', type=str, default='./data',
                       help='Data root directory (default: ./data)')
    
    # Training arguments
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                       help='Batch size (default: 128)')
    parser.add_argument('--learning-rate', default=0.001, type=float,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--epochs', default=50, type=int,
                       help='Number of epochs (default: 50)')
    parser.add_argument('--var-beta', default=0.1, type=float,
                       help='KLD weight beta (default: 0.1)')
    
    # Model arguments - FIXED to 2 for 2D visualization
    parser.add_argument('--var-latent-dim', default=2, type=int,
                       help='Latent dimension - MUST be 2 for 2D visualization')
    parser.add_argument('-a', '--architecture', default='WRN',
                       help='Model architecture (default: WRN)')
    parser.add_argument('--wrn-widen-factor', default=10, type=int,
                       help='WRN width factor (default: 10)')
    parser.add_argument('--wrn-depth', default=14, type=int,
                       help='WRN depth (default: 14)')
    parser.add_argument('--wrn-embedding-size', default=48, type=int,
                       help='WRN embedding size (default: 48)')
    
    # Visualization arguments
    parser.add_argument('--visualize-freq', default=5, type=int,
                       help='Visualize every N epochs (default: 5)')
    parser.add_argument('--visualization-epoch', default=5, type=int,
                       help='Visualization epoch for validate function (default: 5)')
    
    # Testing/Debug arguments
    parser.add_argument('--max-samples', default=None, type=int,
                       help='Limit dataset to first N samples for quick testing (default: None = use all data)')
    
    # Other arguments
    parser.add_argument('-j', '--workers', default=4, type=int,
                       help='Data loading workers (default: 4)')
    parser.add_argument('-p', '--patch-size', default=28, type=int,
                       help='Patch size (default: 28)')
    parser.add_argument('--gray-scale', default=False, type=bool,
                       help='Use grayscale (default: False)')
    parser.add_argument('-bn', '--batch-norm', default=1e-5, type=float,
                       help='Batch normalization (default: 1e-5)')
    parser.add_argument('--out-channels', default=3, type=int,
                       help='Output channels (default: 3)')
    parser.add_argument('--var-samples', default=1, type=int,
                       help='Variational samples (default: 1)')
    parser.add_argument('--double-wrn-blocks', default=False, type=bool,
                       help='Double WRN blocks (default: False)')
    parser.add_argument('-pf', '--print-freq', default=50, type=int,
                       help='Print frequency (default: 50)')
    parser.add_argument('--max-train-samples', default=None, type=int,
                       help='Max training samples (default: None)')
    parser.add_argument('--max-test-samples', default=None, type=int,
                       help='Max test samples (default: None)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Force latent_dim to 2
    if args.var_latent_dim != 2:
        print(f"⚠️  Warning: Setting latent_dim to 2 (was {args.var_latent_dim})")
        args.var_latent_dim = 2
    
    # Create save directory
    save_path = 'runs_2d/' + time.strftime("%Y-%m-%d_%H-%M-%S") + \
                '_' + args.dataset + '_latent2d'
    os.makedirs(save_path, exist_ok=True)
    
    # Setup logging
    logger, log_file = setup_logging(save_path)
    logger.info("="*80)
    logger.info("Train VAE with 2D Latent Space - Using Built-in Functions")
    logger.info("="*80)
    logger.info(f"Save path: {save_path}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Latent Dimension: {args.var_latent_dim}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"Beta (KLD weight): {args.var_beta}")
    logger.info(f"Visualization Frequency: Every {args.visualize_freq} epochs")
    if args.max_samples is not None:
        logger.info(f"Max Samples (Testing Mode): {args.max_samples}")
    
    # Setup dataset - set known classes based on dataset
    if args.dataset == 'bloodmnist':
        args.known = [0, 1, 2, 3, 4]
    elif args.dataset == 'octmnist':
        args.known = [0, 1]
    elif args.dataset == 'dermamnist':
        args.known = [0, 1, 2, 3]
    elif args.dataset == 'tissuemnist':
        args.known = [0, 1, 2, 3, 4]
    else:
        args.known = [0, 1, 2, 3, 4]
    
    # Load dataset
    logger.info("\nLoading dataset...")
    # Use the factory function from datasets module
    dataset = datasets.get_dataset(torch.cuda.is_available(), args)
    
    # Limit dataset if max_samples is specified (for quick testing)
    if args.max_samples is not None:
        logger.info(f"⚠️  Limiting dataset to first {args.max_samples} samples for testing")
        
        # Limit training set
        original_train_size = len(dataset.trainset)
        train_indices = list(range(min(args.max_samples, original_train_size)))
        dataset.trainset = Subset(dataset.trainset, train_indices)
        
        # Limit validation set (use smaller portion)
        val_samples = min(args.max_samples // 4, len(dataset.valset))
        val_indices = list(range(val_samples))
        dataset.valset = Subset(dataset.valset, val_indices)
        
        # Recreate data loaders with limited datasets
        dataset.train_loader = torch.utils.data.DataLoader(
            dataset.trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available()
        )
        
        dataset.val_loader = torch.utils.data.DataLoader(
            dataset.valset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"  Train: {original_train_size} -> {len(dataset.trainset)} samples")
        logger.info(f"  Val: {len(dataset.valset)} samples")
    
    logger.info(f"✓ Dataset loaded: {dataset.num_classes} classes")
    
    # Build model
    logger.info("\nBuilding model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_colors = 1 if args.gray_scale else 3
    num_classes = dataset.num_classes
    
    net_init_method = getattr(architectures, args.architecture)
    model = net_init_method(device, num_classes, num_colors, args).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✓ Model: {args.architecture}")
    logger.info(f"✓ Parameters: {num_params:,}")
    
    # Setup training
    train_criterion = criterion
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    metrics_logger = MetricsLogger(log_file)
    
    best_loss = float('inf')
    best_prec = 0
    
    # Training loop
    logger.info("\n" + "="*80)
    logger.info("Starting Training")
    logger.info("="*80)
    
    for epoch in range(args.epochs):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch [{epoch+1}/{args.epochs}]")
        logger.info(f"{'='*80}")
        
        # Train - sử dụng hàm có sẵn
        train(dataset, model, train_criterion, epoch, optimizer, metrics_logger, device, args)
        
        # Validate - sử dụng hàm có sẵn (đã có visualize_image_grid bên trong)
        prec, loss = validate(dataset, model, criterion, epoch, metrics_logger, device, save_path, args)
        
        # Save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        best_prec = max(prec, best_prec)
        
        save_checkpoint({
            'epoch': epoch,
            'arch': args.architecture,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict()
        }, is_best, save_path)
        
        # Additional visualizations using built-in functions
        if (epoch + 1) % args.visualize_freq == 0 or (epoch + 1) == args.epochs:
            logger.info(f"\n{'='*60}")
            logger.info(f"Creating Visualizations - Epoch {epoch+1}")
            logger.info(f"{'='*60}")
            
            # 1. Visualize 2D latent space - SỬ DỤNG HÀM CÓ SẴN
            visualize_latent_space(model, dataset, device, save_path, epoch+1, args.dataset)
            
            # 2. Generate samples - SỬ DỤNG HÀM CÓ SẴN
            generate_and_visualize_samples(model, device, save_path, epoch+1)
            
            # 3. Visualize reconstructions - SỬ DỤNG HÀM CÓ SẴN
            visualize_reconstructions(model, dataset, device, save_path, epoch+1)
            
            logger.info(f"{'='*60}\n")
    
    logger.info("\n" + "="*80)
    logger.info("Training Completed!")
    logger.info(f"Best Loss: {best_loss:.4f}")
    logger.info(f"Best Accuracy: {best_prec:.2f}%")
    logger.info(f"Results saved to: {save_path}")
    logger.info("="*80)
    
    print(f"\n✓ Training completed! Results: {save_path}")


if __name__ == "__main__":
    main()
