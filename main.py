import os 
import torch
import argparse
import time
import shutil
import json
import random

from lib.Utils.utils import setup_logging, MetricsLogger, save_checkpoint, save_task_checkpoint
from lib.Data import datasets
from lib.Model import architectures
from lib.Model.architectures import grow_classifier
from lib.Model.initialization import WeightInit
from lib.Training.loss_functions import joint_loss_function as criterion
from lib.Training.train import train
from lib.Training.validate import validate
from lib.Utility.visualization import visualize_all_training_results, plot_training_metrics
import json
import time 

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate VAE model with open set recognition")
    parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size. Default: 16')
    parser.add_argument('--learning-rate', default=0.001, type=float, help='initial learning rate. Default: 0.001')
    parser.add_argument('--dataset', type=str, default='BloodMNIST', help="Dataset to use for training and evaluation.")
    parser.add_argument('--dataroot', type=str, default='./data', help='Data root directory. Default: ./data')
    parser.add_argument('--gray-scale', default=False, type=bool, help='use gray scale images. Default: False. If false, single channel images will be repeated to three channels.')
    parser.add_argument('-p', '--patch-size', default=28, type=int, help='patch size for crops. Default: 28')
    parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers. Default: 4')
    parser.add_argument('-a', '--architecture', default='WRN', help='model architecture. Options: WRN, MLP, HRNetEncoder. Default: WRN')
    parser.add_argument('--encoder-variant', default='hrnet_w18', type=str, 
                       help='Encoder variant (only for HRNetEncoder). Options: hrnet_w18, hrnet_w32, hrnet_w48, hrnet_w64. Default: hrnet_w18')
    parser.add_argument('--wrn-widen-factor', default=10, type=int, help='width factor of the wide residual network. Default: 10')
    parser.add_argument('--wrn-depth', default=14, type=int, help='amount of layers in the wide residual network. Default: 14')
    parser.add_argument('-bn', '--batch-norm', default=1e-5, type=float, help='batch normalization. Default 1e-5')
    parser.add_argument('--out-channels', default=3, type=int, help='number of output channels of decoder. Should match input channels. Default: 3')
    parser.add_argument('--double-wrn-blocks', type=bool, help='If turned on, uses 6 instead of 3 blocks and downsamples 6 times by factor 2. Should be used for high resolution data, like flowers')
    parser.add_argument('--var-samples', default=1, type=int, help='number of samples for the expectation in variational training. Default: 1')
    parser.add_argument('--var-latent-dim', default=60, type=int, help='Dimensionality of latent space. Default 60')
    parser.add_argument('--wrn-embedding-size', type=int, default=48, help='number of output channels in the first wrn layer if widen factor is not being')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run. Default: 10')
    parser.add_argument('--var-beta', default=0.1, type=float, help='weight term for KLD loss. Default: 0.1')
    parser.add_argument('-pf', '--print-freq', default=0.2, type=float, help='print frequency. If int (>=1): log every N steps. If float (0-1): log at fraction of total steps per epoch. Default: 100')
    parser.add_argument('--visualization-epoch', default=5, type=int, help='number of epochs after which generations/reconstructions are visualized/saved. Default: 20')
    parser.add_argument('--autoregression', default=False, type=bool, help='use autoregression. Default: False')
    parser.add_argument('--max-samples', default=None, type=int, help='Limit dataset to first N samples for quick testing (train=N, val=N/4). Default: None (use all data)')
    
    # Continual learning arguments
    parser.add_argument('--incremental-data', default=False, type=bool, help='Convert dataloaders to class incremental ones. Default: False')
    parser.add_argument('--num-base-tasks', default=1, type=int, help='Number of tasks to start with for incremental learning. Default: 1')
    parser.add_argument('--num-increment-tasks', default=2, type=int, help='Number of tasks to add at once. Default: 2')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create save path first (before logging setup)
    save_path = 'runs/' + time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime()) + '_' + args.dataset + '_' + args.architecture + '_variational_samples_' + str(args.var_samples) + '_latent_dim_' + str(args.var_latent_dim)
    os.makedirs(save_path, exist_ok=True)
    
    # Setup logging to runs directory
    logger, log_file = setup_logging(save_path)
    logger.info("="*80)
    logger.info("Starting VAE Model Training and Evaluation")
    logger.info("="*80)
    
    logger.info(f"Log file saved to: {log_file}")
    logger.info(f"Arguments: {args}")
    
    # Log configuration
    logger.info("Model Configuration:")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Learning Rate: {args.learning_rate}")   
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  GPU Available: {torch.cuda.is_available()}") 
    logger.info(f"  Gray Scale: {args.gray_scale}")
    logger.info(f"  Patch Size: {args.patch_size}")
    logger.info(f"  Architecture: {args.architecture}")
    logger.info(f"  WRN Widen Factor: {args.wrn_widen_factor}")
    logger.info(f"  WRN Depth: {args.wrn_depth}")
    logger.info(f"  Batch Norm: {args.batch_norm}")
    logger.info(f"  Out Channels: {args.out_channels}")
    logger.info(f"  Double WRN Blocks: {args.double_wrn_blocks}")
    logger.info(f"  Variational Samples: {args.var_samples}")
    logger.info(f"  Variational Latent Dim: {args.var_latent_dim}")
    logger.info(f"  WRN Embedding Size: {args.wrn_embedding_size}")
    logger.info(f"  Total Epochs: {args.epochs}")
    logger.info(f"  Var Beta: {args.var_beta}")
    if args.max_samples is not None:
        logger.info(f"  Max Samples (Testing Mode): Train={args.max_samples}, Val={args.max_samples // 4}")
    
    # Setup known classes for MedMNIST datasets
    medmnist_datasets = ['bloodmnist', 'octmnist', 'dermamnist', 'tissuemnist']
    if args.dataset.lower() in medmnist_datasets:
        logger.info(f"\nLoading MedMNIST dataset: {args.dataset}")
        
        # For continual learning, start with only num_base_tasks classes
        if args.incremental_data:
            # Start with first num_base_tasks classes (0, 1, 2, ..., num_base_tasks-1)
            args.known = list(range(args.num_base_tasks))
            logger.info(f"  Continual Learning Mode: Starting with {args.num_base_tasks} classes: {args.known}")
        else:
            # Normal training: use predefined splits
            if args.dataset.lower() == 'bloodmnist':
                args.known = [0, 1, 2, 3, 4]
            elif args.dataset.lower() == 'octmnist':
                args.known = [0, 1, 2]
            elif args.dataset.lower() == 'dermamnist':
                args.known = [0, 1, 2, 3]
            elif args.dataset.lower() == 'tissuemnist':
                args.known = [0, 1, 2, 3, 4]
            logger.info(f"  Known classes: {args.known}")
        
        dataset = datasets.get_dataset(torch.cuda.is_available(), args)
    else:
        data_init_method = getattr(datasets, args.dataset)
        dataset = data_init_method(torch.cuda.is_available(), args)

    # import model from architectures class
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_colors = 1 if args.gray_scale else 3
    
    # For continual learning, start with base tasks only
    if args.incremental_data:
        num_classes = args.num_base_tasks
        logger.info(f"Continual learning mode: Starting with {num_classes} classes")
    else:
        num_classes = dataset.num_classes
    
    net_init_method = getattr(architectures, args.architecture)
    
    # Validate encoder-variant for HRNet
    if args.architecture == 'HRNetEncoder':
        valid_variants = ['hrnet_w18', 'hrnet_w32', 'hrnet_w48']
        if args.encoder_variant not in valid_variants:
            logger.warning(f"\n{'='*80}")
            logger.warning(f"WARNING: Invalid encoder-variant '{args.encoder_variant}' for HRNetEncoder")
            logger.warning(f"Valid options: {', '.join(valid_variants)}")
            logger.warning(f"Example: --architecture HRNetEncoder --encoder-variant hrnet_w18")
            logger.warning(f"{'='*80}\n")
        else:
            logger.info(f"Using HRNetEncoder with variant: {args.encoder_variant}")
    
    # build the model
    model = net_init_method(device, num_classes, num_colors, args).to(device)
    # print model summary
    logger.info(model)
    # print number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {num_params}")

    train_criterion = criterion
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    
    # Weight initializer for growing classifier
    weight_initializer = WeightInit('kaiming-normal')
    
    # Initialize metrics logger
    metrics_logger = MetricsLogger(log_file)
    
    epoch = 0
    best_prec = 0
    best_loss = random.getrandbits(128)
    
    # Epoch multiplier for incremental learning
    epoch_multiplier = 1
    if args.incremental_data:
        # Calculate total epochs based on number of incremental tasks
        # Use total classes in dataset (not current num_classes)
        if hasattr(dataset, 'n_classes'):
            num_total_classes = dataset.n_classes  # Total classes in MedMNIST dataset
        else:
            num_total_classes = 8  # Default for BloodMNIST
        
        num_tasks = ((num_total_classes - args.num_base_tasks) // args.num_increment_tasks) + 1
        epoch_multiplier = num_tasks
        logger.info(f"Incremental learning: {num_tasks} tasks total")
        logger.info(f"Total classes in dataset: {num_total_classes}")
        logger.info(f"Starting with {args.num_base_tasks} classes, incrementing by {args.num_increment_tasks} classes per task")
    
    # Load checkpoint if resuming
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            epoch = checkpoint['epoch']
            best_prec = checkpoint.get('best_prec', 0)
            best_loss = checkpoint.get('best_loss', random.getrandbits(128))
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info(f"=> Loaded checkpoint (epoch {epoch})")
        else:
            logger.warning(f"=> No checkpoint found at '{args.resume}'")

    # optimize until final amount of epochs is reached
    while epoch < (args.epochs * epoch_multiplier):
        # Continual learning: increment tasks at the end of each task period
        if args.incremental_data:
            if epoch % args.epochs == 0 and epoch > 0:
                logger.info(f"\n{'='*80}")
                logger.info(f"Incrementing tasks at epoch {epoch}")
                logger.info(f"{'='*80}\n")
                
                # Update known classes to include new tasks
                new_known_classes = list(range(len(args.known) + args.num_increment_tasks))
                logger.info(f"Updating known classes from {args.known} to {new_known_classes}")
                args.known = new_known_classes
                
                # Reload dataset with updated known classes
                logger.info("Reloading dataset with new classes...")
                dataset = datasets.get_dataset(torch.cuda.is_available(), args)
                
                # Grow the classifier
                model.num_classes += args.num_increment_tasks
                grow_classifier(device, model.classifier, args.num_increment_tasks, weight_initializer)
                
                # Reset optimizer for new parameters
                optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
                
                # Reset best metrics for new task
                best_prec = 0
                best_loss = random.getrandbits(128)
                
                logger.info(f"Classifier grown to {model.num_classes} classes")
                logger.info(f"Optimizer reset\n")

        train(dataset, model, train_criterion, epoch, optimizer, metrics_logger, device, args)

        # evaluate on validation set
        prec, loss = validate(dataset, model, criterion, epoch, metrics_logger, device, save_path, args)

        # remember best prec@1 and save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        best_prec = max(prec, best_prec)
        
        save_checkpoint({'epoch': epoch + 1,
                            'arch': args.architecture,
                            'state_dict': model.state_dict(),
                            'best_prec': best_prec,
                            'best_loss': best_loss,
                            'optimizer': optimizer.state_dict()},
                        is_best, save_path)
        
        # Save task checkpoint at end of each task period
        if args.incremental_data and (epoch + 1) % args.epochs == 0:
            
            task_num = (epoch + 1) // args.epochs
            num_classes = model.num_classes
            
            # Save task checkpoint
            save_task_checkpoint(save_path, task_num)
            
            # Rename checkpoint to include class count
            checkpoint_src = os.path.join(save_path, f'task_{task_num}_checkpoint.pth.tar')
            checkpoint_dst = os.path.join(save_path, f'task_{task_num}_{num_classes}classes_checkpoint.pth.tar')
            if os.path.exists(checkpoint_src):
                shutil.copy2(checkpoint_src, checkpoint_dst)
            
            # Save task-specific metrics
            metrics_file = log_file.replace('.log', '_metrics.json')
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        all_metrics = json.load(f)
                    
                    # Extract metrics for this task
                    start_epoch = (task_num - 1) * args.epochs
                    end_epoch = task_num * args.epochs
                    task_metrics = {}
                    for key in all_metrics:
                        if isinstance(all_metrics[key], list):
                            task_metrics[key] = all_metrics[key][start_epoch:end_epoch]
                    
                    # Save task metrics with task info in filename
                    task_metrics_file = os.path.join(save_path, f'task_{task_num}_{num_classes}classes_metrics.json')
                    with open(task_metrics_file, 'w') as f:
                        json.dump(task_metrics, f, indent=2)
                    
                    logger.info(f"Task {task_num} ({num_classes} classes) saved:")
                    logger.info(f"  - Checkpoint: {checkpoint_dst}")
                    logger.info(f"  - Metrics: {task_metrics_file}\n")
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    logger.warning(f"Could not save task metrics: {e}")
                    logger.info(f"Task {task_num} checkpoint saved\n")
            else:
                logger.info(f"Task {task_num} checkpoint saved\n")

        # increment epoch counters
        epoch += 1
    
    # Training completed - visualize all results
    logger.info("\n" + "="*80)
    logger.info("Training Completed!")
    logger.info("="*80)
    logger.info(f"Best Precision: {best_prec:.2f}%")
    logger.info(f"Best Loss: {best_loss:.5f}")
    logger.info(f"Results saved in: {save_path}")
    
    # Plot training metrics
    try:
        plot_training_metrics(save_path)
    except Exception as e:
        logger.warning(f"Could not plot training metrics: {e}")
    
    # Visualize all training results
    try:
        visualize_all_training_results(save_path, max_cols=3)
    except Exception as e:
        logger.warning(f"Could not display visualization: {e}")
        logger.info(f"You can manually visualize results later if needed")
