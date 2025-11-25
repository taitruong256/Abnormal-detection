import os 
import torch
import argparse

from lib.Utils.utils import setup_logging, MetricsLogger, save_checkpoint
from lib.Data import datasets
from lib.Model import architectures
from lib.Training.loss_functions import joint_loss_function as criterion
import random 
from lib.Training.train import train
from lib.Training.validate import validate
import time 

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate VAE model with open set recognition")
    parser.add_argument('-b', '--batch-size', default=16, type=int, help='mini-batch size. Default: 16')
    parser.add_argument('--learning-rate', default=0.001, type=float, help='initial learning rate. Default: 0.001')
    parser.add_argument('--dataset', type=str, default='MNIST', help="Dataset to use for training and evaluation.")
    parser.add_argument('--gray-scale', default=False, type=bool, help='use gray scale images. Default: False. If false, single channel images will be repeated to three channels.')
    parser.add_argument('-p', '--patch-size', default=28, type=int, help='patch size for crops. Default: 28')
    parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers. Default: 4')
    parser.add_argument('-a', '--architecture', default='WRN', help='model architecture. Default: WRN')
    parser.add_argument('--wrn-widen-factor', default=10, type=int, help='width factor of the wide residual network. Default: 10')
    parser.add_argument('--wrn-depth', default=14, type=int, help='amount of layers in the wide residual network. Default: 14')
    parser.add_argument('-bn', '--batch-norm', default=1e-5, type=float, help='batch normalization. Default 1e-5')
    parser.add_argument('--out-channels', default=3, type=int, help='number of output channels of decoder. Should match input channels. Default: 3')
    parser.add_argument('--double-wrn-blocks', type=bool, help='If turned on, uses 6 instead of 3 blocks and downsamples 6 times by factor 2. Should be used for high resolution data, like flowers')
    parser.add_argument('--var-samples', default=1, type=int, help='number of samples for the expectation in variational training. Default: 1')
    parser.add_argument('--var-latent-dim', default=60, type=int, help='Dimensionality of latent space. Default 60')
    parser.add_argument('--wrn-embedding-size', type=int, default=48, help='number of output channels in the first wrn layer if widen factor is not being')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run. Default: 120')
    parser.add_argument('--var-beta', default=0.1, type=float, help='weight term for KLD loss. Default: 0.1')
    parser.add_argument('-pf', '--print-freq', default=100, type=int, help='print frequency. Default: 100')
    parser.add_argument('--max-train-samples', default=450, type=int, help='maximum number of training samples. Default: None (use all samples)')
    parser.add_argument('--max-test-samples', default=50, type=int, help='maximum number of test samples. Default: None (use 20%% of training samples)')
    parser.add_argument('--visualization-epoch', default=20, type=int, help='number of epochs after which generations/reconstructions are visualized/saved. Default: 20')
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
    logger.info(f"  Max Train Samples: {args.max_train_samples if args.max_train_samples else 'All'}")
    logger.info(f"  Max Test Samples: {args.max_test_samples if args.max_test_samples else '20% of train'}")
    
    data_init_method = getattr(datasets, args.dataset)
    dataset = data_init_method(torch.cuda.is_available(), args)

    # import model from architectures class
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_colors = 1 if args.gray_scale else 3
    num_classes = dataset.num_classes
    net_init_method = getattr(architectures, args.architecture)
    # build the model
    model = net_init_method(device, num_classes, num_colors, args)
    # print model summary
    logger.info(model)
    # print number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {num_params}")

    train_criterion = criterion
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    
    # Initialize metrics logger
    metrics_logger = MetricsLogger(log_file)
    
    epoch = 0
    best_prec = 0
    best_loss = random.getrandbits(128)

    # optimize until final amount of epochs is reached. Final amount of epochs is determined through the
    while epoch < (args.epochs):
        # visualize the latent space before each task increment and at the end of training if it is 2-D
        if epoch % args.epochs == 0 and epoch > 0 or (epoch + 1) % (args.epochs) == 0:
            pass

        train(dataset, model, train_criterion, epoch, optimizer, metrics_logger, device, args)

        # evaluate on validation set
        prec, loss = validate(dataset, model, criterion, epoch, metrics_logger, device, save_path, args)

        # remember best prec@1 and save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        best_prec = max(prec, best_prec)
        
        save_checkpoint({'epoch': epoch,
                            'arch': args.architecture,
                            'state_dict': model.state_dict(),
                            'best_prec': best_prec,
                            'best_loss': best_loss,
                            'optimizer': optimizer.state_dict()},
                        is_best, save_path)

        # increment epoch counters
        epoch += 1
