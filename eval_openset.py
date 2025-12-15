"""
Stand alone evaluation script for open set recognition and plotting of different datasets

Uses the same command line parser as main.py

The attributes that need to be specified are the number of variational samples (should be greater than one if prediction
uncertainties are supposed to be calculated and compared), the architecture type and the resume flag pointing to a model
checkpoint file.
Other parameters like open set distance function etc. are optional.

Minimum example usage:
--resume /path/checkpoint.pth.tar --var-samples 100 -a MLP
"""

# import collections
import collections
import torch
from collections import Counter
# from lib.cmdparser import parser
import lib.Data.datasets as datasets
import lib.Model.architectures as architectures
# from lib.Models.pixelcnn import PixelCNN
from lib.Training.evaluate import eval_dataset as eval_dataset
from lib.Training.evaluate import eval_openset_dataset as eval_openset_dataset
from lib.Utility.visualization import *
from lib.Openset.meta_recognition import *
import argparse
import torch
import os
import numpy as np
import logging
from datetime import datetime


def setup_logging(save_path):
    """Setup logging to both console and file
    
    Parameters:
        save_path (str): Directory to save log file
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Create log filename with current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_path, f'eval_openset_{timestamp}.log')
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Changed from DEBUG to INFO to reduce log verbosity
    
    # Clear existing handlers
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)  # Changed from DEBUG to INFO
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


def parse_args():
    """Parse command line arguments for open set evaluation"""
    parser = argparse.ArgumentParser(
        description='PyTorch Variational Training - Open Set Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset and loading
    parser.add_argument('--dataset', default='BloodMNIST', help='Name of dataset. Default: MNIST')
    parser.add_argument('-j', '--workers', default=4, type=int, help='Number of data loading workers. Default: 4')
    parser.add_argument('-p', '--patch-size', default=28, type=int, help='Patch size for crops. Default: 28')
    parser.add_argument('--gray-scale', default=False, type=bool, 
                        help='Use gray scale images. Default: False. If false, single channel images will be repeated to three channels.')
    
    # Architecture and weight-init
    parser.add_argument('-a', '--architecture', default='WRN', help='Model architecture. Options: WRN, MLP, HRNetEncoder. Default: WRN')
    parser.add_argument('--encoder-variant', default='hrnet_w18', type=str, 
                       help='Encoder variant (only for HRNetEncoder). Options: hrnet_w18, hrnet_w32, hrnet_w48. Default: hrnet_w18')
    parser.add_argument('--weight-init', default='kaiming-normal', help='Weight-initialization scheme. Default: kaiming-normal')
    parser.add_argument('--wrn-depth', default=14, type=int, help='Amount of layers in the wide residual network. Default: 14')
    parser.add_argument('--wrn-widen-factor', default=10, type=int, help='Width factor of the wide residual network. Default: 10')
    parser.add_argument('--wrn-embedding-size', type=int, default=48,
                        help='Number of output channels in the first wrn layer. Default: 48')
    parser.add_argument('--double-wrn-blocks', type=bool, help='Use 6 instead of 3 blocks. Default: False')
    
    # Training hyper-parameters
    parser.add_argument('-b', '--batch-size', default=128, type=int, help='Mini-batch size. Default: 128')
    parser.add_argument('-bn', '--batch-norm', default=1e-5, type=float, help='Batch normalization. Default: 1e-5')
    
    # Resuming training
    parser.add_argument('--resume', default='', type=str, required=True,
                        help='Path to model checkpoint to load/resume from. Required for evaluation.')
    
    # Variational parameters
    parser.add_argument('--var-latent-dim', default=60, type=int, help='Dimensionality of latent space. Default: 60')
    parser.add_argument('--var-beta', default=0.1, type=float, help='Weight term for KLD loss. Default: 0.1')
    parser.add_argument('--var-samples', default=1, type=int,
                        help='Number of samples for the expectation in variational training. Default: 1. Use >1 for uncertainty calculation.')
    
    # Open set arguments
    parser.add_argument('--distance-function', default='cosine', 
                        help='Openset distance function. Default: cosine. Choices: euclidean|cosine|mix')
    parser.add_argument('-tailsize', '--openset-weibull-tailsize', default=0.05, type=float,
                        help='Tailsize in percent of data (float in range 0-1). Default: 0.05')
    
    # Open set standalone script
    parser.add_argument('--openset-datasets', default='FashionMNIST,AudioMNIST,KMNIST,CIFAR10,CIFAR100,SVHN',
                        help='Comma-separated names of openset datasets. Default: FashionMNIST,AudioMNIST,KMNIST,CIFAR10,CIFAR100,SVHN')
    parser.add_argument('--percent-validation-outliers', default=0.05, type=float,
                        help='Assumed percentage of inherent outliers in validation set. Default: 0.05 (5%%). Used to find priors and thresholds.')
    parser.add_argument('--calc-reconstruction', default=False, type=bool,
                        help='Calculate decoder/reconstruction loss. Computationally expensive. Default: False')
    
    # PixelVAE
    parser.add_argument('--autoregression', default=False, type=bool, help='Use PixelCNN decoder for generation. Default: False')
    parser.add_argument('--out-channels', default=60, type=int, 
                        help='Number of output channels of decoder when autoregression is used. Default: 60')

    parser.add_argument('--max-train-samples', default=450, type=int, help='maximum number of training samples. Default: None (use all samples)')
    parser.add_argument('--max-test-samples', default=None, type=int, help='maximum number of test samples for openset datasets. Default: None (use all 300K TinyImageNet samples)')
    
    parser.add_argument('--baseline', type=str, default='openmax', choices=['openmax', 'softmax'], help='Chọn baseline để so sánh: openmax hoặc softmax. Default: openmax')
    return parser.parse_args()


def main():
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Command line options
    args = parse_args()
    
    # Setup logging - save to same directory as model checkpoint
    checkpoint_dir = os.path.dirname(args.resume)
    logger, log_file = setup_logging(checkpoint_dir)
    
    logger.info("="*80)
    logger.info("Starting Open Set Recognition Evaluation")
    logger.info("="*80)
    logger.info(f"Log file saved to: {log_file}")
    logger.info("Command line options:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")

    # Get the dataset which has been trained and the corresponding number of classes
    data_init_method = getattr(datasets, args.dataset)
    dataset = data_init_method(torch.cuda.is_available(), args)
    num_classes = dataset.num_classes
    net_input, _ = next(iter(dataset.train_loader))
    num_colors = net_input.size(1)

    # Split a part of the non-used dataset to use as validation set for determining open set (e.g entropy)
    # rejection thresholds
    split_perc = 0.5
    val_len = len(dataset.valset)
    split1_len = int((1 - split_perc) * val_len)
    split2_len = val_len - split1_len  # Ensure total equals val_len
    
    split_sets = torch.utils.data.random_split(dataset.valset, [split1_len, split2_len])

    # overwrite old set and create new split set to determine thresholds/priors
    dataset.valset = split_sets[0]
    dataset.threshset = split_sets[1]

    # overwrite old data loader and create new loader for thresh set
    is_gpu = torch.cuda.is_available()
    dataset.val_loader = torch.utils.data.DataLoader(dataset.valset, batch_size=args.batch_size, shuffle=False,
                                                     num_workers=args.workers, pin_memory=is_gpu, sampler=None)
    dataset.threshset_loader = torch.utils.data.DataLoader(dataset.threshset, batch_size=args.batch_size, shuffle=False,
                                                           num_workers=args.workers, pin_memory=is_gpu, sampler=None)

    # Load open set datasets
    openset_datasets_names = args.openset_datasets.strip().split(',')
    openset_datasets = []
    for openset_dataset in openset_datasets_names:
        openset_data_init_method = getattr(datasets, openset_dataset)
        openset_datasets.append(openset_data_init_method(torch.cuda.is_available(), args))

    if not args.autoregression:
        args.out_channels = num_colors

    # Initialize empty model
    net_init_method = getattr(architectures, args.architecture)
    model = net_init_method(device, num_classes, num_colors, args)

    model = torch.nn.DataParallel(model).to(device)

    # load model (using the resume functionality)
    assert(os.path.isfile(args.resume)), "=> no model checkpoint found at '{}'".format(args.resume)

    # Fill the random model with the parameters of the checkpoint
    logger.info("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    best_prec = checkpoint['best_prec']
    best_loss = checkpoint['best_loss']
    # print the saved model's validation accuracy (as a check to see if the loaded model has really been trained)
    logger.info(f"Saved model's validation accuracy: {best_prec}")
    logger.info(f"Saved model's validation loss: {best_loss}")
    
    # Load state dict - handle both DataParallel and non-DataParallel models
    state_dict = checkpoint['state_dict']
    if isinstance(model, torch.nn.DataParallel):
        # If keys have 'module.' prefix, remove them for DataParallel
        if any(k.startswith('module.') for k in state_dict.keys()):
            pass  # Already has module prefix
        else:
            # Add 'module.' prefix to all keys
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    else:
        # Remove 'module.' prefix if present for non-DataParallel model
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # set the save path to the directory from which the model has been loaded
    save_path = os.path.dirname(args.resume)

    # Visualize class distribution for the dataset
    logger.info("="*80)
    visualize_class_distribution(dataset, args.dataset, save_path, split='all')
    visualize_class_distribution(dataset, args.dataset, save_path, split='train')
    visualize_class_distribution(dataset, args.dataset, save_path, split='val')
    logger.info("="*80)

    # start of the model evaluation on the training dataset and fitting
    logger.info("Evaluating original train dataset: " + args.dataset + ". This may take a while...")
    dataset_eval_dict_train = eval_dataset(model, dataset.train_loader, dataset.num_classes, device,
                                           samples=args.var_samples, calc_reconstruction=args.calc_reconstruction,
                                           autoregression=args.autoregression)
    logger.info(f"Training accuracy: {dataset_eval_dict_train['accuracy']}")

    # Get the mean of z for correctly classified data inputs
    mean_zs = get_means(dataset_eval_dict_train["zs_correct"])

    # visualize the mean z vectors
    # Filter out empty lists and convert to tensor
    mean_zs_tensors = [m for m in mean_zs if isinstance(m, torch.Tensor) and m.numel() > 0]
    if len(mean_zs_tensors) > 0:
        mean_zs_tensor = torch.stack(mean_zs_tensors, dim=0)
        
        # Create class labels only for classes that have valid means
        valid_class_indices = [i for i, m in enumerate(mean_zs) if isinstance(m, torch.Tensor) and m.numel() > 0]
        valid_class_to_idx = {list(dataset.class_to_idx.keys())[i]: i for i in valid_class_indices if i < len(dataset.class_to_idx)}
        
        visualize_means(mean_zs_tensor, valid_class_to_idx, args.dataset, save_path, "z")
    else:
        logger.warning("No valid mean z vectors to visualize")

    # calculate each correctly classified example's distance to the mean z
    distances_to_z_means_correct_train = calc_distances_to_means(mean_zs, dataset_eval_dict_train["zs_correct"],
                                                                 args.distance_function)

    # Weibull fitting
    # set tailsize according to command line parameters (according to percentage of dataset size)
    # but ensure it doesn't exceed the minimum number of correctly classified samples per class
    tailsize = int(len(dataset.trainset) * args.openset_weibull_tailsize / num_classes)
    
    # Find minimum number of correctly classified samples across all classes
    min_correct_samples = float('inf')
    for i, zs in enumerate(dataset_eval_dict_train["zs_correct"]):
        if isinstance(zs, list):
            num_samples = len(zs)
        elif isinstance(zs, torch.Tensor):
            num_samples = zs.size(0)
        else:
            num_samples = 0
        
        if num_samples > 0:
            min_correct_samples = min(min_correct_samples, num_samples)
    
    # Handle case where no class has correctly classified samples
    if min_correct_samples == float('inf'):
        logger.error("No correctly classified samples found in any class!")
        logger.error(f"Training accuracy: {dataset_eval_dict_train['accuracy']:.2%}")
        raise ValueError("Cannot fit Weibull models: no correctly classified samples. Please train the model better.")
    
    # Adjust tailsize to be at most 80% of minimum correct samples
    tailsize = min(tailsize, int(min_correct_samples * 0.8))
    tailsize = max(tailsize, 5)  # Ensure minimum tailsize of 5
    
    logger.info(f"Fitting Weibull models:")
    logger.info(f"  Calculated tailsize: {int(len(dataset.trainset) * args.openset_weibull_tailsize / num_classes)}")
    logger.info(f"  Min correct samples per class: {min_correct_samples}")
    logger.info(f"  Adjusted tailsize: {tailsize}")
    
    tailsizes = [tailsize] * num_classes
    weibull_models, valid_weibull = fit_weibull_models(distances_to_z_means_correct_train, tailsizes)
    assert valid_weibull, "Weibull fit is not valid"

    # Determine rejection thresholds/priors on the created split set
    logger.info("Evaluating original threshold split dataset: " + args.dataset + ". This may take a while...")
    threshset_eval_dict = eval_dataset(model, dataset.threshset_loader, num_classes, device, samples=args.var_samples,
                                       calc_reconstruction=args.calc_reconstruction, autoregression=args.autoregression)

    # Again calculate distances to mean z
    logger.info(f"Split set accuracy: {threshset_eval_dict['accuracy']}")
    distances_to_z_means_threshset = calc_distances_to_means(mean_zs, threshset_eval_dict["zs_correct"],
                                                             args.distance_function)

    outlier_probs_threshset = calc_outlier_probs(weibull_models, distances_to_z_means_threshset)

    threshset_classification = calc_openset_classification(outlier_probs_threshset, num_classes,
                                                           num_outlier_threshs=100)
    max_entropy = np.max(threshset_eval_dict["out_entropy"])
    threshset_entropy_classification = calc_entropy_classification(threshset_eval_dict["out_entropy"],
                                                                   max_entropy,
                                                                   num_outlier_threshs=100)

    # We have added a flag to turn off calculation of the decoder because it is computationally heavy for many samples
    # (repeated calculation of the decoder), whereas latent space sampling and repeated calculation of our latent based
    # EVT approach and even the single layer classifier is cheap.
    if args.calc_reconstruction:
        max_recon_loss = np.max(threshset_eval_dict["recon_loss_mus"])
        threshset_recon_classification = calc_reconstruction_classification(threshset_eval_dict["recon_loss_mus"],
                                                                            max_recon_loss,
                                                                            num_outlier_threshs=1000)

    # determine the index for the corresponding rejection priors/thresholds. Although this should never happen,
    # we also set a default if no threshold satisfies the 95% inlier condition.
    if (np.array(threshset_classification["outlier_percentage"]) <= args.percent_validation_outliers).any() == True:
        EVT_prior_index = np.argwhere(np.array(threshset_classification["outlier_percentage"])
                                      <= 0.05)[0][0]
        EVT_prior = threshset_classification["thresholds"][EVT_prior_index]
    else:
        EVT_prior = 0.5
        EVT_prior_index = 50

    if (np.array(threshset_entropy_classification["entropy_outlier_percentage"]) <=
        args.percent_validation_outliers).any() == True:
        entropy_threshold_index = np.argwhere(np.array(threshset_entropy_classification["entropy_outlier_percentage"])
                                              <= 0.05)[0][0]
        entropy_threshold = threshset_entropy_classification["entropy_thresholds"][entropy_threshold_index]
    else:
        entropy_threshold = np.median(threshset_entropy_classification["entropy_thresholds"])
        entropy_threshold_index = 50

    if args.calc_reconstruction:
        if (np.array(threshset_recon_classification["reconstruction_outlier_percentage"]) <=
            args.percent_validation_outliers).any() == True:
            recon_threshold_index = np.argwhere(
                np.array(threshset_recon_classification["reconstruction_outlier_percentage"]) <= 0.05)[0][0]
            recon_threshold = threshset_recon_classification["reconstruction_thresholds"][recon_threshold_index]
        else:
            recon_threshold = np.median(threshset_recon_classification["reconstruction_thresholds"])
            recon_threshold_index = 500

    logger.info("EVT prior: " + str(EVT_prior) + "; Entropy threshold: " + str(entropy_threshold))
    if args.calc_reconstruction:
        logger.info("Reconstruction loss threshold: " + str(recon_threshold))

    # ------------------------------------------------------------------------------------------
    # Fitting on train dataset complete. Beginning of all testing/open set recognition on validation and unknown sets.
    # ------------------------------------------------------------------------------------------
    # We evaluate the validation set to later evaluate trained dataset's statistical inlier/outlier estimates.
    logger.info("Evaluating original validation dataset: " + args.dataset + ". This may take a while...")
    dataset_eval_dict = eval_dataset(model, dataset.val_loader, num_classes, device, samples=args.var_samples,
                                     calc_reconstruction=args.calc_reconstruction, autoregression=args.autoregression)

    # Again calculate distances to mean z
    logger.info(f"Validation accuracy: {dataset_eval_dict['accuracy']}")
    distances_to_z_means_correct = calc_distances_to_means(mean_zs, dataset_eval_dict["zs_correct"],
                                                           args.distance_function)

    # Evaluate outlier probability of trained dataset's validation set
    outlier_probs_correct = calc_outlier_probs(weibull_models, distances_to_z_means_correct)

    dataset_classification_correct = calc_openset_classification(outlier_probs_correct, num_classes,
                                                                 num_outlier_threshs=100)
    dataset_entropy_classification_correct = calc_entropy_classification(dataset_eval_dict["out_entropy"],
                                                                         max_entropy,
                                                                         num_outlier_threshs=100)
    if args.calc_reconstruction:
        dataset_recon_classification_correct = calc_reconstruction_classification(dataset_eval_dict["recon_loss_mus"],
                                                                                  max_recon_loss,
                                                                                  num_outlier_threshs=1000)

    logger.info(args.dataset + '(trained) EVT outlier percentage: ' +
          str(dataset_classification_correct["outlier_percentage"][EVT_prior_index]))
    logger.info(args.dataset + '(trained) entropy outlier percentage: ' +
          str(dataset_entropy_classification_correct["entropy_outlier_percentage"][entropy_threshold_index]))
    if args.calc_reconstruction:
        logger.info(args.dataset + '(trained) reconstruction loss outlier percentage: ' +
              str(dataset_recon_classification_correct["reconstruction_outlier_percentage"][recon_threshold_index]))

    # ------------------------------------------------------------------------------------------
    # Repeat process for open set recognition (no fitting, just testing) on all unseen datasets
    # ------------------------------------------------------------------------------------------
    # dicitionaries to hold results
    openset_dataset_eval_dicts = collections.OrderedDict()
    openset_outlier_probs_dict = collections.OrderedDict()
    openset_classification_dict = collections.OrderedDict()
    openset_entropy_classification_dict = collections.OrderedDict()
    if args.calc_reconstruction:
        openset_recon_classification_dict = collections.OrderedDict()

    for od, openset_dataset in enumerate(openset_datasets):
        logger.info("Evaluating openset dataset: " + openset_datasets_names[od] + ". This may take a while...")

        openset_dataset_eval_dict = eval_openset_dataset(model, openset_dataset.val_loader, num_classes, device,
                                                         samples=args.var_samples, autoregression=args.autoregression,
                                                         calc_reconstruction=args.calc_reconstruction)

        # --- OPENMAX/EVT ---
        openset_distances_to_z_means = calc_distances_to_means(mean_zs, openset_dataset_eval_dict["zs"],
                                                               args.distance_function)
        openset_outlier_probs = calc_outlier_probs(weibull_models, openset_distances_to_z_means)
        openset_classification = calc_openset_classification(openset_outlier_probs, num_classes,
                                                             num_outlier_threshs=100)
        openset_entropy_classification = calc_entropy_classification(openset_dataset_eval_dict["out_entropy"],
                                                                     max_entropy,
                                                                     num_outlier_threshs=100)
        if args.calc_reconstruction:
            openset_recon_classification_correct = calc_reconstruction_classification(
                openset_dataset_eval_dict["recon_loss_mus"], max_recon_loss, num_outlier_threshs=1000)


        # --- SOFTMAX LABEL (label có score cao nhất, có threshold) ---
        # Tính max softmax score cho từng mẫu
        softmax_max_scores = []
        softmax_preds = []
        for i in range(len(openset_dataset_eval_dict["zs"][0])):
            sample_probs = []
            for c in range(num_classes):
                if len(openset_dataset_eval_dict["out_mus"][c]) > i:
                    sample_probs.append(openset_dataset_eval_dict["out_mus"][c][i])
                else:
                    sample_probs.append(0.0)
            sample_probs_tensor = torch.tensor(sample_probs)
            max_score = sample_probs_tensor.max().item()
            pred_label = int(sample_probs_tensor.argmax().item())
            softmax_max_scores.append(max_score)
            softmax_preds.append(pred_label)


        # --- Tự động chọn ngưỡng tối ưu trên tập validation (threshset_eval_dict) ---
        # Quét các ngưỡng từ 0.5 đến 0.99, chọn ngưỡng sao cho tỷ lệ unknown ~5% (hoặc gần nhất)
        val_softmax_max_scores = []
        val_softmax_preds = []
        val_eval = threshset_eval_dict if 'threshset_eval_dict' in locals() else None
        if val_eval is not None:
            for i in range(len(val_eval["zs_correct"][0])):
                sample_probs = []
                for c in range(num_classes):
                    if len(val_eval["out_mus_correct"][c]) > i:
                        sample_probs.append(val_eval["out_mus_correct"][c][i])
                    else:
                        sample_probs.append(0.0)
                sample_probs_tensor = torch.tensor(sample_probs)
                val_softmax_max_scores.append(sample_probs_tensor.max().item())
                val_softmax_preds.append(int(sample_probs_tensor.argmax().item()))
            best_T = 0.5
            best_gap = 1.0
            best_unknown = 1.0
            for T in [round(x, 3) for x in list(torch.arange(0.5, 0.991, 0.01).numpy())]:
                unknown_count = sum([score < T for score in val_softmax_max_scores])
                unknown_rate = unknown_count / len(val_softmax_max_scores)
                gap = abs(unknown_rate - 0.05)
                if gap < best_gap:
                    best_gap = gap
                    best_T = T
                    best_unknown = unknown_rate
            SOFTMAX_THRESHOLD = best_T
            logger.info(f"[SOFTMAX] Auto-selected threshold={SOFTMAX_THRESHOLD} (unknown rate on val: {best_unknown:.3f})")
        else:
            SOFTMAX_THRESHOLD = 0.8
            logger.info(f"[SOFTMAX] Default threshold={SOFTMAX_THRESHOLD}")

        softmax_final_labels = []
        for score, label in zip(softmax_max_scores, softmax_preds):
            if score >= SOFTMAX_THRESHOLD:
                softmax_final_labels.append(label)
            else:
                softmax_final_labels.append(-1)  # -1 là unknown
        softmax_label_dist = Counter(softmax_final_labels)
        logger.info(f"[SOFTMAX] {openset_datasets_names[od]} label distribution (top-1, threshold={SOFTMAX_THRESHOLD}, unknown=-1): {dict(softmax_label_dist)}")

        # --- OPENMAX/EVT như cũ ---
        evt_rejects = []
        for i in range(len(openset_dataset_eval_dict["zs"][0])):
            sample_outlier_probs = [openset_outlier_probs[c][i] if len(openset_outlier_probs[c]) > i else 0.0 for c in range(num_classes)]
            if all([p > EVT_prior for p in sample_outlier_probs]):
                evt_rejects.append(-1)
            else:
                evt_rejects.append(int(torch.tensor(sample_outlier_probs).argmin().item()))
        evt_label_dist = Counter(evt_rejects)
        logger.info(f"[OPENMAX/EVT] {openset_datasets_names[od]} label distribution (top-1 or reject=-1): {dict(evt_label_dist)}")

        openset_dataset_eval_dicts[openset_datasets_names[od]] = openset_dataset_eval_dict
        openset_outlier_probs_dict[openset_datasets_names[od]] = openset_outlier_probs
        openset_classification_dict[openset_datasets_names[od]] = openset_classification
        openset_entropy_classification_dict[openset_datasets_names[od]] = openset_entropy_classification
        if args.calc_reconstruction:
            openset_recon_classification_dict[openset_datasets_names[od]] = openset_recon_classification_correct

    # Print the results
    for other_data_name, other_data_dict in openset_classification_dict.items():
        logger.info(other_data_name + ' EVT outlier percentage: ' +
              str(other_data_dict["outlier_percentage"][entropy_threshold_index]))
    for other_data_name, other_data_dict in openset_entropy_classification_dict.items():
        logger.info(other_data_name + ' entropy outlier percentage: ' +
              str(other_data_dict["entropy_outlier_percentage"][entropy_threshold_index]))
    if args.calc_reconstruction:
        for other_data_name, other_data_dict in openset_recon_classification_dict.items():
            logger.info(other_data_name + ' reconstruction loss outlier percentage: ' +
                  str(other_data_dict["reconstruction_outlier_percentage"][recon_threshold_index]))

    # joint prediction uncertainty plot for all datasets
    if args.var_samples > 1:
        visualize_classification_uncertainty(dataset_eval_dict["out_mus_correct"],
                                             dataset_eval_dict["out_sigmas_correct"],
                                             openset_dataset_eval_dicts,
                                             "out_mus", "out_sigmas",
                                             args.dataset + ' (trained)',
                                             args.var_samples, save_path)

    # visualize 2D latent embedding for open set (if latent_dim=2)
    if args.var_latent_dim == 2:
        logger.info("Creating 2D open-set latent embedding visualization...")
        # Collect unknown embeddings from all openset datasets
        unknown_embeddings_dict = {name: eval_dict["zs"] 
                                  for name, eval_dict in openset_dataset_eval_dicts.items()}
        visualize_openset_2d_embedding(dataset_eval_dict_train["zs_correct"],
                                      unknown_embeddings_dict,
                                      args.dataset,
                                      save_path,
                                      num_classes)

    # visualize the outlier probabilities
    visualize_weibull_outlier_probabilities(outlier_probs_correct, openset_outlier_probs_dict,
                                            args.dataset + ' (trained)', save_path, tailsize)

    # Visualize Open-Set Recognition confusion matrices
    logger.info("="*80)
    visualize_openset_confusion_matrix(dataset_eval_dict, openset_dataset_eval_dicts,
                                      outlier_probs_correct, openset_outlier_probs_dict,
                                      EVT_prior, entropy_threshold,
                                      args.dataset, num_classes, save_path)
    logger.info("="*80)

    # histograms
    visualize_classification_scores(dataset_eval_dict["out_mus_correct"], openset_dataset_eval_dicts, 'out_mus',
                                    args.dataset + ' (trained)', save_path)
    visualize_entropy_histogram(dataset_eval_dict["out_entropy"], openset_dataset_eval_dicts,
                                dataset_entropy_classification_correct["entropy_thresholds"][-1], "out_entropy",
                                args.dataset + ' (trained)', save_path)
    if args.calc_reconstruction:
        visualize_recon_loss_histogram(dataset_eval_dict["recon_loss_mus"], openset_dataset_eval_dicts,
                                       dataset_recon_classification_correct["reconstruction_thresholds"][-1],
                                       "recon_loss_mus", args.dataset + ' (trained)', save_path)

    # joint plot for outlier detection accuracy for both seen and unseen datasets
    visualize_openset_classification(dataset_classification_correct["outlier_percentage"],
                                     openset_classification_dict, "outlier_percentage",
                                     args.dataset + ' (trained)',
                                     dataset_classification_correct["thresholds"], save_path, tailsize)
    visualize_entropy_classification(dataset_entropy_classification_correct["entropy_outlier_percentage"],
                                     openset_entropy_classification_dict, "entropy_outlier_percentage",
                                     args.dataset + ' (trained)',
                                     dataset_entropy_classification_correct["entropy_thresholds"], save_path)
    if args.calc_reconstruction:
        visualize_reconstruction_classification(dataset_recon_classification_correct["reconstruction_outlier_percentage"],
                                                openset_recon_classification_dict, "reconstruction_outlier_percentage",
                                                args.dataset + ' (trained)',
                                                dataset_recon_classification_correct["reconstruction_thresholds"],
                                                save_path, autoregression=args.autoregression)


if __name__ == '__main__':
    main()