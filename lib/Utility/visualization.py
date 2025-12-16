
import torch
import torchvision
import os
import math
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
import glob
from PIL import Image
from matplotlib.gridspec import GridSpec
import logging
import json
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

# matplotlib backend, required for plotting of images to tensorboard
matplotlib.use('Agg')

# setting font sizes
title_font_size = 60
axes_font_size = 45
legend_font_size = 36
ticks_font_size = 48

# setting seaborn specifics
sns.set(font_scale=2.5)
sns.set_style("whitegrid")
colors = sns.color_palette("Set2")
pal = sns.cubehelix_palette(10, light=0.0)
linestyles = [(0, (1, 3)),  # 'dotted'
              (0, (1, 1)),  # 'densely dotted'
              (0, (2, 2)),  # 'dashed'
              (0, (3, 1)),  # 'densely dashed'
              (0, (3, 3, 1, 3)),  # 'dashdotted'
              (0, (3, 1, 1, 1)),  # 'densely dashdotted'
              (0, (3, 3, 1, 3, 1, 3)),  # 'dashdotdotted'
              (0, (3, 1, 1, 1, 1, 1))]  # 'densely dashdotdotted'


def args_to_tensorboard(writer, args):
    """
    Takes command line parser arguments and formats them to
    display them in TensorBoard text.

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        args (dict): dictionary of command line arguments
    """

    txt = ""
    for arg in vars(args):
        txt += arg + ": " + str(getattr(args, arg)) + "<br/>"

    writer.add_text('command_line_parameters', txt, 0)


def visualize_image_grid(images, writer, count, name, save_path):
    """
    Visualizes a grid of images and saves it to hard-drive

    Parameters:
        images (torch.Tensor): Tensor of images.
        writer: Deprecated parameter (kept for backward compatibility, can be None).
        count (int): counter usually specifying steps/epochs/time.
        name (str): name of the file to save.
        save_path (str): path where image grid is going to be saved.
    """
    logger = logging.getLogger()
    logger.info(f"Creating image grid: {name} (epoch {count})...")
    size = images.size(0)
    save_file = os.path.join(save_path, name + '_epoch_' + str(count) + '.png')
    torchvision.utils.save_image(images, save_file,
                                 nrow=int(math.sqrt(size)), padding=5)
    logger.info(f'✓ Image grid saved: {save_file}')


def visualize_confusion(writer, step, matrix, class_dict, save_path):
    """
    Visualization of confusion matrix. Is saved to hard-drive.

    Parameters:
        writer: Deprecated parameter (kept for backward compatibility, can be None).
        step (int): Counter usually specifying steps/epochs/time.
        matrix (numpy.array): Square-shaped array of size class x class.
            Should specify cross-class accuracies/confusion in percent
            values (range 0-1).
        class_dict (dict): Dictionary specifying class names as keys and
            corresponding integer labels/targets as values.
        save_path (str): Path used for saving
    """
    logger = logging.getLogger()
    
    logger.info(f"Creating confusion matrix (step {step})...")

    all_categories = sorted(class_dict, key=class_dict.get)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax, boundaries=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Turn off the grid for this plot
    ax.grid(False)
    plt.tight_layout()

    save_file = os.path.join(save_path, 'confusion_epoch_' + str(step) + '.png')
    plt.savefig(save_file, bbox_inches='tight')
    plt.close()
    logger.info(f'✓ Confusion matrix saved: {save_file}')


def visualize_confusion_heatmap(writer, step, matrix, class_dict, save_path, known_classes=None):
    """
    Generate a confusion matrix heatmap with auto-scaled cell sizes.
    Only shows known/closed-set classes if known_classes is provided.
    
    Parameters:
        writer: Deprecated parameter (kept for backward compatibility, can be None).
        step (int): Counter usually specifying steps/epochs/time.
        matrix (numpy.array): Square-shaped confusion matrix.
        class_dict (dict): Dictionary mapping class names to indices.
        save_path (str): Path used for saving.
        known_classes (list): List of known class indices to display. If None, show all classes.
    """
    logger = logging.getLogger()
    
    logger.info(f"Creating confusion matrix heatmap (step {step})...")
    
    # If known_classes is provided, filter matrix to only show those classes
    if known_classes is not None:
        # Extract only known classes from matrix
        matrix_filtered = matrix[np.ix_(known_classes, known_classes)]
        # Get class names for known classes only
        all_class_names = sorted(class_dict, key=class_dict.get)
        class_names = [all_class_names[i] for i in known_classes]
        num_classes = len(known_classes)
        
        # Log confusion matrix for known classes
        logger.info(f"\n{'='*60}")
        logger.info(f"Confusion Matrix (Known/Closed-Set Classes Only) - Epoch {step}")
        logger.info(f"Known classes: {known_classes}")
        logger.info(f"{'='*60}")
        logger.info(f"{'True/Pred':<12}" + "".join([f"{name:>8}" for name in class_names]))
        logger.info("-" * (12 + 8 * num_classes))
        for i, true_class in enumerate(class_names):
            row_str = f"{true_class:<12}" + "".join([f"{matrix_filtered[i, j]:>8d}" for j in range(num_classes)])
            logger.info(row_str)
        logger.info(f"{'='*60}\n")
    else:
        # Use full matrix
        matrix_filtered = matrix
        class_names = sorted(class_dict, key=class_dict.get)
        num_classes = len(class_names)

    matrix_int = matrix_filtered.astype(int)
    max_value = np.max(matrix_int) if np.max(matrix_int) > 0 else 1

    cell_size = 1.1
    fig_w = max(8, num_classes * cell_size)
    fig_h = max(8, num_classes * cell_size)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    annot_font = max(6, min(18, 220 // max(num_classes, 1)))
    tick_font = max(6, min(16, 180 // max(num_classes, 1)))

    sns.heatmap(
        matrix_int,
        annot=True,
        fmt="d",
        cmap="Blues",
        square=True,
        vmin=0,
        vmax=max_value,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": annot_font},
        cbar_kws={"label": "Sample Count"},
        ax=ax
    )

    ax.set_aspect('equal')         
    ax.set_xlim(0, num_classes)
    ax.set_ylim(num_classes, 0)

    ax.set_xlabel("Predicted Class", fontsize=16, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=16, fontweight="bold")
    
    title_suffix = " (Known Classes Only)" if known_classes is not None else ""
    ax.set_title(f"Confusion Matrix – Epoch {step}", fontsize=18, fontweight="bold", pad=20)

    ax.tick_params(axis="both", labelsize=tick_font)

    plt.tight_layout(pad=2.0)

    out_file = os.path.join(save_path, f"confusion_heatmap_epoch_{step}.png")
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Confusion matrix heatmap saved to: {out_file}")



def visualize_dataset_in_2d_embedding(writer, encoding_list, dataset_name, save_path, task=1):
    """
    Visualization of 2-D latent embedding. Is saved to hard-disc as image file.

    Parameters:
        writer: Deprecated parameter (kept for backward compatibility, can be None).
        encoding_list (list): List of Tensors containing encoding values
        dataset_name (str): Dataset name.
        save_path (str): Path used for saving.
        task (int): task counter. Used for naming.
    """
    logger = logging.getLogger()

    logger.info(f"Creating 2D latent space visualization for {dataset_name} (task {task})...")
    
    num_classes = len(encoding_list)
    encoded_classes = []
    for i in range(len(encoding_list)):
        if isinstance(encoding_list[i], torch.Tensor):
            encoded_classes.append([i] * encoding_list[i].size(0))
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            encoding_list[i] = torch.Tensor(encoding_list[i]).to(device)
            encoded_classes.append([i] * 0)
    # Fix: concatenate the list of lists properly
    encoded_classes = np.concatenate([np.array(cls) for cls in encoded_classes if len(cls) > 0], axis=0)
    encoding = torch.cat(encoding_list, dim=0)

    if encoding.size(1) != 2:
        print("Skipping visualization of latent space because it is not 2-D")
        return

    # select first and second dimension
    encoded_dim1 = np.squeeze(encoding.narrow(1, 0, 1).cpu().numpy())
    encoded_dim2 = np.squeeze(encoding.narrow(1, 1, 1).cpu().numpy())

    xlabel = 'z dimension 1'
    ylabel = 'z dimension 2'

    my_cmap = ListedColormap(sns.color_palette("Paired", num_classes).as_hex())
    fig = plt.figure(figsize=(20, 20))
    plt.scatter(encoded_dim1, encoded_dim2, c=encoded_classes, cmap=my_cmap)

    plt.xlabel(xlabel, fontsize=axes_font_size)
    plt.ylabel(ylabel, fontsize=axes_font_size)
    plt.xticks(fontsize=ticks_font_size)
    plt.yticks(fontsize=ticks_font_size)

    cbar = plt.colorbar(ticks=np.linspace(0, num_classes-1, num_classes))
    cbar.ax.set_yticklabels([str(i) for i in range(num_classes)])
    cbar.ax.tick_params(labelsize=legend_font_size)

    plt.tight_layout()

    # Save to file (ignore writer/TensorBoard)
    save_file = os.path.join(save_path, dataset_name + '_latent_2d_embedding_task_' +
                             str(task) + '.png')
    plt.savefig(save_file, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"✓ 2D latent space visualization saved: {save_file}")


def visualize_means(means, classes_order, data_name, save_path, name):
    """
    Visualization of means, e.g. of latent code z.

    Parameters:
        means (torch.Tensor): 2-D Tensor with one mean z vector per class.
        classes_order (dict): Defines mapping between integer indices and class names (strings).
        data_name (str): Dataset name. Used for naming.
        save_path (str): Saving path.
        name (str): Name for type of mean, e.g. "z".
    """
    classes_order = sorted(classes_order)
    classes = []
    for key in classes_order:
        classes.append(key)

    plt.figure(figsize=(20, 20))
    ax = sns.heatmap(means.cpu().numpy(), cmap="BrBG")
    ax.set_title(data_name, fontsize=title_font_size)
    ax.set_xlabel(name + ' mean activations', fontsize=axes_font_size)
    ax.set_yticklabels(classes, rotation=0)
    plt.savefig(os.path.join(save_path, name + '_mean_activations.png'), bbox_inches='tight')


def visualize_classification_uncertainty(data_mus, data_sigmas, other_data_dicts, other_data_mu_key,
                                         other_data_sigma_key,
                                         data_name, num_samples, save_path):
    """
    Visualization of prediction uncertainty computed over multiple samples for each input.

    Parameters:
        data_mus (list or torch.Tensor): Encoded mu values for trained dataset's validation set.
        data_sigmas (list or torch.Tensor): Encoded sigma values for trained dataset's validation set.
        other_data_dicts (dictionary of dictionaries): A dataset with values per dictionary, among them mus and sigmas
        other_data_mu_key (str): Dictionary key for the mus
        other_data_sigma_key (str): Dictionary key for the sigmas
        data_name (str): Original dataset's name.
        num_samples (int): Number of used samples to obtain prediction values.
        save_path (str): Saving path.
    """

    data_mus = [y for x in data_mus for y in x]
    data_sigmas = [y for x in data_sigmas for y in x]

    plt.figure(figsize=(20, 14))
    plt.scatter(data_mus, data_sigmas, label=data_name, s=75, c=colors[0], alpha=1.0)

    c = 0
    for other_data_name, other_data_dict in other_data_dicts.items():
        other_data_mus = [y for x in other_data_dict[other_data_mu_key] for y in x]
        other_data_sigmas = [y for x in other_data_dict[other_data_sigma_key] for y in x]
        plt.scatter(other_data_mus, other_data_sigmas, label=other_data_name, s=75, c=colors[c], alpha=0.3,
                    marker='*')
        c += 1

    plt.xlabel("Prediction mean", fontsize=axes_font_size)
    plt.ylabel("Prediction standard deviation", fontsize=axes_font_size)
    plt.xlim(left=-0.05, right=1.05)
    plt.ylim(bottom=-0.05, top=0.55)
    plt.legend(loc=1, fontsize=legend_font_size)
    plt.savefig(os.path.join(save_path, data_name + '_vs_' + ",".join(list(other_data_dicts.keys())) +
                             '_classification_uncertainty_' + str(num_samples) + '_samples.pdf'),
                bbox_inches='tight')


def visualize_classification_scores(data, other_data_dicts, dict_key, data_name, save_path):
    """
    Visualization of classification scores per dataset.

    Parameters:
        data (list): Classification scores.
        other_data_dicts (dictionary of dictionaries): Dictionary of key-value pairs per dataset
        dict_key (string): Dictionary key to plot
        data_name (str): Original trained dataset's name.
        save_path (str): Saving path.
    """

    data = [y for x in data for y in x]

    plt.figure(figsize=(20, 20))
    plt.hist(data, label=data_name, alpha=1.0, bins=20, color=colors[0])

    c = 0
    for other_data_name, other_data_dict in other_data_dicts.items():
        other_data = [y for x in other_data_dict[dict_key] for y in x]
        plt.hist(other_data, label=other_data_name, alpha=0.5, bins=20, color=colors[c])
        c += 1

    plt.title("Dataset classification", fontsize=title_font_size)
    plt.xlabel("Classification confidence", fontsize=axes_font_size)
    plt.ylabel("Number of images", fontsize=axes_font_size)
    plt.legend(loc=0)
    plt.xlim(left=-0.0, right=1.05)

    plt.savefig(os.path.join(save_path, data_name + '_' + ",".join(list(other_data_dicts.keys()))
                             + '_classification_scores.png'),
                bbox_inches='tight')


def visualize_entropy_histogram(data, other_data_dicts, max_entropy, dict_key, data_name, save_path):
    """
    Visualization of the entropy the datasets.

    Parameters:
        data (list):
        other_data_dicts (dictionary of dictionaries): Dictionary of key-value pairs per dataset
        dict_key (str): Dictionary key to plot
        data_name (str): Original trained dataset's name.
        save_path (str): Saving path.
    """
    data = [x for x in data]

    plt.figure(figsize=(20, 20))
    plt.hist(data, label=data_name, alpha=1.0, bins=25, color=colors[0])

    c = 0
    for other_data_name, other_data_dict in other_data_dicts.items():
        other_data = [x for x in other_data_dict[dict_key]]
        plt.hist(other_data, label=other_data_name, alpha=0.5, bins=25, color=colors[c])
        c += 1

    plt.title("Dataset classification entropy", fontsize=title_font_size)
    plt.xlabel("Classification entropy", fontsize=axes_font_size)
    plt.ylabel("Number of images", fontsize=axes_font_size)
    plt.legend(loc=0)
    plt.xlim(left=-0.0, right=max_entropy)
    plt.savefig(os.path.join(save_path, data_name + '_' + ",".join(list(other_data_dicts.keys()))
                             + '_classification_entropies.png'),
                bbox_inches='tight')


def visualize_recon_loss_histogram(data, other_data_dicts, max_recon_loss, dict_key, data_name, save_path):
    """
    Visualization of the entropy the datasets.

    Parameters:
        data (list):
        other_data_dicts (dictionary of dictionaries): Dictionary of key-value pairs per dataset
        dict_key (str): Dictionary key to plot
        data_name (str): Original trained dataset's name.
        save_path (str): Saving path.
    """
    data = [x for x in data]

    plt.figure(figsize=(20, 20))
    plt.hist(data, label=data_name, alpha=1.0, bins=25, color=colors[0])

    c = 0
    for other_data_name, other_data_dict in other_data_dicts.items():
        other_data = [x for x in other_data_dict[dict_key]]
        plt.hist(other_data, label=other_data_name, alpha=0.5, bins=25, color=colors[c])
        c += 1

    plt.title("Dataset reconstruction", fontsize=title_font_size)
    plt.xlabel("Reconstruction loss (nats)", fontsize=axes_font_size)
    plt.ylabel("Number of images", fontsize=axes_font_size)
    plt.legend(loc=0)
    plt.xlim(left=-0.0, right=max_recon_loss)
    plt.savefig(os.path.join(save_path, data_name + '_' + ",".join(list(other_data_dicts.keys()))
                             + '_reconstruction_losses.png'),
                bbox_inches='tight')


def visualize_weibull_outlier_probabilities(data_outlier_probs, other_data_outlier_probs_dict,
                                            data_name, save_path, tailsize):
    """
    Visualization of Weibull CDF outlier probabilites.

    Parameters:
        data_outlier_probs (np.array): Outlier probabilities for each input of the trained dataset's validation set.
        other_data_outlier_probs_dict (dictionary): Outlier probabilities for each input of an unseen dataset.
        data_name (str): Original trained dataset's name.
        save_path (str): Saving path.
        tailsize (int): Fitted Weibull model's tailsize.
    """

    data_outlier_probs = np.concatenate(data_outlier_probs, axis=0)

    data_weights = np.ones_like(data_outlier_probs) / float(len(data_outlier_probs))

    plt.figure(figsize=(20, 20))
    plt.hist(data_outlier_probs, label=data_name, weights=data_weights, bins=50, color=colors[0],
             alpha=1.0, edgecolor='white', linewidth=5)

    c = 0
    for other_data_name, other_data_outlier_probs in other_data_outlier_probs_dict.items():
        other_data_outlier_probs = np.concatenate(other_data_outlier_probs, axis=0)
        other_data_weights = np.ones_like(other_data_outlier_probs) / float(len(other_data_outlier_probs))
        plt.hist(other_data_outlier_probs, label=other_data_name, weights=other_data_weights,
                 bins=50, color=colors[c], alpha=0.5, edgecolor='white', linewidth=5)
        c += 1

    plt.title("Outlier probabilities: tailsize " + str(tailsize), fontsize=title_font_size)
    plt.xlabel("Outlier probability according to Weibull CDF", fontsize=axes_font_size)
    plt.ylabel("Percentage", fontsize=axes_font_size)
    plt.xlim(left=-0.05, right=1.05)
    plt.ylim(bottom=-0.05, top=1.05)
    plt.legend(loc=0)

    plt.savefig(os.path.join(save_path, data_name + '_' + ",".join(list(other_data_outlier_probs_dict.keys()))
                             + '_weibull_outlier_probabilities_tailsize_'
                             + str(tailsize) + '.png'), bbox_inches='tight')


def visualize_openset_classification(data, other_data_dicts, dict_key, data_name,
                                     thresholds, save_path, tailsize):
    """
    Visualization of percentage of datasets considered as statistical outliers evaluated for different
    Weibull CDF rejection priors.

    Parameters:
        data (list): Dataset outlier percentages per rejection prior value for the trained dataset's validation set.
        other_data_dicts (dictionary of dictionaries):
            Dataset outlier percentages per rejection prior value for an unseen dataset.
        dict_key (str): Dictionary key of the values to visualize
        data_name (str): Original trained dataset's name.
        thresholds (list): List of integers with rejection prior values.
        save_path (str): Saving path.
        tailsize (int): Weibull model's tailsize.
    """

    lw = 10
    plt.figure(figsize=(20, 20))
    plt.plot(thresholds, data, label=data_name, color=colors[0], linestyle='solid', linewidth=lw)

    c = 0
    for other_data_name, other_data_dict in other_data_dicts.items():
        plt.plot(thresholds, other_data_dict[dict_key], label=other_data_name, color=colors[c],
                 linestyle=linestyles[c % len(linestyles)], linewidth=lw)
        c += 1

    plt.xlabel(r"Weibull CDF outlier rejection prior $\Omega_t$", fontsize=axes_font_size)
    plt.ylabel("Percentage of dataset outliers", fontsize=axes_font_size)
    plt.xlim(left=-0.05, right=1.05)
    plt.ylim(bottom=-0.05, top=1.05)
    plt.legend(loc=0, fontsize=legend_font_size - 15)
    plt.savefig(os.path.join(save_path, data_name + '_' + ",".join(list(other_data_dicts.keys())) +
                             '_outlier_classification' + '_tailsize_' + str(tailsize) + '.pdf'),
                bbox_inches='tight')


def visualize_entropy_classification(data, other_data_dicts, dict_key, data_name,
                                     thresholds, save_path):
    """
    Visualization of percentage of datasets considered as statistical outliers evaluated for different
    entropy thresholds.

    Parameters:
        data (list): Dataset outlier percentages per rejection prior value for the trained dataset's validation set.
        other_data_dicts (dictionary of dictionaries):
            Dataset outlier percentages per rejection prior value for an unseen dataset.
        dict_key (str): Dictionary key of the values to visualize
        data_name (str): Original trained dataset's name.
        thresholds (list): List of integers with rejection prior values.
        save_path (str): Saving path.
    """

    lw = 10
    plt.figure(figsize=(20, 20))
    plt.plot(thresholds, data, label=data_name, color=colors[0], linestyle='solid', linewidth=lw)

    c = 0
    for other_data_name, other_data_dict in other_data_dicts.items():
        plt.plot(thresholds, other_data_dict[dict_key], label=other_data_name, color=colors[c],
                 linestyle=linestyles[c % len(linestyles)], linewidth=lw)
        c += 1

    plt.xlabel(r"Predictive entropy", fontsize=axes_font_size)
    plt.ylabel("Percentage of dataset outliers", fontsize=axes_font_size)
    plt.xlim(left=-0.05, right=thresholds[-1])
    plt.ylim(bottom=-0.05, top=1.05)
    plt.legend(loc=0, fontsize=legend_font_size - 15)
    plt.savefig(os.path.join(save_path, data_name + '_' + ",".join(list(other_data_dicts.keys())) +
                             '_entropy_outlier_classification' + '.pdf'),
                bbox_inches='tight')


def visualize_reconstruction_classification(data, other_data_dicts, dict_key, data_name,
                                            thresholds, save_path, autoregression=False):
    """
    Visualization of percentage of datasets considered as statistical outliers evaluated for different
    entropy thresholds.

    Parameters:
        data (list): Dataset outlier percentages per rejection prior value for the trained dataset's validation set.
        other_data_dicts (dictionary of dictionaries):
            Dataset outlier percentages per rejection prior value for an unseen dataset.
        dict_key (str): Dictionary key of the values to visualize
        data_name (str): Original trained dataset's name.
        thresholds (list): List of integers with rejection prior values.
        save_path (str): Saving path.
    """

    lw = 10
    plt.figure(figsize=(20, 20))
    plt.plot(thresholds, data, label=data_name, color=colors[0], linestyle='solid', linewidth=lw)

    c = 0
    for other_data_name, other_data_dict in other_data_dicts.items():
        plt.plot(thresholds, other_data_dict[dict_key], label=other_data_name, color=colors[c],
                 linestyle=linestyles[c % len(linestyles)], linewidth=lw)
        c += 1

    if autoregression:
        plt.xlabel(r"Dataset reconstruction loss (bits per dim)", fontsize=axes_font_size)
    else:
        plt.xlabel(r"Dataset reconstruction loss (nats)", fontsize=axes_font_size)
    plt.ylabel("Percentage of dataset outliers", fontsize=axes_font_size)
    plt.xlim(left=-0.05, right=thresholds[-1])
    plt.ylim(bottom=-0.05, top=1.05)
    plt.legend(loc=0, fontsize=legend_font_size - 15)
    plt.savefig(os.path.join(save_path, data_name + '_' + ",".join(list(other_data_dicts.keys())) +
                             '_reconstruction_loss_outlier_classification' + '.pdf'), bbox_inches='tight')


def visualize_all_training_results(save_path, max_cols=3):
    """
    Visualize all training result images in a grid layout.
    
    Args:
        save_path (str): Path to directory containing result images
        max_cols (int): Maximum number of columns in grid layout. Default: 3
    
    Returns:
        str: Path to saved summary image, or None if no images found
    """
    logger = logging.getLogger()
    
    if not os.path.exists(save_path):
        logger.warning(f"Directory not found: {save_path}")
        return None
    
    # Find all PNG images
    all_images = glob.glob(os.path.join(save_path, '*.png'))
    
    if not all_images:
        logger.info("No images found to visualize")
        return None
    
    # Sort by filename
    all_images.sort()
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Visualizing {len(all_images)} images from training results")
    logger.info(f"{'='*80}")
    
    # Calculate grid layout
    num_images = len(all_images)
    num_cols = min(max_cols, num_images)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    # Create figure
    fig = plt.figure(figsize=(6 * num_cols, 5 * num_rows))
    gs = GridSpec(num_rows, num_cols, figure=fig, hspace=0.3, wspace=0.3)
    
    # Display each image
    for idx, img_path in enumerate(all_images):
        row = idx // num_cols
        col = idx % num_cols
        
        ax = fig.add_subplot(gs[row, col])
        
        try:
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis('off')
            
            # Create title from filename
            img_name = os.path.basename(img_path)
            ax.set_title(img_name, fontsize=10, pad=10)
            
            logger.info(f"  ✓ {img_name}")
            
        except Exception as e:
            logger.warning(f"  ⚠️  Error loading {os.path.basename(img_path)}: {e}")
            ax.text(0.5, 0.5, f'Error loading\n{os.path.basename(img_path)}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    plt.suptitle(f'Training Results: {os.path.basename(save_path)}', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save summary visualization
    summary_path = os.path.join(save_path, 'all_results_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    logger.info(f"\n✓ Saved summary visualization: {summary_path}")
    
    plt.show()
    logger.info(f"{'='*80}\n")
    
    return summary_path


def plot_training_metrics(save_path):
    """
    Plot training and validation metrics from JSON log file.
    Each metric is saved as a separate PNG file.
    
    Args:
        save_path (str): Path to directory containing metrics JSON file
    
    Returns:
        list: List of paths to saved metric plots, or None if no metrics found
    """
    
    logger = logging.getLogger()
    
    # Find metrics JSON file
    json_files = glob.glob(os.path.join(save_path, '*_metrics.json'))
    
    if not json_files:
        logger.warning(f"No metrics JSON file found in {save_path}")
        return None
    
    metrics_file = json_files[0]
    logger.info(f"\n{'='*80}")
    logger.info(f"Plotting training metrics from: {os.path.basename(metrics_file)}")
    logger.info(f"{'='*80}")
    
    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    if not metrics:
        logger.warning("No metrics data found in JSON file")
        return None
    
    # Separate training and validation metrics
    train_metrics = {k: v for k, v in metrics.items() if k.startswith('training/')}
    val_metrics = {k: v for k, v in metrics.items() if k.startswith('validation/')}
    
    # Define metrics to plot
    metrics_config = [
        ('train_average_loss', 'val_average_loss', 'Average Loss', 'average_loss'),
        ('train_class_loss', 'val_class_loss', 'Classification Loss', 'classification_loss'),
        ('train_recon_loss', 'val_recon_loss_nat', 'Reconstruction Loss', 'reconstruction_loss'),
        ('train_KLD', 'val_KLD', 'KL Divergence', 'kl_divergence'),
        ('train_precision@1', 'val_precision@1', 'Precision@1 (%)', 'precision'),
    ]
    
    saved_plots = []
    
    # Plot each metric separately
    for train_key, val_key, title, filename in metrics_config:
        train_full_key = f'training/{train_key}'
        val_full_key = f'validation/{val_key}'
        
        # Create individual figure for this metric
        fig, ax = plt.subplots(figsize=(10, 6))
        
        has_data = False
        
        # Plot training data
        if train_full_key in train_metrics:
            data = train_metrics[train_full_key]
            steps = [d['step'] for d in data]
            values = [d['value'] for d in data]
            ax.plot(steps, values, color='steelblue', label='Training', linewidth=2, alpha=0.8)
            logger.info(f"  ✓ Plotted {train_key}: {len(steps)} points")
            has_data = True
        
        # Plot validation data
        if val_full_key in val_metrics:
            data = val_metrics[val_full_key]
            steps = [d['step'] for d in data]
            values = [d['value'] for d in data]
            ax.plot(steps, values, color='indianred', label='Validation', linewidth=2, alpha=0.8)
            logger.info(f"  ✓ Plotted {val_key}: {len(steps)} points")
            has_data = True
        
        if has_data:
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
            ax.legend(loc='best', fontsize=11, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            
            # Save individual metric plot
            metric_plot_path = os.path.join(save_path, f'metric_{filename}.png')
            plt.savefig(metric_plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"  → Saved: {os.path.basename(metric_plot_path)}")
            saved_plots.append(metric_plot_path)
            
            plt.close(fig)
        else:
            plt.close(fig)
            logger.warning(f"  ⚠️  No data for {title}")
    
    if saved_plots:
        logger.info(f"\n✓ Saved {len(saved_plots)} metric plots")
    logger.info(f"{'='*80}\n")
    
    return saved_plots if saved_plots else None


def visualize_class_distribution(dataset, dataset_name, save_path, split='train'):
    """
    Visualize the class distribution of a dataset with a bar chart.
    Shows the number of samples per class.
    
    Parameters:
        dataset: Dataset object with class_to_idx attribute
        dataset_name (str): Name of the dataset
        save_path (str): Path to save the visualization
        split (str): Dataset split ('train', 'val', 'test', or 'all')
    """
    logger = logging.getLogger()
    
    logger.info(f"Creating class distribution visualization for {dataset_name} ({split} set)...")
    
    # Get the appropriate data loader(s)
    loaders = []
    if split == 'all':
        # Combine all splits
        if hasattr(dataset, 'train_loader'):
            loaders.append(dataset.train_loader)
        if hasattr(dataset, 'val_loader'):
            loaders.append(dataset.val_loader)
        if hasattr(dataset, 'test_loader'):
            loaders.append(dataset.test_loader)
    else:
        if split == 'train':
            loader = dataset.train_loader if hasattr(dataset, 'train_loader') else None
        elif split == 'val':
            loader = dataset.val_loader if hasattr(dataset, 'val_loader') else None
        else:
            loader = dataset.test_loader if hasattr(dataset, 'test_loader') else None
        
        if loader is not None:
            loaders.append(loader)
    
    if not loaders:
        logger.warning(f"No {split} loader found for {dataset_name}")
        return
    
    # Count samples per class
    class_counts = {}
    total_samples = 0
    
    for loader in loaders:
        for _, labels in loader:
            for label in labels:
                label_item = label.item()
                class_counts[label_item] = class_counts.get(label_item, 0) + 1
                total_samples += 1
    
    # Get class names
    if hasattr(dataset, 'class_to_idx') and dataset.class_to_idx:
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
        class_names = [idx_to_class.get(i, f'Class {i}') for i in sorted(class_counts.keys())]
    else:
        class_names = [f'Class {i}' for i in sorted(class_counts.keys())]
    
    counts = [class_counts[i] for i in sorted(class_counts.keys())]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(20, 12))
    
    x_pos = np.arange(len(class_names))
    bars = ax.bar(x_pos, counts, color=sns.color_palette("Set2", len(class_names)), 
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    

    # Add value labels on top of bars 
    max_count = max(counts)
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=legend_font_size-6, fontweight='bold')

    ax.set_ylim(0, max_count * 1.15)
    
    ax.set_xlabel('Class', fontsize=axes_font_size)
    ax.set_ylabel('Number of Samples', fontsize=axes_font_size)
    
    title_text = f'{dataset_name} - Class Distribution ({split.capitalize()} Set)'
    if split == 'all':
        title_text = f'{dataset_name} - Class Distribution (All Data)\nTotal: {total_samples} samples'
    ax.set_title(title_text, fontsize=title_font_size)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=ticks_font_size-8)
    ax.tick_params(axis='y', labelsize=ticks_font_size)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_file = os.path.join(save_path, f'{dataset_name}_class_distribution_{split}.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"✓ Class distribution saved: {save_file}")
    logger.info(f"  Total samples: {total_samples}")
    logger.info(f"  Number of classes: {len(class_counts)}")
    logger.info(f"  Samples per class: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.1f}")
    
    return save_file


def visualize_openset_2d_embedding(known_embeddings, unknown_embeddings_dict, 
                                   known_dataset_name, save_path, num_classes):
    """
    Visualize 2D latent embeddings for open set recognition evaluation.
    Shows known classes (trained on) vs unknown classes (never seen).
    
    Parameters:
        known_embeddings (list): List of tensors containing z values for each known class
        unknown_embeddings_dict (dict): Dictionary with dataset_name -> list of z tensors
        known_dataset_name (str): Name of the known/trained dataset
        save_path (str): Path to save the visualization
        num_classes (int): Number of known classes
    """
    logger = logging.getLogger()
    
    logger.info(f"Creating 2D open-set embedding visualization...")
    
    # Prepare known class data and check dimensionality
    known_z_list = []
    known_labels = []
    latent_dim = None
    
    for class_idx, z_tensor in enumerate(known_embeddings):
        if isinstance(z_tensor, torch.Tensor) and z_tensor.numel() > 0:
            if latent_dim is None:
                latent_dim = z_tensor.size(1)
            known_z_list.append(z_tensor.cpu().numpy())
            known_labels.extend([class_idx] * z_tensor.size(0))
    
    # Check if embeddings are 2D
    if latent_dim != 2:
        logger.warning(f"Skipping open-set 2D visualization - latent dimension is {latent_dim}, not 2D")
        return
    
    if len(known_z_list) == 0:
        logger.warning("No valid known embeddings to visualize")
        return
    
    known_z = np.vstack(known_z_list)
    known_labels = np.array(known_labels)
    
    # Prepare unknown data
    all_unknown_z_list = []
    all_unknown_labels = []
    unknown_dataset_names = []
    
    for idx, (dataset_name, z_list) in enumerate(unknown_embeddings_dict.items()):
        unknown_z_list = []
        for z_tensor in z_list:
            if isinstance(z_tensor, list):
                for sub_tensor in z_tensor:
                    if isinstance(sub_tensor, torch.Tensor) and sub_tensor.numel() > 0:
                        unknown_z_list.append(sub_tensor.cpu().numpy())
            elif isinstance(z_tensor, torch.Tensor) and z_tensor.numel() > 0:
                unknown_z_list.append(z_tensor.cpu().numpy())
        
        if len(unknown_z_list) > 0:
            unknown_z = np.vstack(unknown_z_list)
            all_unknown_z_list.append(unknown_z)
            all_unknown_labels.extend([idx] * len(unknown_z))
            unknown_dataset_names.append(dataset_name)
            logger.info(f"  Collected {len(unknown_z)} points from {dataset_name}")
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(60, 18))
    
    # Color palettes
    known_colors = sns.color_palette("tab10", num_classes)
    unknown_colors = sns.color_palette("Set1", len(unknown_dataset_names))
    
    # Plot 1: Close-set only (Known classes)
    ax1 = plt.subplot(1, 3, 1)
    for class_idx in range(num_classes):
        mask = known_labels == class_idx
        if np.sum(mask) > 0:
            ax1.scatter(known_z[mask, 0], known_z[mask, 1], 
                       c=[known_colors[class_idx]], 
                       s=80, alpha=0.7, edgecolors='black', linewidth=0.5,
                       label=f'Class {class_idx}')
    
    ax1.set_xlabel('z dimension 1', fontsize=axes_font_size)
    ax1.set_ylabel('z dimension 2', fontsize=axes_font_size)
    ax1.set_title(f'Close-set: {known_dataset_name}\n(Known Classes Only)', fontsize=title_font_size)
    ax1.tick_params(labelsize=ticks_font_size)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=legend_font_size-6, loc='upper right', framealpha=0.9, ncol=2)
    
    # Plot 2: Open-set only (Unknown datasets)
    ax2 = plt.subplot(1, 3, 2)
    if len(all_unknown_z_list) > 0:
        all_unknown_z = np.vstack(all_unknown_z_list)
        all_unknown_labels = np.array(all_unknown_labels)
        
        for idx, dataset_name in enumerate(unknown_dataset_names):
            mask = all_unknown_labels == idx
            if np.sum(mask) > 0:
                ax2.scatter(all_unknown_z[mask, 0], all_unknown_z[mask, 1],
                           c=[unknown_colors[idx]], 
                           s=80, alpha=0.7, marker='x', linewidths=2.5,
                           label=dataset_name)
    
    ax2.set_xlabel('z dimension 1', fontsize=axes_font_size)
    ax2.set_ylabel('z dimension 2', fontsize=axes_font_size)
    ax2.set_title(f'Open-set: Unknown Datasets\n(Never Seen During Training)', fontsize=title_font_size)
    ax2.tick_params(labelsize=ticks_font_size)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=legend_font_size, loc='upper right', framealpha=0.9)
    
    # Plot 3: Combined (Close-set + Open-set)
    ax3 = plt.subplot(1, 3, 3)
    
    # Plot known classes with reduced opacity
    for class_idx in range(num_classes):
        mask = known_labels == class_idx
        if np.sum(mask) > 0:
            ax3.scatter(known_z[mask, 0], known_z[mask, 1], 
                       c=[known_colors[class_idx]], 
                       s=40, alpha=0.4, edgecolors='none')
    
    # Plot unknown datasets on top
    if len(all_unknown_z_list) > 0:
        for idx, dataset_name in enumerate(unknown_dataset_names):
            mask = all_unknown_labels == idx
            if np.sum(mask) > 0:
                ax3.scatter(all_unknown_z[mask, 0], all_unknown_z[mask, 1],
                           c=[unknown_colors[idx]], 
                           s=120, alpha=0.8, marker='x', linewidths=3,
                           label=f'{dataset_name} (Unknown)')
    
    # Add legend for known classes
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=known_colors[i], alpha=0.4, label=f'Known Class {i}') 
                      for i in range(num_classes)]
    
    ax3.set_xlabel('z dimension 1', fontsize=axes_font_size)
    ax3.set_ylabel('z dimension 2', fontsize=axes_font_size)
    ax3.set_title(f'Combined: Close-set vs Open-set\n{known_dataset_name} (background) vs Unknown (foreground)', 
                 fontsize=title_font_size)
    ax3.tick_params(labelsize=ticks_font_size)
    ax3.grid(True, alpha=0.3)
    
    # Dual legend
    first_legend = ax3.legend(handles=legend_elements, fontsize=legend_font_size-8, 
                             loc='upper left', framealpha=0.9, title='Known Classes', ncol=2)
    ax3.add_artist(first_legend)
    ax3.legend(fontsize=legend_font_size-6, loc='upper right', framealpha=0.9, title='Unknown Datasets')
    
    plt.tight_layout()
    
    # Save figure
    save_file = os.path.join(save_path, f'{known_dataset_name}_openset_2d_embedding.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"✓ Open-set 2D embedding saved: {save_file}")
    logger.info(f"  - Close-set (known): {len(known_z)} samples from {num_classes} classes")
    if len(all_unknown_z_list) > 0:
        logger.info(f"  - Open-set (unknown): {len(all_unknown_z)} samples from {len(unknown_dataset_names)} dataset(s)")
    
    return save_file


def visualize_openset_confusion_matrix(known_eval_dict, openset_eval_dicts, 
                                      outlier_probs_correct, openset_outlier_probs_dict,
                                      evt_threshold, entropy_threshold,
                                      known_dataset_name, num_classes, save_path):
    """
    Visualize confusion matrix for Open Set Recognition.
    Shows:
    - Known classes classification (diagonal = correct)
    - Unknown rejection (last column/row)
    
    Parameters:
        known_eval_dict (dict): Evaluation results for known dataset
        openset_eval_dicts (dict): Dict of evaluation results for openset datasets
        outlier_probs_correct (list): Weibull outlier probabilities for known dataset
        openset_outlier_probs_dict (dict): Dict of Weibull outlier probs for openset datasets
        evt_threshold (float): Threshold for EVT method
        entropy_threshold (float): Threshold for Entropy method
        known_dataset_name (str): Name of known dataset
        num_classes (int): Number of known classes
        save_path (str): Path to save visualization
    """
    logger = logging.getLogger()
    
    logger.info(f"Creating Open-Set Recognition confusion matrices...")
    
    # Create confusion matrix for EVT method
    # (Entropy method would require tracking predicted class for each sample, 
    # which is not readily available in current data structure)
    
    matrix_size = num_classes + 1
    confusion_matrix = np.zeros((matrix_size, matrix_size))
    
    # Process known dataset - EVT method
    for class_idx in range(num_classes):
        # Correctly classified samples
        outlier_probs = outlier_probs_correct[class_idx]
        for i in range(len(outlier_probs)):
            if outlier_probs[i] > evt_threshold:
                # Rejected as unknown (False Positive)
                confusion_matrix[class_idx, num_classes] += 1
            else:
                # Accepted as known (True Positive for closed-set)
                confusion_matrix[class_idx, class_idx] += 1
    
    # Process openset datasets - EVT method
    for openset_name, openset_eval_dict in openset_eval_dicts.items():
        for class_idx in range(num_classes):
            outlier_probs = openset_outlier_probs_dict[openset_name][class_idx]
            for i in range(len(outlier_probs)):
                if outlier_probs[i] > evt_threshold:
                    # Correctly rejected as unknown (True Positive for open-set)
                    confusion_matrix[num_classes, num_classes] += 1
                else:
                    # Misclassified as known class (False Negative for open-set)
                    confusion_matrix[num_classes, class_idx] += 1
    
    # Calculate percentages for better visualization
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    confusion_matrix_pct = (confusion_matrix / row_sums) * 100
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40, 18))
    
    # Class names
    class_names = [f'Class {i}' for i in range(num_classes)] + ['Unknown']
    
    # Plot 1: Raw counts
    sns.heatmap(confusion_matrix, annot=True, fmt='.0f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Count'}, 
                linewidths=1, linecolor='gray')
    
    ax1.set_xlabel('Predicted Label', fontsize=axes_font_size)
    ax1.set_ylabel('True Label', fontsize=axes_font_size)
    ax1.set_title(f'OSR Confusion Matrix - EVT Method\n{known_dataset_name} vs Unknown\n(Raw Counts)', 
                 fontsize=title_font_size)
    ax1.tick_params(labelsize=ticks_font_size-8, rotation=45)
    
    # Plot 2: Percentages
    sns.heatmap(confusion_matrix_pct, annot=True, fmt='.1f', cmap='RdYlGn', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Percentage (%)'}, 
                linewidths=1, linecolor='gray', vmin=0, vmax=100)
    
    ax2.set_xlabel('Predicted Label', fontsize=axes_font_size)
    ax2.set_ylabel('True Label', fontsize=axes_font_size)
    ax2.set_title(f'OSR Confusion Matrix - EVT Method\n{known_dataset_name} vs Unknown\n(Percentages)', 
                 fontsize=title_font_size)
    ax2.tick_params(labelsize=ticks_font_size-8, rotation=45)
    

    # Add statistics
    known_correct = np.diag(confusion_matrix[:num_classes, :num_classes]).sum()
    known_total = confusion_matrix[:num_classes, :].sum()
    unknown_rejected = confusion_matrix[num_classes, num_classes]
    unknown_total = confusion_matrix[num_classes, :].sum()

    known_acc = (known_correct / known_total * 100) if known_total > 0 else 0
    unknown_recall = (unknown_rejected / unknown_total * 100) if unknown_total > 0 else 0
    fpr = ((confusion_matrix[:num_classes, num_classes].sum()) / known_total * 100) if known_total > 0 else 0

    # Metrics for binary close/open (for reference)
    y_true = []
    y_pred = []
    # Known set (close set):
    for class_idx in range(num_classes):
        outlier_probs = outlier_probs_correct[class_idx]
        for i in range(len(outlier_probs)):
            y_true.append(0)  # true close
            if outlier_probs[i] > evt_threshold:
                y_pred.append(1)  # predicted open (FP)
            else:
                y_pred.append(0)  # predicted close (TP)
    # Open set:
    for openset_name, openset_eval_dict in openset_eval_dicts.items():
        for class_idx in range(num_classes):
            outlier_probs = openset_outlier_probs_dict[openset_name][class_idx]
            for i in range(len(outlier_probs)):
                y_true.append(1)  # true open
                if outlier_probs[i] > evt_threshold:
                    y_pred.append(1)  # predicted open (TN)
                else:
                    y_pred.append(0)  # predicted close (FN)

    acc = accuracy_score(y_true, y_pred)
    recall_open = recall_score(y_true, y_pred, pos_label=1)
    precision_open = precision_score(y_true, y_pred, pos_label=1)
    f1_open = f1_score(y_true, y_pred, pos_label=1)
    recall_close = recall_score(y_true, y_pred, pos_label=0)
    precision_close = precision_score(y_true, y_pred, pos_label=0)
    f1_close = f1_score(y_true, y_pred, pos_label=0)


    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Save figure
    save_file = os.path.join(save_path, f'{known_dataset_name}_OSR_confusion_EVT.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"✓ OSR confusion matrix (EVT) saved: {save_file}")
    logger.info(f"  Known accuracy (per-class): {known_acc:.2f}%")
    logger.info(f"  Unknown detection (TPR): {unknown_recall:.2f}%")
    logger.info(f"  False Positive Rate: {fpr:.2f}%")
    logger.info(f"  [Binary metrics]")
    logger.info(f"    Accuracy: {acc:.4f}")
    logger.info(f"    Recall (open): {recall_open:.4f}, Precision (open): {precision_open:.4f}, F1 (open): {f1_open:.4f}")
    logger.info(f"    Recall (close): {recall_close:.4f}, Precision (close): {precision_close:.4f}, F1 (close): {f1_close:.4f}")


def visualize_openset_binary_confusion_matrix(known_eval_dict, openset_eval_dicts, 
                                             outlier_probs_correct, openset_outlier_probs_dict,
                                             evt_threshold, known_dataset_name, num_classes, save_path):
    """
    Visualize 2x2 confusion matrix for Open Set Recognition (close vs open set).
    Logs accuracy, recall, precision, f1, etc.
    
    Parameters:
        known_eval_dict (dict): Evaluation results for known dataset
        openset_eval_dicts (dict): Dict of evaluation results for openset datasets
        outlier_probs_correct (list): Weibull outlier probabilities for known dataset
        openset_outlier_probs_dict (dict): Dict of Weibull outlier probs for openset datasets
        evt_threshold (float): Threshold for EVT method
        known_dataset_name (str): Name of known dataset
        num_classes (int): Number of known classes
        save_path (str): Path to save visualization
    """
    logger = logging.getLogger()
    logger.info(f"[visualize_openset_binary_confusion_matrix] Drawing 2x2 close/open confusion matrix...")

    # y_true: 0=close, 1=open; y_pred: 0=close, 1=open
    y_true = []
    y_pred = []

    # Known set (close set):
    for class_idx in range(num_classes):
        outlier_probs = outlier_probs_correct[class_idx]
        for i in range(len(outlier_probs)):
            y_true.append(0)  # true close
            if outlier_probs[i] > evt_threshold:
                y_pred.append(1)  # predicted open (FP)
            else:
                y_pred.append(0)  # predicted close (TP)

    # Open set:
    for openset_name, openset_eval_dict in openset_eval_dicts.items():
        for class_idx in range(num_classes):
            outlier_probs = openset_outlier_probs_dict[openset_name][class_idx]
            for i in range(len(outlier_probs)):
                y_true.append(1)  # true open
                if outlier_probs[i] > evt_threshold:
                    y_pred.append(1)  # predicted open (TN)
                else:
                    y_pred.append(0)  # predicted close (FN)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    # cm: rows = true, cols = pred
    # [[TP, FP],
    #  [FN, TN]]

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, pos_label=1)
    precision = precision_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)

    logger.info(f"[visualize_openset_binary_confusion_matrix] Confusion matrix (rows=true, cols=pred):\n{cm}")
    logger.info(f"[visualize_openset_binary_confusion_matrix] Accuracy: {acc:.4f}")
    logger.info(f"[visualize_openset_binary_confusion_matrix] Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Close', 'Open'], yticklabels=['Close', 'Open'],
                ax=ax, cbar_kws={'label': 'Count'}, linewidths=1, linecolor='gray')
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_ylabel('True Label', fontsize=16)
    ax.set_title(f'OSR 2x2 Confusion Matrix\n{known_dataset_name} (Close vs Open)', fontsize=18)

    # Add metrics as text box
    stats_text = (f"Accuracy: {acc:.2%}\n"
                  f"Recall: {recall:.2%}\n"
                  f"Precision: {precision:.2%}\n"
                  f"F1: {f1:.2%}")
    plt.gcf().text(0.99, 0.01, stats_text, fontsize=13, ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save_file = os.path.join(save_path, f'{known_dataset_name}_OSR_confusion_EVT_binary.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Confusion matrix:\n{cm.tolist()}")
    logger.info(f"✓ OSR 2x2 confusion matrix (EVT) saved: {save_file}")
    logger.info(f"  - Accuracy: {acc:.4f}")
    logger.info(f"  - Recall (open): {recall:.4f}, Precision (open): {precision:.4f}, F1 (open): {f1:.4f}")


