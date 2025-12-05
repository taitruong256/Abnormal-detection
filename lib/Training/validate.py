import time
import math
import torch
import logging
import torch.nn.functional as F
from tqdm import tqdm
from lib.Utility.metrics import AverageMeter
from lib.Utility.metrics import ConfusionMeter
from lib.Utility.metrics import accuracy
from lib.Utility.visualization import visualize_confusion, visualize_confusion_heatmap
from lib.Utility.visualization import visualize_image_grid
from lib.Utility.visualization import visualize_dataset_in_2d_embedding


def validate(Dataset, model, criterion, epoch, metrics_logger, device, save_path, args):
    """
    Evaluates/validates the model

    Parameters:
        Dataset (torch.utils.data.Dataset): The dataset
        model (torch.nn.module): Model to be evaluated/validated
        criterion (torch.nn.criterion): Loss function
        epoch (int): Epoch counter
        metrics_logger: Logger for tracking metrics
        device (str): device name where data is transferred to
        save_path (str): path to save data to
        args (dict): Dictionary of (command line) arguments.
            Needs to contain print_freq (int), epochs (int), incremental_data (bool), autoregression (bool),
            visualization_epoch (int), num_base_tasks (int), num_increment_tasks (int) and
            patch_size (int).

    Returns:
        float: top1 precision/accuracy
        float: average loss
    """

    logger = logging.getLogger()

    # initialize average meters to accumulate values
    class_losses = AverageMeter()
    recon_losses_nat = AverageMeter()
    kld_losses = AverageMeter()
    losses = AverageMeter()

    batch_time = AverageMeter()
    top1 = AverageMeter()

    # confusion matrix
    confusion = ConfusionMeter(model.num_classes, normalized=True)

    # switch to evaluate mode
    model.eval()

    end = time.time()

    # evaluate the entire validation dataset
    with torch.no_grad():
        for i, (inp, target) in enumerate(tqdm(Dataset.val_loader, desc=f"Epoch {epoch+1} Validation")):
            inp = inp.to(device)
            target = target.to(device)

            recon_target = inp
            class_target = target

            # compute output
            class_samples, recon_samples, mu, std = model(inp)

            # compute loss
            class_loss, recon_loss, kld_loss = criterion(class_samples, class_target, recon_samples, recon_target, mu,
                                                         std, device, args)

            # take mean to compute accuracy
            # (does nothing if there isn't more than 1 sample per input other than removing dummy dimension)
            class_output = torch.mean(class_samples, dim=0)
            recon_output = torch.mean(recon_samples, dim=0)

            # measure accuracy, record loss, fill confusion matrix
            prec1 = accuracy(class_output, target)[0]
            top1.update(prec1.item(), inp.size(0))
            confusion.add(class_output.data, target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # for autoregressive models generate reconstructions by sequential sampling from the
            # multinomial distribution (Reminder: the original output is a 255 way Softmax as PixelVAEs are posed as a
            # classification problem). This serves two purposes: visualization of reconstructions and computation of
            # a reconstruction loss in nats using a BCE loss, comparable to that of a regular VAE.
            recon_target = inp
         

            # If not autoregressive simply apply the Sigmoid and visualize
            recon = torch.sigmoid(recon_output)
            if (i == (len(Dataset.val_loader) - 1)) and (epoch % args.visualization_epoch == 0) and (epoch > 0):
                visualize_image_grid(recon, None, epoch + 1, 'reconstruction_snapshot', save_path)

            # update the respective loss values. To be consistent with values reported in the literature we scale
            # our normalized losses back to un-normalized values.
            # For the KLD this also means the reported loss is not scaled by beta, to allow for a fair comparison
            # across potential weighting terms.
            class_losses.update(class_loss.item() * model.num_classes, inp.size(0))
            kld_losses.update(kld_loss.item() * model.latent_dim, inp.size(0))
            recon_losses_nat.update(recon_loss.item() * inp.size()[1:].numel(), inp.size(0))
            losses.update((class_loss + recon_loss + kld_loss).item(), inp.size(0))

            # If we are at the end of validation, create one mini-batch of example generations. Only do this every
            # other epoch specified by visualization_epoch to avoid generation of lots of images and computationally
            # expensive calculations of the autoregressive model's generation.
            if i == (len(Dataset.val_loader) - 1) and epoch % args.visualization_epoch == 0 and (epoch > 0):
                # generation
                gen = model.generate()
                visualize_image_grid(gen, None, epoch + 1, 'generation_snapshot', save_path)

            # Print progress
            if i % args.print_freq == 0:
                logger.info('Validate: [{0}][{1}/{2}]\t' 
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Class Loss {cl_loss.val:.4f} ({cl_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Recon Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f})\t'
                      'KL {KLD_loss.val:.4f} ({KLD_loss.avg:.4f})'.format(
                       epoch+1, i, len(Dataset.val_loader), batch_time=batch_time, loss=losses, cl_loss=class_losses,
                       top1=top1, recon_loss=recon_losses_nat, KLD_loss=kld_losses))

    # TensorBoard summary logging
    if metrics_logger:
        metrics_logger.add_scalar('validation/val_precision@1', top1.avg, epoch)
        metrics_logger.add_scalar('validation/val_average_loss', losses.avg, epoch)
        metrics_logger.add_scalar('validation/val_class_loss', class_losses.avg, epoch)
        metrics_logger.add_scalar('validation/val_recon_loss_nat', recon_losses_nat.avg, epoch)
        metrics_logger.add_scalar('validation/val_KLD', kld_losses.avg, epoch)

    logger.info(' * Validation: Loss {loss.avg:.5f} Prec@1 {top1.avg:.3f}'.format(loss=losses, top1=top1))

    # At the end of training isolated, or at the end of every task visualize the confusion matrix
    if (epoch + 1) % args.epochs == 0 and epoch > 0:
        # visualize the confusion matrix (original)
        visualize_confusion(None, epoch + 1, confusion.value(), Dataset.class_to_idx, save_path)
        # visualize the confusion matrix (heatmap with actual counts - known classes only)
        known_classes = Dataset.known if hasattr(Dataset, 'known') else None
        visualize_confusion_heatmap(None, epoch + 1, confusion.value_counts(), Dataset.class_to_idx, save_path, known_classes)
        
        # Save task-specific confusion matrices for continual learning
        if args.incremental_data:
            task_num = (epoch + 1) // args.epochs
            num_classes = model.num_classes
            logger.info(f'\nSaving Task {task_num} confusion matrices ({num_classes} classes)...')
            visualize_confusion(None, f"task{task_num}_{num_classes}classes", confusion.value(), Dataset.class_to_idx, save_path)
            visualize_confusion_heatmap(None, f"task{task_num}_{num_classes}classes", confusion.value_counts(), Dataset.class_to_idx, save_path, known_classes)
            logger.info(f'  ✓ Task {task_num} confusion matrices saved')
    
    # Visualize 2D latent space if latent_dim is 2
    if hasattr(model, 'latent_dim') and model.latent_dim == 2:
        if (epoch + 1) % args.visualization_epoch == 0 or (epoch + 1) == args.epochs or (args.incremental_data and (epoch + 1) % args.epochs == 0 and epoch > 0):
            logger.info(f'Creating 2D latent space visualization (epoch {epoch + 1})...')
            
            # Encode entire validation dataset and organize by class
            # encoding_list[class_id] = tensor of all encodings for that class
            num_classes = model.num_classes
            encoding_by_class = [[] for _ in range(num_classes)]
            
            with torch.no_grad():
                for inp, target in Dataset.val_loader:
                    inp = inp.to(device)
                    target = target.to(device)
                    
                    # Get latent encoding (mu)
                    _, _, mu, _ = model(inp)
                    
                    # Group encodings by class
                    for i in range(mu.size(0)):
                        class_id = target[i].item()
                        encoding_by_class[class_id].append(mu[i].cpu())
            
            # Convert lists to tensors
            encoding_list = []
            for class_encodings in encoding_by_class:
                if len(class_encodings) > 0:
                    encoding_list.append(torch.stack(class_encodings))
                else:
                    # Empty class - add empty tensor
                    encoding_list.append(torch.empty(0, 2))
            
            # Visualize 2D embedding
            dataset_name = args.dataset if hasattr(args, 'dataset') else 'Dataset'
            visualize_dataset_in_2d_embedding(
                writer=None,
                encoding_list=encoding_list, 
                dataset_name=dataset_name,
                save_path=save_path,
                task=epoch + 1
            )
            logger.info(f'✓ 2D latent space visualization saved for epoch {epoch + 1}')
            
            # Save task-specific 2D embedding for continual learning
            if args.incremental_data and (epoch + 1) % args.epochs == 0 and epoch > 0:
                task_num = (epoch + 1) // args.epochs
                num_classes = model.num_classes
                
                visualize_dataset_in_2d_embedding(
                    writer=None,
                    encoding_list=encoding_list, 
                    dataset_name=f"{dataset_name}_task{task_num}_{num_classes}classes",
                    save_path=save_path,
                    task=task_num
                )
                logger.info(f'  ✓ Task {task_num} 2D embedding saved')

    return top1.avg, losses.avg