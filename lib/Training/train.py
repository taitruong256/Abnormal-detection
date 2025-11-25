import time
import torch
import logging
from lib.Utility.metrics import AverageMeter
from lib.Utility.metrics import accuracy


def train(Dataset, model, criterion, epoch, optimizer, metrics_logger, device, args):
    """
    Trains/updates the model for one epoch on the training dataset.

    Parameters:
        Dataset (torch.utils.data.Dataset): The dataset
        model (torch.nn.module): Model to be trained
        criterion (torch.nn.criterion): Loss function
        epoch (int): Continuous epoch counter
        optimizer (torch.optim.optimizer): optimizer instance like SGD or Adam
        metrics_logger: Logger for tracking metrics
        device (str): device name where data is transferred to
        args (dict): Dictionary of (command line) arguments.
            Needs to contain print_freq (int) and log_weights (bool).
    """

    logger = logging.getLogger()

    # Create instances to accumulate losses etc.
    class_losses = AverageMeter()
    recon_losses = AverageMeter()

    kld_losses = AverageMeter()

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    # train
    for i, (inp, target) in enumerate(Dataset.train_loader):

        inp = inp.to(device)
        target = target.to(device)

        recon_target = inp
        class_target = target


        # measure data loading time
        data_time.update(time.time() - end)

        # compute model forward
        class_samples, recon_samples, mu, std = model(inp)

        # calculate loss
        class_loss, recon_loss, kld_loss = criterion(class_samples, class_target, recon_samples, recon_target, mu, std,
                                                        device, args)

        # add the individual loss components together and weight the KL term.
        loss = class_loss + recon_loss + args.var_beta * kld_loss

        # take mean to compute accuracy. Note if variational samples are 1 this only gets rid of a dummy dimension.
        class_output = torch.mean(class_samples, dim=0)

        # record precision/accuracy and losses
        losses.update((class_loss + recon_loss + kld_loss).item(), inp.size(0))
        class_losses.update(class_loss.item(), inp.size(0))
        recon_losses.update(recon_loss.item(), inp.size(0))
        kld_losses.update(kld_loss.item(), inp.size(0))

        prec1 = accuracy(class_output, target)[0]
        top1.update(prec1.item(), inp.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print progress
        if i % args.print_freq == 0:
            logger.info('Training: [{0}][{1}/{2}]\t' 
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Class Loss {cl_loss.val:.4f} ({cl_loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Recon Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f})\t'
                    'KL {KLD_loss.val:.4f} ({KLD_loss.avg:.4f})'.format(
                    epoch+1, i, len(Dataset.train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, cl_loss=class_losses, top1=top1,
                    recon_loss=recon_losses, KLD_loss=kld_losses))

    # Log metrics to file
    if metrics_logger:
        metrics_logger.add_scalar('training/train_precision@1', top1.avg, epoch)
        metrics_logger.add_scalar('training/train_average_loss', losses.avg, epoch)
        metrics_logger.add_scalar('training/train_KLD', kld_losses.avg, epoch)
        metrics_logger.add_scalar('training/train_class_loss', class_losses.avg, epoch)
        metrics_logger.add_scalar('training/train_recon_loss', recon_losses.avg, epoch)

    logger.info(' * Train: Loss {loss.avg:.5f} Prec@1 {top1.avg:.3f}'.format(loss=losses, top1=top1))