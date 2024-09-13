import math
import sys
import os
from typing import Iterable

import util.misc as misc
import util.lr_sched as lr_sched
import util.transforms as T
from util.pytorch_ssim import *


def train_one_epochx(model: torch.nn.Module,
                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
                     device: torch.device, epoch: int, loss_scaler,
                     log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train Epoch {}'.format(epoch)
    print_freq = 200

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, d in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        data, label, mask, label_mask = d[0], d[1], d[2], d[3]
        meta_data, meta_data_mask = d[5], d[6]

        data = data.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        label_mask = label_mask.to(device, non_blocking=True)
        meta_data = meta_data.to(device, non_blocking=True)
        meta_data_mask = meta_data_mask.to(device, non_blocking=True)

        if args.remove_shell > 0:
            assert (mask[:, :, :, :, args.remove_shell + 2] == True).all()

        if (data_iter_step + 1) % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        noise_list = []
        if args.noise_injection:
            for param in model.parameters():
                noise = torch.randn(param.data.size()).to(device) * args.noise_injection
                param.data += noise
                noise_list.append(noise)

        with torch.cuda.amp.autocast(enabled=False):
            loss, pred = model(data, label, mask, label_mask, meta_data, meta_data_mask)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        if args.noise_injection:
            for param in model.parameters():
                param.data -= noise_list[0].to(device)
                noise_list = noise_list[1:]

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evalx(model: torch.nn.Module, data_loader: Iterable, device: torch.device, label_min, label_max, sat, args, name='', save=True, prefix=''):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Eval'
    print_freq = 100

    saving_path = os.path.join(args.output_dir, name)

    label_tensor, pred_tensor = [], []  # store denormalized predcition & gt in tensor
    label_mask_list, label_ind_list = [], []

    for data_iter_step, d in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        data, label, mask, label_mask, label_ind = d[0], d[1], d[2], d[3], d[4]
        meta_data, meta_data_mask = d[5], d[6]

        data = data.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        label_mask = label_mask.to(device, non_blocking=True)
        meta_data = meta_data.to(device, non_blocking=True)
        meta_data_mask = meta_data_mask.to(device, non_blocking=True)

        if args.remove_shell > 0:
            assert (mask[:, :, :, :, args.remove_shell + 2] == True).all()

        if args.model == 'FluexNet_single':
            label = label[:, :, :, args.target_shell].unsqueeze(-1)
            label_mask = label_mask[:, :, :, args.target_shell].unsqueeze(-1)

        with torch.cuda.amp.autocast(enabled=False):
            loss, pred = model(data, label, mask, label_mask, meta_data, meta_data_mask)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        metric_logger.update(loss=loss_value)

        label_np = T.tensor_denormalize(label, label_min, label_max, exp=args.log, norm=args.norm, log10=args.log10)
        label_pred_np = T.tensor_denormalize(pred, label_min, label_max, exp=args.log, norm=args.norm, log10=args.log10)
        label_tensor.append(label_np.detach().cpu())
        pred_tensor.append(label_pred_np.detach().cpu())

        label_mask_list.append(label_mask.detach().cpu())

        label_ind_list.append(label_ind.numpy())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    label_tensor, pred_tensor, label_mask_list = torch.cat(label_tensor), torch.cat(pred_tensor), torch.cat(label_mask_list)  # mask 0 for valid value
    print(label_tensor.shape, pred_tensor.shape)

    label = T.sample_to_whole(label_tensor.squeeze(1))
    pred = T.sample_to_whole(pred_tensor.squeeze(1))
    label_mask = T.sample_to_whole(label_mask_list.squeeze(1))

    l1 = misc.l1_loss
    l2 = misc.l2_loss
    r2 = misc.r2_score
    # ssim_loss = SSIM(window_size=11)

    print('* Loss in Original Domain:')
    print(f'MAE: {l1(label.squeeze(), pred.squeeze(), label_mask.bool().squeeze())}')
    print(f'MSE: {l2(label.squeeze(), pred.squeeze(), label_mask.bool().squeeze())}')
    print(f'R2 Score: {r2(label.squeeze(), pred.squeeze(), label_mask.bool().squeeze())}')
    print()

    label = T.log_transform_tensor(label)
    pred = T.log_transform_tensor(pred)
    print('* Loss in Log Domain:')
    print(f'MAE: {l1(label.squeeze(), pred.squeeze(), label_mask.bool().squeeze())}')
    print(f'MSE: {l2(label.squeeze(), pred.squeeze(), label_mask.bool().squeeze())}')
    print(f'R2 Score: {r2(label.squeeze(), pred.squeeze(), label_mask.bool().squeeze())}')
    print()
    val_loss = r2(label.squeeze(), pred.squeeze(), label_mask.bool().squeeze())

    # label = T.minmax_normalize_tensor(label, sat['min'].item(), sat['max'].item())
    # pred = T.minmax_normalize_tensor(pred, sat['min'].item(), sat['max'].item())
    #
    # print('* Loss in Min-Max Normalized (-1,1) Domain:')
    # print(f'MAE: {l1(label.squeeze(), pred.squeeze(), label_mask.bool().squeeze())}')
    # print(f'MSE: {l2(label.squeeze(), pred.squeeze(), label_mask.bool().squeeze())}')
    # print(f'R2 Score: {r2(label.squeeze(), pred.squeeze(), label_mask.bool().squeeze())}')
    # print(f'SSIM: {ssim_loss((label*label_mask).unsqueeze(0).unsqueeze(0) / 2 + 0.5, (pred*label_mask).unsqueeze(0).unsqueeze(0) / 2 + 0.5)}')  # (-1, 1) to (0, 1)

    label, label_pred, label_mask = label_tensor.cpu().numpy(), pred_tensor.cpu().numpy(), 1 - label_mask_list.cpu().numpy()  # mask 1 for valid value
    label_ind = np.concatenate(label_ind_list)

    # prefix = 'train_' if trainset else ''

    if args.save_window:
        np.save(os.path.join(saving_path, prefix + 'label.npy'), label)
        np.save(os.path.join(saving_path, prefix + 'pred.npy'), label_pred)
        np.save(os.path.join(saving_path, prefix + 'label_mask_list.npy'), label_mask)
        np.save(os.path.join(saving_path, prefix + 'label_ind.npy'), label_ind)

    return val_loss, label_pred  # {k: meter.global_avg for k, meter in metric_logger.meters.items()}
