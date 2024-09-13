import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path
import copy

import errno

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import inf
import numpy as np
from scipy.interpolate import UnivariateSpline


from torchvision.models import vgg16, VGG16_Weights


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable) - 1, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable) - 1, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ and args.world_size > 1:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)

    # os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    # os.environ['NCCL_IB_DISABLE'] = '1'

    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, name):
    output_dir = Path(args.output_dir)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
            print("Resume checkpoint %s" % checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % name, client_state=client_state)


def load_model(args, model_without_ddp, optimizer, loss_scaler, phase):
    args.start_epoch = 0
    if args.resume and phase in args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if not (hasattr(args, 'start') and args.start) and 'optimizer' in checkpoint and 'epoch' in checkpoint and not (
                hasattr(args, 'eval') and args.eval):  # and args.dataset in args.resume
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")
    else:
        args.start_epoch = 0


def load_model_inv(args, model_without_ddp, optimizer, loss_scaler):
    args.start_epoch = 0

    if args.resume.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.resume, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])
    print("Resume checkpoint %s" % args.resume)
    if not (hasattr(args, 'start') and args.start) and 'optimizer' in checkpoint and 'epoch' in checkpoint and not (
            hasattr(args, 'eval') and args.eval):  # and args.dataset in args.resume
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        print("With optim & sched!")


def load_model_part1(args, model_without_ddp, part_name):
    args.start_epoch = 0

    resume_x = args.output_dir + f'/{args.resume_x}'

    print("Load From: ", resume_x)
    state_dict_s = torch.load(resume_x, map_location='cpu')['model']
    state_dict_s_v2 = copy.deepcopy(state_dict_s)

    for key in state_dict_s:
        state_dict_s_v2[f'{part_name}.' + key] = state_dict_s_v2.pop(key)

    msg = model_without_ddp.load_state_dict(state_dict_s_v2, strict=False)
    print(msg)


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def pad_both(data, geo_data=False):
    new_data = np.full((data.shape[0], 37), np.nan)
    if geo_data:
        new_data[:, 31:] = data[:, 17:23]
    else:
        new_data[:, 14:31] = data[:, :17]
    return new_data


def pad_both2(data, geo_data=False):
    new_data = np.full((data.shape[0], 37), np.nan)
    new_data[:, 14:] = data[:, :23]
    return new_data


def interpolate_1D(data, kind='linear', k=2, spoint=3):
    """Interpolate along the last dimension of 2D data.

    Parameters:
    - data (numpy.ndarray): Input 2D array with nan values.
    - kind (str): Type of interpolation ('linear', 'quadratic').

    Returns:
    - numpy.ndarray: 2D array with interpolated values.
    """
    # Create an array to hold the interpolated data
    interpolated_data = np.empty_like(data)

    # Create an index array representing the x-axis of the 2D data
    x = np.arange(data.shape[1])

    # Apply 1D interpolation for each row
    for i, row in enumerate(data):
        valid_mask = ~np.isnan(row)  # Mask of non-nan values
        if np.isnan(row).all():
            interpolated_data[i] = row.copy()
            continue
        interpolated_data[i] = np.interp(x, x[valid_mask], row[valid_mask])

        # If kind is quadratic and there are more than 2 valid data points
        if kind == 'quadratic' and np.sum(valid_mask) > 2:
            f = np.poly1d(np.polyfit(x[valid_mask], row[valid_mask], 2))
            interpolated_data[i] = f(x)
            interpolated_data[i][valid_mask] = row[valid_mask]

        if kind == 'quadratic linear':
            valid_idx = np.where(~np.isnan(row))[0]
            if len(valid_idx) < 2 or len(valid_idx) == len(row):
                continue
            cs = UnivariateSpline(valid_idx, row[valid_idx], s=spoint, k=k)  # CubicSpline(valid_idx, row[valid_idx])
            interpolated_data[i, :28] = cs(x)[:28]
            interpolated_data[i][valid_mask] = row[valid_mask]

        if kind == 'CubicSpline':
            valid_idx = np.where(~np.isnan(row))[0]
            if len(valid_idx) < 2 or len(valid_idx) == len(row):
                continue
            cs = UnivariateSpline(valid_idx, row[valid_idx], s=spoint, k=k)  # CubicSpline(valid_idx, row[valid_idx])
            interpolated_data[i] = cs(x)
            interpolated_data[i][valid_mask] = row[valid_mask]
            # nan_idx = np.where(np.isnan(row))[0]
            # data[i, nan_idx] = cs(nan_idx)
            # interpolated_data[i] = row

    return interpolated_data


def pose_interpolate(data, pose, geo, k=2, spoint=3, pad_pose=False):
    if geo is not None:
        sum_array = np.nansum(np.array([data, geo]), axis=0)
        both_nan_mask = np.isnan(data) & np.isnan(geo)
        sum_array[both_nan_mask] = np.nan
    else:
        sum_array = data

    p = np.log10(1500)
    q = np.log10(4500)
    m = np.log10(3000)

    pose_mask = pose < 0
    if pad_pose:
        pose[pose_mask] = 0
    # pose = interpolate_1D(pose[np.newaxis,], kind='linear').astype(int)[0]
    indices = (pose * 5).astype(int)

    for i in range(sum_array.shape[0]):
        if pose[i] >= 0:
            # if sum_array[i, indices[i]] < p or sum_array[i, indices[i]] > q or np.isnan(sum_array[i, indices[i]]):
            sum_array[i, indices[i]] = np.log10(3000)
            sum_array[i, indices[i] + 1:20] = np.nan

    linear_filled_data = interpolate_1D(sum_array, kind='linear')
    quadratic_filled_data = interpolate_1D(sum_array, kind='quadratic')
    quadratic_linear_filled_data = interpolate_1D(sum_array, kind='quadratic linear', k=k, spoint=spoint)
    CubicSpline_filled_data = interpolate_1D(sum_array, kind='CubicSpline', k=k, spoint=spoint)

    # linear_filled_data = fill_nan_2D(data, kind='linear')
    # quadratic_filled_data = fill_nan_2D(data, kind='cubic')

    return linear_filled_data, quadratic_filled_data, CubicSpline_filled_data, quadratic_linear_filled_data


def save_all(args, name, train_pred, val_pred, test_pred, label, meta, geo_all, interp=False, pad_pose=False, interp_geo=True, pred=0):
    saving_path = os.path.join(args.output_dir, name)

    all_result = []
    for preds in [train_pred, val_pred, test_pred]:
        x = preds[:, 0]

        xx = []
        for i in range(x.shape[0]):
            if i == 0:
                xx.append(x[i])
            else:
                xx.append(x[i, -1, :][np.newaxis, :])

        xx = np.concatenate(xx)
        if pred > 0:
            shape = (pred, xx.shape[1])
            all_result.append(np.full(shape, np.nan))
        all_result.append(xx)
        print(xx.shape)
    all_result = np.concatenate(all_result)

    print(all_result.shape, label.shape)

    if not interp:
        xxx = np.concatenate([label[:, :2], all_result], axis=1)
        print(xxx.shape)

        column = ['date_num', 'UT'] + ["{:.3f}".format(i) for i in np.arange(2.8, 8.2, 0.2)]
        header = ','.join(column)

        if not os.path.exists(name):
            # If the folder does not exist, create it
            os.makedirs(name)

        np.savetxt(os.path.join(saving_path, 'pred.csv'), xxx, delimiter=',', header=header, fmt='%.6f', comments='')

    else:
        pose = meta[0][0, :, -1]
        geo = geo_all[0][0, :, 2:]
        geo_mask = geo_all[1][0, :, 2:]
        xxx, ppp = pad_both(all_result.copy()) if interp_geo else pad_both2(all_result.copy()), pose.copy()
        ggg, ggm = pad_both(geo.copy(), True), pad_both(geo_mask.copy(), True)

        xxxx = np.log10(xxx)
        gggg = np.log10(np.abs(ggg)) * np.sign(ggg)
        mask = (1 - ggm) == 0
        gggg[mask] = np.nan
        if not interp_geo:
            gggg = None
        l_data, q_data, s_data, ql_data = pose_interpolate(xxxx, ppp.copy(), gggg, k=4, spoint=3, pad_pose=pad_pose)

        for all, pre in zip([ql_data], [' Quadratic Linear']):
            all = np.power(10, all)
            xxx = np.concatenate([label[:, :2], all[:, 14:]], axis=1)
            print(pre, xxx.shape)

            interp_name = pre if not pad_pose else pre + ' + Padding'

            column = ['date_num', 'UT'] + ["{:.3f}".format(i) for i in np.arange(2.8, 7.4, 0.2)]
            header = ','.join(column)

            if not os.path.exists(name + interp_name):
                # If the folder does not exist, create it
                os.makedirs(name + interp_name)

            np.savetxt(os.path.join(saving_path, f'pred_{interp_name}.csv'), xxx, delimiter=',', header=header, fmt='%.6f', comments='')


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(vgg16(weights=VGG16_Weights.DEFAULT).features[:4].eval())  # relu1_2
        blocks.append(vgg16(weights=VGG16_Weights.DEFAULT).features[4:9].eval())  # relu2_2
        # blocks.append(vgg16(weights =VGG16_Weights.DEFAULT).features[9:16].eval()) # relu3_3
        # blocks.append(vgg16(weights =VGG16_Weights.DEFAULT).features[16:23].eval()) # relu4_3
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = nn.ModuleList(blocks)
        self.transform = nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()

    def forward(self, input, target, rescale=True, feature_layers=[1]):
        input = input.view(-1, 1, input.shape[-2], input.shape[-1]).repeat(1, 3, 1, 1)
        target = target.view(-1, 1, target.shape[-2], target.shape[-1]).repeat(1, 3, 1, 1)
        if rescale:  # from [-1, 1] to [0, 1]
            input = input / 2 + 0.5
            target = target / 2 + 0.5
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss_l1, loss_l2 = 0.0, 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss_l1 += self.l1loss(x, y)
                loss_l2 += self.l2loss(x, y)
        return loss_l1, loss_l2


class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1, ] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx // 2, step=1), torch.arange(start=-nx // 2, end=0, step=1)), 0).reshape(nx, 1).repeat(1, ny)
        k_y = torch.cat((torch.arange(start=0, end=ny // 2, step=1), torch.arange(start=-ny // 2, end=0, step=1)), 0).reshape(1, ny).repeat(nx, 1)
        k_x = torch.abs(k_x).reshape(1, nx, ny, 1).to(x.device)
        k_y = torch.abs(k_y).reshape(1, nx, ny, 1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced == False:
            weight = 1
            if k >= 1:
                weight += a[0] ** 2 * (k_x ** 2 + k_y ** 2)
            if k >= 2:
                weight += a[1] ** 2 * (k_x ** 4 + 2 * k_x ** 2 * k_y ** 2 + k_y ** 4)
            weight = torch.sqrt(weight)
            loss = self.rel(x * weight, y * weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x ** 2 + k_y ** 2)
                loss += self.rel(x * weight, y * weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x ** 4 + 2 * k_x ** 2 * k_y ** 2 + k_y ** 4)
                loss += self.rel(x * weight, y * weight)
            loss = loss / (k + 1)

        return loss


def l1_loss(y_true, y_pred, mask):
    y_pred[mask] = torch.nan
    y_true[mask] = torch.nan
    return torch.nanmean(torch.abs(y_true - y_pred))


def l2_loss(y_true, y_pred, mask):
    y_pred[mask] = torch.nan
    y_true[mask] = torch.nan
    return torch.nanmean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred, mask):
    y_pred[mask] = torch.nan
    y_true[mask] = torch.nan
    SS_res = torch.nansum((y_true - y_pred) ** 2)
    SS_tot = torch.nansum((y_true - torch.nanmean(y_true)) ** 2)
    return (1 - SS_res / (SS_tot + 1e-7))


if __name__ == '__main__':
    vgg = vgg16(pretrained=True)
    print(vgg)
