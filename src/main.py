import argparse
from pathlib import Path
import copy

import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import timm.optim.optim_factory as optim_factory

import net
from engine_pretrain import train_one_epochx, evalx
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.dataset_generation import *


def get_args_parser():
    parser = argparse.ArgumentParser('Space Whether', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='FluexNet', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--depth', default=2, type=int, help='network depth')
    parser.add_argument('--patch_size', nargs='+', default=[6, 9], type=int)
    parser.add_argument('--num_res_blocks', default=1, type=int)
    parser.add_argument('--act_layer', default='gelu', type=str)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=300, metavar='N',
                        help='epochs to warmup LR')

    # Dir parameters
    parser.add_argument('--output_dir', default='./ckpt',
                        help='path where to save and load ckpt')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    # Dataset parameters
    parser.add_argument('--anno-path', default='C:/Users/yunan/Desktop/2023_05_25', help='dataset files location')
    parser.add_argument('-s', "--suffix", type=str, default=None)
    parser.add_argument('--distributed', default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--drop_fluxe', nargs='+', default=[1, 2, 3, 4], type=int,
                        help='Indicate the fluxe that do not use. Refering dataset_generation.py for more info')
    parser.add_argument('--choose_sat', nargs='+',
                        default=['ns61', 'ns53', 'ns57', 'ns56', 'ns58', 'ns55', 'ns59', 'ns63', 'ns62', 'ns60', 'ns66', 'ns65'], type=str,
                        help='The list of Sat used')

    # Training parameters
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=2, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--drop_path', default=0.1, type=float)
    parser.add_argument('--input_size', default=72, type=int, help='input size (number of days)')
    parser.add_argument('--embed_dim', default=256, type=int, help='number of decoder dim')
    parser.add_argument('--latent_dim', default=256, type=int)
    parser.add_argument('--l1w', default=1., type=float)
    parser.add_argument('--l2w', default=1., type=float)
    parser.add_argument('--l3w', default=0., type=float)
    parser.add_argument('--l4w', default=0.01, type=float)

    parser.add_argument('--predict', default=0, type=int, help='predict future hours')

    parser.add_argument('--ckpt_name', default='', help='decoder')
    parser.add_argument('--resume_auto', default=False, type=bool, action=argparse.BooleanOptionalAction, help='formate the resume based on ckpt_name')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume_x', default='', help='resume from checkpoint for 2-step training')
    parser.add_argument('--only_test', default=False, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--save_window', default=False, type=bool, action=argparse.BooleanOptionalAction, help='Save the results for each window')

    # Misc. Do not change unless you know what they are.
    parser.add_argument('--log', default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--geo_log', default=False, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--log10', default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--interp', default=False, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--token', default=-5, type=float)
    parser.add_argument("--norm", type=str, default='std')
    parser.add_argument('--geo', default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--old_version', default=False, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--omni', default=False, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--input_geo', default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--resample', default=False, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--old_geo_input', default=False, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--rescale_loss', default=False, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--step_two', default=False, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--res', default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--target_shell', default=0, type=int)
    parser.add_argument('--drop_sat', default=0, type=float)
    parser.add_argument('--noise_injection', default=0, type=float)
    parser.add_argument('--low_shell', default=False, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--remove_shell', default=-1, type=int)
    parser.add_argument('--time_avg', default=-1, type=int)
    parser.add_argument('--meta_mode', nargs='+', default=[], type=str)
    parser.add_argument('--meta_pose', default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--meta_feature', default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--new_split', default=False, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument('--smooth_label', default=False, type=bool, action=argparse.BooleanOptionalAction)
    # parser.add_argument('--weight_loss', default=False, type=bool, action=argparse.BooleanOptionalAction)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


act_dict = {'gelu': nn.GELU, 'silu': nn.SiLU}


def train(args, device, log_writer, data_loader_train, data_loader_valid, data_loader_test, en, ed, eh, ew, dn, dd, dh, dw, meta_dim, num_sat, shell_num, label_min, label_max, sat,
          name=''):
    model = net.__dict__[args.model](img_size=(eh, ew), patch_size=tuple(args.patch_size), in_chans=ed, embed_dim=args.embed_dim, depth=args.depth,
                                      latent_dim=args.latent_dim, num_heads=16, drop_path=args.drop_path, act_layer=act_dict[args.act_layer], rescale_loss=args.rescale_loss,
                                      lambda_g1v=args.l1w, lambda_g2v=args.l2w, lambda_g3v=args.l3w, lambda_g4v=args.l4w, target_shell=args.target_shell, meta_dim=meta_dim,
                                      num_sat=num_sat, drop_sat=args.drop_sat, shell_num=shell_num, num_res_blocks=args.num_res_blocks, step_two=args.step_two, res=args.res,
                                      means=sat['mean'], stds=sat['std'], meta_feature=args.meta_feature)

    model.to(device)

    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr  # * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    try:
        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)  # , skip_list=['out_shell_net.weight', 'out_shell_net.weight']
    except:
        param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)  # , no_weight_decay_list=['out_shell_net.weight', 'out_shell_net.weight']
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()
    
    if args.resume_auto:
        args.resume = args.output_dir + f'/checkpoint-{name}.pth'

    if args.resume:
        misc.load_model_inv(args, model_without_ddp, optimizer, loss_scaler)
    elif args.resume_x:
        misc.load_model_part1(args, model_without_ddp, 'nowcast_model')
        for pname, param in model_without_ddp.named_parameters():
            if 'nowcast_model' in pname:
                param.requires_grad = False
    else:
        args.start_epoch = 0

    _, val_pred = evalx(model, data_loader_valid, device, label_min, label_max, sat, args, name=name, prefix='val_')
    _, test_pred = evalx(model, data_loader_test, device, label_min, label_max, sat, args, name=name, prefix='test_')
    if args.only_test:
        _, train_pred = evalx(model, data_loader_train, device, label_min, label_max, sat, args, name=name, prefix='train_')
        return train_pred, val_pred, test_pred

    for pname, param in model_without_ddp.named_parameters():
        print(pname, param.requires_grad)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    training_loss = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        metric = train_one_epochx(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args)
        train_loss = metric['loss']

        if (epoch + 1) % 5 == 0:
            val_loss, _ = evalx(model, data_loader_valid, device, label_min, label_max, sat, args, name=name)

            if val_loss >= training_loss:
                print('Update val loss:', val_loss)
                training_loss = val_loss

                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, name=name)

                best_state = copy.deepcopy(model.state_dict())

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    model.load_state_dict(best_state)
    print('Training time {}; MinLoss {}'.format(total_time_str, training_loss))

    print("Start Eval")
    _, val_pred = evalx(model, data_loader_valid, device, label_min, label_max, sat, args, name=name, prefix='val_')
    _, test_pred = evalx(model, data_loader_test, device, label_min, label_max, sat, args, name=name, prefix='test_')
    _, train_pred = evalx(model, data_loader_train, device, label_min, label_max, sat, args, name=name, prefix='train_')

    return train_pred, val_pred, test_pred


def main(args, name):
    if args.distributed:
        misc.init_distributed_mode(args)
    else:
        print('Not using distributed mode')
        misc.setup_for_distributed(is_master=True)  # hack

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    cudnn.benchmark = True
    if args.only_test:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
        
    if args.only_test:
        rand = False
    else:
        rand = True
    print('Training set Random: ', rand)

    en, ed, eh, ew, dn, dd, dh, dw, meta_dim, num_sat, shell_num, scaler1, scaler2, sat, data_loader_train, data_loader_valid, dataloader_test, output, meta, geo_all = creat_dataset(args, rand=rand)
    print(en, ed, eh, ew, dn, dd, dh, dw, meta_dim, num_sat, shell_num, scaler1, scaler2, sat)

    print("Start training")
    train_pred, val_pred, test_pred = train(args, device, log_writer, data_loader_train, data_loader_valid, dataloader_test, en, ed, eh, ew, dn, dd, dh, dw, meta_dim, num_sat, shell_num, scaler1, scaler2, sat,
          name=name)

    misc.save_all(args, name, train_pred, val_pred, test_pred, output[0][0], meta, geo_all, interp=False)
    misc.save_all(args, name, train_pred, val_pred, test_pred, output[0][0], meta, geo_all, interp=True, interp_geo=False, pad_pose=False)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    model = args.model if args.model != 'FluexNet_meta_geolater_old' else 'FluexNet_meta_geolater'
    name = args.ckpt_name
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(args.output_dir, name)).mkdir(parents=True, exist_ok=True)
    main(args, name)
