"""
    CompletionFormer
    ======================================================================

    main script for training and testing.
"""


from apex import amp
from apex.parallel import DistributedDataParallel as DDP
import apex
import torch.distributed as dist
import torch.multiprocessing as mp
import cv2
from loss.l1l2loss import L1L2Loss
from datasets.hammer import HammerDataset
from utils.mics import save_output
from utils.metrics import PDNEMetric
from torch.utils.tensorboard import SummaryWriter
from model.PDNE import PDNE
from utils import train_utils
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from collections import OrderedDict
from torch import nn
import torch
from tqdm import tqdm
import numpy as np
import json
from config import args as args_config
import time
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args_config.gpus
os.environ["MASTER_ADDR"] = args_config.address
os.environ["MASTER_PORT"] = args_config.port


torch.autograd.set_detect_anomaly(True)


# Multi-GPU and Mixed precision supports
# NOTE : Only 1 process per GPU is supported now
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

best_rmse = 100
best_mae = 100

# Minimize randomness


def init_seed(seed=None):
    if seed is None:
        seed = args_config.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_args(args):
    new_args = args
    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        if args.resume:
            checkpoint = torch.load(args.pretrain, map_location={'cuda:0':'cuda:1'})

            new_args = checkpoint['args']
            new_args.test_only = args.test_only
            new_args.pretrain = args.pretrain
            new_args.dir_data = args.dir_data
            new_args.resume = args.resume
            new_args.layer0=False

    return new_args


def train(gpu, args):
    global best_rmse
    global best_mae

    # Initialize workers
    # NOTE : the worker with gpu=0 will do logging
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.num_gpus, rank=gpu)
    torch.cuda.set_device(gpu)


    # Prepare dataset
    dataset = HammerDataset(args)

    sampler_train = DistributedSampler(
        dataset, num_replicas=args.num_gpus, rank=gpu)

    batch_size = args.batch_size

    loader_train = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False,
        num_workers=args.num_threads, pin_memory=True, sampler=sampler_train,
        drop_last=True)

    # Network
    net = PDNE(args)
    net.cuda(gpu)

    if gpu == 0:
        if args.pretrain is not None:
            assert os.path.exists(args.pretrain), \
                "file not found: {}".format(args.pretrain)

            checkpoint = torch.load(args.pretrain)
            net.load_state_dict(checkpoint['net'])

            print('Load network parameters from : {}'.format(args.pretrain))

    # Loss
    loss = (args)
    loss.cuda(gpu)
    

    # Optimizer
    optimizer, scheduler = train_utils.make_optimizer_scheduler(
        args, net, len(loader_train))
    net = apex.parallel.convert_syncbn_model(net)
    net, optimizer = amp.initialize(
        net, optimizer, opt_level=args.opt_level, verbosity=0)
    
    init_epoch = 1

    if gpu == 0:
        if args.pretrain is not None:
            if args.resume:
                print('Resume:', args.resume)
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    amp.load_state_dict(checkpoint['amp'])
                    init_epoch = checkpoint['epoch']

                    print('Resume optimizer, scheduler and amp '
                          'from : {}'.format(args.pretrain))
                except KeyError:
                    print('State dicts for resume are not saved. '
                          'Use --save_full argument')

            del checkpoint

    net = DDP(net)

    metric = PDNEMetric(args)

    if gpu == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.save_dir + '/train', exist_ok=True)
        writer_train = SummaryWriter(log_dir=args.save_dir + '/' + 'train')
        writer_val = SummaryWriter(log_dir=args.save_dir + '/' + 'val')
        total_losses = np.zeros(np.array(loss.loss_name).shape)
        total_metrics = np.zeros(np.array(metric.metric_name).shape)

        with open(args.save_dir + '/args.json', 'w') as args_json:
            json.dump(args.__dict__, args_json, indent=4)

    if args.warm_up:
        warm_up_cnt = 0.0
        warm_up_max_cnt = len(loader_train)+1.0

    for epoch in range(init_epoch, args.epochs+1):
        # Train
        net.train()

        sampler_train.set_epoch(epoch)
        if gpu == 0:
            current_time = time.strftime('%y%m%d@%H:%M:%S')

            list_lr = []
            for g in optimizer.param_groups:
                list_lr.append(g['lr'])

        num_sample = len(loader_train) * \
            loader_train.batch_size * args.num_gpus

        if gpu == 0:
            pbar = tqdm(total=num_sample)
            log_cnt = 0.0
            log_loss = 0.0

        init_seed(seed=int(time.time()))
        # print("Before load")
        for batch, sample in enumerate(loader_train):
            # print("Finish load")
            sample = {key: val.cuda(gpu) for key, val in sample.items()
                      if (val is not None) and key != 'base_name'}

            if epoch == 1 and args.warm_up:
                warm_up_cnt += 1

                for param_group in optimizer.param_groups:
                    lr_warm_up = param_group['initial_lr'] \
                        * warm_up_cnt / warm_up_max_cnt
                    param_group['lr'] = lr_warm_up

            optimizer.zero_grad()


            output = net(sample)
            
            output['pred'] = output['pred'] * sample['net_mask']

            loss_dep, loss_norm = loss(sample, output)

                
                loss_sum_norm = loss_sum_norm / loader_train.batch_size


            # Divide by batch size
            loss_sum = loss_sum / loader_train.batch_size
            loss_val = loss_val / loader_train.batch_size

            with amp.scale_loss(loss_sum, optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()

            if gpu == 0:
                for i in range(len(loss.loss_name)):
                    total_losses[i] += loss_val[0][i]

                log_cnt += 1
                log_loss += loss_sum.item()

                e_string = f"{(log_loss/log_cnt):.2f}"
                if batch % args.print_freq == 0:
                    pbar.set_description(e_string)
                    pbar.update(loader_train.batch_size * args.num_gpus)

        if gpu == 0:
            pbar.close()

            for i in range(len(loss.loss_name)):
                writer_train.add_scalar(
                    loss.loss_name[i], total_losses[i] / len(loader_train), epoch)
            writer_train.add_scalar('normal_loss', loss_sum_norm.item(), epoch)
            writer_train.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

            if ((epoch) % args.save_freq == 0)and (epoch>60):
                if args.save_full or epoch == args.epochs:
                    state = {
                        'net': net.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'amp': amp.state_dict(),
                        'args': args,
                        'epoch': epoch
                    }
                else:
                    state = {
                        'net': net.module.state_dict(),
                        'args': args,
                        'epoch': epoch
                    }

                torch.save(
                    state, '{}/model_{:05d}.pt'.format(args.save_dir, epoch))
                
        scheduler.step()

        if gpu == 0:
            total_losses = np.zeros(np.array(loss.loss_name).shape)
            total_metrics = np.zeros(np.array(metric.metric_name).shape)

    if gpu == 0:
        writer_train.close()
        writer_val.close()


def test(args):
    # Prepare dataset
    data = get_data(args)

    data_test = data(args, 'test')

    loader_test = DataLoader(dataset=data_test, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_threads)

    # Network
    net = PDNE(args)

    # device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
    net.cuda()

    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        checkpoint = torch.load(args.pretrain)
        print('Start epoch:', checkpoint['epoch'])
        net.load_state_dict(checkpoint['net'], strict=False)
        print('Checkpoint loaded from {}!'.format(args.pretrain))

    net = nn.DataParallel(net)

    metric = PDNEMetric(args)

    try:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.save_dir + '/test', exist_ok=True)
    except OSError:
        pass

    total_metrics = np.zeros(np.array(metric.metric_name).shape)
    total_metrics_missing = np.zeros(np.array(metric.metric_name).shape)

    net.eval()

    num_sample = len(loader_test)*loader_test.batch_size

    pbar = tqdm(total=num_sample)

    t_total = 0

    init_seed()
    for batch, sample in enumerate(loader_test):
        base_name = sample['base_name']
        sample = {key: val.cuda() for key, val in sample.items()
                  if (val is not None) and key != 'base_name'}
        sample['base_name'] = base_name

        if args.mode == 'pd':
            image_coordinate = sample['coordinate']
            viewing_direction = sample['vd']
            aop = sample['input'][:, 5:6, ...]
            net_in = get_net_input_cuda(
                sample['input'], image_coordinate, viewing_direction, args)
            sample['input'] = net_in
        else: aop=None

        t0 = time.time()
        with torch.no_grad():
            output = net(sample)
        t1 = time.time()

        # vis the output

        if args.mode == 'pd' or 'grayd':
            sparse_type = args.sparse_dir.split('spw2_sparse_')[1]

            if args.polar:
                input_type = 'polar'
            else:
                input_type = 'gray'
            save_dir = os.path.join(args.save_dir, args.vis_dir)
            save_output(sample, output, aop, save_dir, input_type, sparse_type, args.with_norm)

        t_total += (t1 - t0)

        # print(output['pred'].shape)
        metric_val = metric.evaluate(sample, output, 'test')
        # print(metric_val[0])
        metric_val_missing = metric.evaluate(sample, output, 'on_missing')

        total_metrics = np.add(total_metrics, metric_val[0].detach().cpu())

        total_metrics_missing = np.add(
            total_metrics_missing, metric_val_missing[0].detach().cpu())

        current_time = time.strftime('%y%m%d@%H:%M:%S')
        error_str = '{} | Test'.format(current_time)
        if batch % args.print_freq == 0:
            pbar.set_description(error_str)
            pbar.update(loader_test.batch_size)

    pbar.close()
    mean_metrics = np.divide(total_metrics, len(loader_test))
    mean_metrics_missing = np.divide(total_metrics_missing, len(loader_test))
    metrics_dict = {}
    metrics_dict_missing = {}

    for i in range(len(mean_metrics)):
        name = metric.metric_name[i]
        metrics_dict[name] = mean_metrics[i]
        metrics_dict_missing[name] = mean_metrics_missing[i]

    # writer_test.update(args.epochs, sample, output)
    return metrics_dict, metrics_dict_missing
    # t_avg = t_total / num_sample
    # print('Elapsed time : {} sec, '
    #       'Average processing time : {} sec'.format(t_total, t_avg))


def main(args):
    init_seed()
    if not args.test_only:
        if args.no_multiprocessing:
            train(0, args)
        else:
            assert args.num_gpus > 0

            spawn_context = mp.spawn(train, nprocs=args.num_gpus, args=(args,),
                                     join=False)

            while not spawn_context.join():
                pass

            for process in spawn_context.processes:
                if process.is_alive():
                    process.terminate()
                process.join()

        args.pretrain = '{}/model_best.pt'.format(args.save_dir)

    else:
        # test_sparsity_list = ['spw2_sparse_ranged_tof', 'spw2_sparse_ranged_feature',
        #                     'spw2_sparse_ranged_random', 'spw2_sparse_holes', 'spw2_sparse_ranged_fov']
        # test_sparsity_list = ['spw2_sparse_tof']
        test_sparsity_list = [args.sparse_dir]
        os.makedirs(args.save_dir, exist_ok=True)
        result = open(os.path.join(args.save_dir, 'test_result.txt'), 'w+')
        if args.mixed:
            args.mixed = False
        for i in test_sparsity_list:
            args.sparse_dir = i
            args.vis_dir = 'vis_'+i.split('spw2_sparse_')[1]
            metric_dict, metric_dict_missing = test(args)
            if args.polar:
                mode = 'polar'
            else:
                mode = 'gray'
            result.writelines(i+' '+mode+'\n')

            for i in metric_dict.keys():
                current_str = f'{i}: {metric_dict[i]}    {i}_on_missing: {metric_dict_missing[i]}'+'\n'
                result.writelines(current_str)
        result.close()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args_config.gpus
    os.environ["MASTER_ADDR"] = args_config.address
    os.environ["MASTER_PORT"] = args_config.port
    args_main = check_args(args_config)

    # print('\n\n=== Arguments ===')
    # cnt = 0
    # for key in sorted(vars(args_main)):
    #     print(key, ':',  getattr(args_main, key), end='  |  ')
    #     cnt += 1
    #     if (cnt + 1) % 5 == 0:
    #         print('')
    # print('\n')

    main(args_main)
