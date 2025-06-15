import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
# from test import repeat_eval_ckpt

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import wandb

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_active_dataloader
from pcdet.models import build_network, model_fn_decorator

from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler

from train_utils.train_active_utils import train_model_actively

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    ## dynamically assigned by torch.distributed.launch
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=1, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    # active learning tag
    strategy_tag = args.extra_tag
    args.extra_tag = strategy_tag + '-select-{}'.format(cfg.ACTIVE_TRAIN.PRE_TRAIN_SAMPLE_NUMS)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        ## cfg.LOCAL_RANK is dynamically assigned by args.local_rank
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        ## in the args, batch_size means the total_batch_size for all gpus
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    assert args.fix_random_seed, "we must fix random seed (519)."
    if args.fix_random_seed:
        common_utils.set_random_seed(519)

    # exp_group_path: dataset name (sunrgbd_models/scannet_models)
    # tag: model name (CAGroup3D)
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    ps_label_dir = output_dir / 'ps_label'
    active_label_dir = output_dir / 'active_label' 
    # for all active training models, they share the same pretrained weights
    backbone_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / 'backbone' / args.extra_tag / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ps_label_dir.mkdir(parents=True, exist_ok=True)
    active_label_dir.mkdir(parents=True, exist_ok=True)
    backbone_dir.mkdir(parents=True, exist_ok=True)


    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))
        # run_name_elements: sunrgbd_dataset_CAGroup3D_active_40_40_100_100_20240604-133954
        dataset_name = cfg.DATA_CONFIG._BASE_CONFIG_.split('/')[-1].split('.')[0]
        run_name_elements = [dataset_name] + [cfg.TAG] + [cfg.ACTIVE_TRAIN.PRE_TRAIN_EPOCH_NUMS] + [cfg.ACTIVE_TRAIN.SELECT_LABEL_EPOCH_INTERVAL] + \
        [cfg.ACTIVE_TRAIN.PRE_TRAIN_SAMPLE_NUMS] + [cfg.ACTIVE_TRAIN.SELECT_NUMS] + [datetime.datetime.now().strftime('%Y%m%d-%H%M%S')]
        run_name_elements = '_'.join([str(i) for i in run_name_elements])
        
        # wandb, project=sunrgbd_dataset_train(scannet_dataset_train), tags=CAGroup3D
        wandb.init(project=dataset_name+'_train', tags=args.cfg_file.split('/')[-1].split('.')[0])
        wandb.run.name = run_name_elements
        wandb.config.update(args)
        wandb.config.update(cfg)

    

    # -----------------------create dataloader & network & optimizer---------------------------

    # ---------------- Orgnaize Random Dataset ----------------
    ## step 1: randomize pretrained data
    ## training=True: random samples
    ## the randomness is fixed
    labelled_set, unlabelled_set, labelled_loader, unlabelled_loader, \
            labelled_sampler, unlabelled_sampler = build_active_dataloader(
            cfg.DATA_CONFIG, cfg.CLASS_NAMES, args.batch_size,
            dist_train, workers=args.workers, logger=logger, training=True
        )
    

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=labelled_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    logger.info('device count: {}'.format(torch.cuda.device_count()))
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    # logger.info(model)
    
    
    # -----------------------Pretrain + Active Train---------------------------
    ## step 2: pretrain + active train
    total_iters_each_epoch = len(labelled_loader) if not args.merge_all_iters_to_one_epoch \
                                            else len(labelled_loader) // args.epochs
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=total_iters_each_epoch, total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )
    
    ## active training mode
    train_func = train_model_actively


    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    ## pretrain + active train
    train_func(
        model,
        optimizer,
        labelled_loader,
        unlabelled_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        active_label_dir=active_label_dir,
        backbone_dir=backbone_dir,
        labelled_sampler=labelled_sampler,
        unlabelled_sampler=unlabelled_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        logger=logger, # NOTE: also feed logger to train_model
        ema_model=None,
        dist_train=dist_train
    )


    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

 
    # logger.info('**********************Start evaluation %s/%s(%s)**********************' 
    #             %(cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    # test_set, test_loader, sampler = build_dataloader(
    #     dataset_cfg=cfg.DATA_CONFIG,
    #     class_names=cfg.CLASS_NAMES,
    #     # TODO: rbgnet only support batch_size = 1
    #     # batch_size=args.batch_size,
    #     batch_size=1,
    #     dist=dist_train, workers=args.workers, logger=logger, training=False,
    #     repeat_tag=None, select_idx=None
    # )
    # eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    # eval_output_dir.mkdir(parents=True, exist_ok=True)
    # args.start_epoch = int((cfg.ACTIVE_TRAIN.TOTAL_BUDGET_NUMS / cfg.ACTIVE_TRAIN.SELECT_NUMS) 
    #                    * (cfg.ACTIVE_TRAIN.SELECT_LABEL_EPOCH_INTERVAL)) + cfg.ACTIVE_TRAIN.PRE_TRAIN_EPOCH_NUMS

    # repeat_eval_ckpt(
    #     model.module if dist_train else model,
    #     test_loader, args, eval_output_dir, logger, ckpt_dir,
    #     dist_test=dist_train
    # )
    # logger.info('**********************End evaluation %s/%s(%s)**********************' %
    #             (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

if __name__ == '__main__':
    main()
