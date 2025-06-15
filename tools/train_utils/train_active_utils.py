import glob
import os
import torch
import tqdm
from torch.nn.utils import clip_grad_norm_
# added
from pcdet.utils import active_training_utils
from pcdet.config import cfg
from .optimization import build_scheduler
from .train_utils import save_checkpoint, checkpoint_state, resume_dataset, model_state_to_cpu

import wandb


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, cur_epoch=None, history_accumulated_iter=None):
    # update for active learning framework

    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader) # [k, dict]

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
    

    model.train()
    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter) # dict: points + gt_boxes
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        optimizer.zero_grad()

        # NOTE: also feed epoch id to model
        # for the semantics threshold adjustment
        batch['cur_epoch'] = cur_epoch
        loss, tb_dict, disp_dict = model_func(model, batch)
        
        cur_semantic_value = disp_dict.pop('cur_semantic_value') # sem_thr
        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()
        lr_scheduler.step(accumulated_iter)

        # log_accumulated_iter: 
        # for re-train the model in different selected samples stage, align the total iteration number
        if history_accumulated_iter is not None:
            history_accumulated_iter += 1
            accumulated_iter += 1
            log_accumulated_iter = history_accumulated_iter
        else:
            accumulated_iter += 1
            log_accumulated_iter = accumulated_iter
        
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=log_accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, log_accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, log_accumulated_iter)
                wandb.log({'train/loss': loss, 'meta_data/learning_rate': cur_lr}, step=log_accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, log_accumulated_iter)
                    wandb.log({'train/' + key: val}, step=log_accumulated_iter)

    if rank == 0:
        pbar.close()
    if history_accumulated_iter is not None:
        return accumulated_iter, history_accumulated_iter
    else:
        return accumulated_iter


def train_model_actively(model, optimizer, labelled_loader, unlabelled_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, active_label_dir, backbone_dir,
                labelled_sampler=None, unlabelled_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1, 
                max_ckpt_save_num=4, merge_all_iters_to_one_epoch=False, logger=None, ema_model=None, dist_train=False):

    ## Step 1: Given random samples (labelled_sampler)
    ## Pre-train models for #(cfg.ACTIVE_TRAIN.PRE_TRAIN_EPOCH_NUMS) epochs
    accumulated_iter = start_iter
    active_pre_train_epochs = cfg.ACTIVE_TRAIN.PRE_TRAIN_EPOCH_NUMS
    logger.info("***** Start Active Pre-train *****")
    ## whether finishing pretrain 
    pretrain_finished = False
    ## if finish pretraining, the selected active samples
    selected_active_samples_list = []
    ## pretrain logged models
    backbone_init_ckpt = str(backbone_dir / 'init_checkpoint.pth')

    ## save init backbone
    if not os.path.isfile(backbone_init_ckpt):
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
        torch.save(model_state, backbone_init_ckpt)
        logger.info("**init backbone weights saved...**")

    ## whether resume training, if so, then load existed backbone
    if cfg.ACTIVE_TRAIN.TRAIN_RESUME:
        ## pretrained checkpoint
        backbone_ckpt_list = [i for i in glob.glob(str(backbone_dir / 'checkpoint_epoch_*.pth'))]
        assert(len(backbone_ckpt_list) > 0) # otherwise nothing to resume
        ## active learning (further selection) checkpoint
        ckpt_list = [i for i in glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))]

        ## filter out the backbones trained after active_pre_train_epochs
        backbone_ckpt_list = [i for i in backbone_ckpt_list if int(i.split('_')[-1].split('.')[0]) <= active_pre_train_epochs]
        if len(ckpt_list) < 1:
            ## only at the pretrain stage, not select any active samples
        
            backbone_ckpt_list.sort(key=os.path.getmtime)
            last_epoch = int(backbone_ckpt_list[-1].split('_')[-1].split('.')[0])
            if last_epoch >= active_pre_train_epochs:
                pretrain_finished = True
            elif cfg.ACTIVE_TRAIN.METHOD=='llal':
                logger.info("need to finish the backbone pre-training first...")
                raise NotImplementedError
            logger.info('found {}th epoch pretrain model weights, start resuming...'.format(last_epoch))
            ## model string, containing start_epoch=int(model_str.split('_')[-1].split('.')[0])
            model_str = str(backbone_dir / backbone_ckpt_list[-1])
            ## load the latest pretrained checkpoint
            ckpt_last_epoch = torch.load(str(backbone_dir / backbone_ckpt_list[-1]))
        
        else:
            ## at the active training stage
            
            pretrain_finished = True
            ckpt_list.sort(key=os.path.getmtime)
            last_epoch = int(ckpt_list[-1].split('_')[-1].split('.')[0])
            ## model string, containing start_epoch=int(model_str.split('_')[-1].split('.')[0])
            model_str = str(ckpt_save_dir / ckpt_list[-1])
            ## load the latest active training checkpoint
            ckpt_last_epoch = torch.load(str(ckpt_save_dir / ckpt_list[-1]))

            #### lack this part
            selected_active_samples_list = [str(active_label_dir / i) for i in glob.glob(str(active_label_dir / 'selected_active_samples_epoch_*.pkl'))]
        
        if isinstance(model, torch.nn.parallel.DistributedDataParallel) or isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(ckpt_last_epoch['model_state'], strict=cfg.ACTIVE_TRAIN.METHOD!='llal')
        else:
            model.load_state_dict(ckpt_last_epoch['model_state'], strict=cfg.ACTIVE_TRAIN.METHOD!='llal')
    
        # if ckpt_last_epoch['lr_scheduler'] is not None:
        #     lr_scheduler.load_state_dict(ckpt_last_epoch['lr_scheduler']) 
        
        start_epoch = int(model_str.split('_')[-1].split('.')[0])
        cur_epoch = start_epoch
        accumulated_iter = ckpt_last_epoch['it']
        if len(selected_active_samples_list) > 0:
            labelled_loader, unlabelled_loader = resume_dataset(labelled_loader, unlabelled_loader, selected_active_samples_list, dist_train, logger, cfg)
            trained_steps = (cur_epoch - cfg.ACTIVE_TRAIN.PRE_TRAIN_EPOCH_NUMS) % cfg.ACTIVE_TRAIN.SELECT_LABEL_EPOCH_INTERVAL
            ## for active training accumulated_iter
            new_accumulated_iter = len(labelled_loader) * trained_steps


    if not pretrain_finished:
        # pretrain not complete
         with tqdm.trange(start_epoch, active_pre_train_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
            total_it_each_epoch = len(labelled_loader)
            
            if merge_all_iters_to_one_epoch:
                assert hasattr(labelled_loader.dataset, 'merge_all_iters_to_one_epoch')
                labelled_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=active_pre_train_epochs)
                total_it_each_epoch = len(labelled_loader) // max(active_pre_train_epochs, 1)

            dataloader_iter = iter(labelled_loader)
            for cur_epoch in tbar:
                # shuffle different cur_epoch
                if labelled_sampler is not None:
                    labelled_sampler.set_epoch(cur_epoch)
                
                # train one epoch
                if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                    cur_scheduler = lr_warmup_scheduler 
                else:
                    cur_scheduler = lr_scheduler
                accumulated_iter = train_one_epoch(
                    model, optimizer, labelled_loader, model_func,
                    lr_scheduler=cur_scheduler,
                    accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                    rank=rank, tbar=tbar, tb_log=tb_log,
                    leave_pbar=(cur_epoch + 1 == active_pre_train_epochs),
                    total_it_each_epoch=total_it_each_epoch,
                    dataloader_iter=dataloader_iter,
                    cur_epoch=cur_epoch,
                    history_accumulated_iter=None
                )


                # save pre-trained model
                trained_epoch = cur_epoch + 1
                if trained_epoch > 3 and trained_epoch % ckpt_save_interval == 0 and rank == 0:
                    backbone_ckpt_list = glob.glob(str(backbone_dir / 'checkpoint_epoch_*.pth'))
                    backbone_ckpt_list.sort(key=os.path.getmtime)
                    if backbone_ckpt_list.__len__() >= max_ckpt_save_num:
                        for cur_file_idx in range(0, len(backbone_ckpt_list) - max_ckpt_save_num + 1):
                            os.remove(backbone_ckpt_list[cur_file_idx])
                    ckpt_name = backbone_dir / ('checkpoint_epoch_%d' % trained_epoch)

                    save_checkpoint(
                        checkpoint_state(model, optimizer, trained_epoch, accumulated_iter, lr_scheduler), filename=ckpt_name,
                    )
            ## i.e., start_epoch = cur_epoch + 1
            start_epoch = active_pre_train_epochs

    logger.info("***** Complete Active Pre-train *****")


        
    ## Step 2: Active training loops
    total_epochs = active_pre_train_epochs + \
                   int((cfg.ACTIVE_TRAIN.TOTAL_BUDGET_NUMS / cfg.ACTIVE_TRAIN.SELECT_NUMS) # 40 + (600 / 100) * 40
                       * (cfg.ACTIVE_TRAIN.SELECT_LABEL_EPOCH_INTERVAL))

    logger.info("***** Start Active Train Loop *****")
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True,leave=(rank == 0)) as tbar:
        selection_num = 0

        for cur_epoch in tbar:
            ## 1. select active samples (optionally update the active samples)
            if (cur_epoch == active_pre_train_epochs) or ((cur_epoch - active_pre_train_epochs)% cfg.ACTIVE_TRAIN.SELECT_LABEL_EPOCH_INTERVAL == 0):
                #### lack llal case, need to train first
                selection_num = selection_num + cfg.ACTIVE_TRAIN.SELECT_NUMS
                ## accumulate the labelled / unlabelled dataloader
                
                if cur_epoch == active_pre_train_epochs:
                    record_epoch = cur_epoch
                else:
                    record_epoch = cfg.ACTIVE_TRAIN.SELECT_LABEL_EPOCH_INTERVAL
                
    
                labelled_loader, unlabelled_loader \
                    = active_training_utils.select_active_labels(
                    model,
                    labelled_loader,
                    unlabelled_loader,
                    rank,
                    logger,
                    method = cfg.ACTIVE_TRAIN.METHOD,
                    leave_pbar=True,
                    cur_epoch=cur_epoch,
                    dist_train=dist_train,
                    active_label_dir=active_label_dir,
                    accumulated_iter=accumulated_iter,
                    record_epoch=record_epoch
                )
                ## for new training set (new active samples), reset:
                ## 1. model parameters, 2. #(iteration), 3. optimizer lr, 4. scheduler, 5. new accumulate iter
                logger.info("**finished selection: reload init weights of the model")
                backbone_init_ckpt = torch.load(str(backbone_dir / 'init_checkpoint.pth'))

                ## maintain the same initialization of backbone
                model.load_state_dict(backbone_init_ckpt, strict=cfg.ACTIVE_TRAIN.METHOD!='llal')
                total_iters_each_epoch = len(labelled_loader)
                # if merge_all_iters_to_one_epoch:
                #     assert hasattr(labelled_loader.dataset, 'merge_all_iters_to_one_epoch')
                #     labelled_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=active_pre_train_epochs)
                #     total_it_each_epoch = len(labelled_loader) // max(active_pre_train_epochs, 1)
        

                for g in optimizer.param_groups:
                    g['lr'] = cfg.OPTIMIZATION.LR #/ decay_rate
                lr_scheduler, lr_warmup_scheduler = build_scheduler(
                    optimizer, total_iters_each_epoch=total_iters_each_epoch, total_epochs=cfg.ACTIVE_TRAIN.SELECT_LABEL_EPOCH_INTERVAL,
                    last_epoch=-1, optim_cfg=cfg.OPTIMIZATION
                )

                ## when actively selecting samples, reset the new_accumulated_iter to refresh the leraning rate
                new_accumulated_iter = 1

            ## 2. active training process
            if labelled_sampler is not None:
                labelled_sampler.set_epoch(cur_epoch)
            
            total_it_each_epoch = len(labelled_loader)
            dataloader_iter = iter(labelled_loader)
            logger.info("currently {} iterations to learn per epoch".format(total_it_each_epoch))

            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            ## accumulated_iter: for log information
            ## new_accumulated_iter: for trace the learning rate
            new_accumulated_iter, accumulated_iter = train_one_epoch(
                model, optimizer, labelled_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=new_accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                leave_pbar=False,
                cur_epoch=cur_epoch,
                history_accumulated_iter=accumulated_iter
            )

            ## 3. save models
            trained_epoch = cur_epoch + 1
            total_round = cfg.ACTIVE_TRAIN.TOTAL_BUDGET_NUMS // cfg.ACTIVE_TRAIN.SELECT_NUMS + 1
            total_epoch = total_round * cfg.ACTIVE_TRAIN.SELECT_LABEL_EPOCH_INTERVAL
            # if (((trained_epoch - active_pre_train_epochs) % cfg.ACTIVE_TRAIN.SELECT_LABEL_EPOCH_INTERVAL == 0) or (total_epoch - trained_epoch <= 5)) and rank == 0:
            #     ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % (trained_epoch))
            #     state = checkpoint_state(model, optimizer, trained_epoch, accumulated_iter, lr_scheduler)
            #     save_checkpoint(state, filename=ckpt_name)
            
            if (trained_epoch - active_pre_train_epochs) % cfg.ACTIVE_TRAIN.SELECT_LABEL_EPOCH_INTERVAL == 0 and rank == 0:
                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % (trained_epoch))
                state = checkpoint_state(model, optimizer, trained_epoch, accumulated_iter, lr_scheduler)
                save_checkpoint(state, filename=ckpt_name)
            
            # if trained_epoch % ckpt_save_interval == 0 and rank == 0:
            #     ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
            #     ckpt_list.sort(key=os.path.getmtime)
            #     if ckpt_list.__len__() >= max_ckpt_save_num:
            #         for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
            #             os.remove(ckpt_list[cur_file_idx])
            #     ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % (trained_epoch))
            #     state = checkpoint_state(model, optimizer, trained_epoch, accumulated_iter, lr_scheduler)
            #     save_checkpoint(state, filename=ckpt_name)
        
        
        
        
     
   
   



