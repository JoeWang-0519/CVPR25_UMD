import random
from .strategy import Strategy
import tqdm
import torch
from pcdet.models import load_data_to_gpu
import torch.nn.functional as F
import numpy as np
import wandb
import time
import scipy
from torch.distributions import Categorical
from sklearn.cluster import kmeans_plusplus, k_means, MeanShift, Birch
import copy
from ..ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

from torch.utils.data import DataLoader
from ..models.fn_regressor.fn_regressor import FNRegressor
import torch.optim as optim

class ProposedSampling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(ProposedSampling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

    def query(self, leave_pbar=True, cur_epoch=None, record_epoch=None, logger=None):
        total_training_epochs_fnp = 80
        learning_rate_decay_interval = [40, 60]
        
        num_class = len(self.labelled_loader.dataset.class_names)

        num_cluster = self.cfg.ACTIVE_TRAIN.NUM_CLUSTER
        preserve_rate = self.cfg.ACTIVE_TRAIN.PRESERVE_RATE
        max_num_prototype = self.cfg.ACTIVE_TRAIN.MAX_PROTOTYPE
        box_threshold = self.cfg.ACTIVE_TRAIN.CONF_THRESHOLD
        delta = self.cfg.ACTIVE_TRAIN.DELTA
        iou_power = self.cfg.ACTIVE_TRAIN.IOU_POWER
        sim_threshold = self.cfg.ACTIVE_TRAIN.SIM_THRESHOLD
        scale = self.cfg.ACTIVE_TRAIN.SCALE
        selected_frames_per_cluster = self.divide_numbers(self.cfg.ACTIVE_TRAIN.SELECT_NUMS, num_cluster)
        bs_infer = self.cfg.ACTIVE_TRAIN.BS_INFER
        bs_train_fn = self.cfg.ACTIVE_TRAIN.BS_FN
        
        pred_labels_distribution_list = []
        classwise_prototype_dict = {}
        classwise_embedding_dict, classwise_frameid_dict = {}, {}
        for cls_idx in range(num_class):
            classwise_embedding_dict[cls_idx] = []
            classwise_frameid_dict[cls_idx] = []
            classwise_prototype_dict[cls_idx] = []
        index_list = []
        select_dic_instance_entropy, select_dic_instance_iou = {}, {}
        select_dic_fn_number = {}

        infer_labelled_dataset = self.labelled_loader.dataset
        train_labelled_dataset = copy.deepcopy(self.labelled_loader.dataset)

        infer_labelled_dataloader = DataLoader(
            infer_labelled_dataset, batch_size=bs_infer, pin_memory=True, num_workers=4,
            shuffle=False, collate_fn=infer_labelled_dataset.collate_batch,
            drop_last=False, timeout=0
        )
        
        labelled_dataloader_iter = iter(infer_labelled_dataloader)
        labelled_it_each_epoch = len(infer_labelled_dataloader)

        if self.rank == 0:
            pbar = tqdm.tqdm(total=labelled_it_each_epoch, leave=leave_pbar,
                                desc='infer LABELLED dataset prediction', dynamic_ncols=True)
        self.model.eval()
  
        for cur_it in range(labelled_it_each_epoch):
            try:
                labelled_batch = next(labelled_dataloader_iter)
            except StopIteration:
                labelled_batch = next(iter(infer_labelled_dataloader))
            with torch.no_grad():
                load_data_to_gpu(labelled_batch)
                labelled_batch['cur_epoch'] = record_epoch
                pred_dicts, _ = self.model(labelled_batch)
                sp_tensor = labelled_batch['sp_tensor']

                for batch_idx in range(len(pred_dicts)):
                    cur_id = cur_it * bs_infer + batch_idx
                    frame_id = labelled_batch['frame_id'][batch_idx]
                    mask = sp_tensor.coordinates[:, 0] == batch_idx
                    sp_feat, sp_coord = sp_tensor.features[mask], sp_tensor.coordinates[mask]
                    global_feature = torch.mean(sp_feat, dim=0)

                    confidence_mask_shrink = pred_dicts[batch_idx]['pred_scores_shrink'] > box_threshold
                    pred_boxes_shrink = pred_dicts[batch_idx]['pred_boxes_shrink'][confidence_mask_shrink]
                    pred_labels_shrink = pred_dicts[batch_idx]['pred_labels_shrink'][confidence_mask_shrink]
                    embeddings_shrink = pred_dicts[batch_idx]['embeddings_shrink'][confidence_mask_shrink]

                    detected_feature_list = self.get_feature_from_boxes(pred_boxes_shrink, sp_coord, sp_feat)
                    detected_feature_tensor = torch.stack(detected_feature_list, dim=0)  # [N, C]

                    gt_boxes = labelled_batch['gt_bboxes_3d'][batch_idx]
                    gt_labels = labelled_batch['gt_labels_3d'][batch_idx]

                    pred_boxes_np = pred_boxes_shrink.clone().detach().cpu().numpy()
                    pred_labels_np = pred_labels_shrink.clone().detach().cpu().numpy()
                    pred_boxes_np = np.concatenate((pred_boxes_np, pred_labels_np[:, np.newaxis]), axis=1)

                    for cls in torch.unique(pred_labels_shrink):
                        cls_mask = pred_labels_shrink == cls
                        cls_detected_feature_tensor = embeddings_shrink[cls_mask, :]
                        classwise_prototype_dict[int(cls)].append(cls_detected_feature_tensor)

                    if self.cfg.DATA_CONFIG.DATASET == 'ScannetDataset':
                        train_labelled_dataset.scannet_infos[cur_id].update({
                            'fn_number': self.get_fn_number(gt_boxes, gt_labels, pred_boxes_shrink, pred_labels_shrink),
                            'pred_boxes': pred_boxes_np,
                            'global_feat': global_feature.clone().detach().cpu().numpy(),
                            'detected_feat': torch.stack(detected_feature_list).clone().detach().cpu().numpy()
                        })

                    else:    
                        train_labelled_dataset.sunrgbd_infos[cur_id].update({
                            'fn_number': self.get_fn_number(gt_boxes, gt_labels, pred_boxes_shrink, pred_labels_shrink),
                            'pred_boxes': pred_boxes_np,
                            'global_feat': global_feature.clone().detach().cpu().numpy(),
                            'detected_feat': torch.stack(detected_feature_list).clone().detach().cpu().numpy()
                        })

            if self.rank == 0:
                pbar.update()
                pbar.refresh()

        if self.rank == 0:
            pbar.close()

        dev = embeddings_shrink.device
        for cls in range(num_class):
            classwise_embedding_list = classwise_prototype_dict[cls]  # [Tensor(N_1,C), ...]
            if len(classwise_embedding_list)==0:
                classwise_avg_embedding = torch.rand(128, device=dev) * 2 - 1
            else:
                classwise_embedding_tensor = torch.cat(classwise_embedding_list, dim=0) # Tensor(N, C)
                classwise_avg_embedding = torch.mean(classwise_embedding_tensor, dim=0)  # [C]
            classwise_prototype_dict[cls] = [classwise_avg_embedding]

        train_labelled_dataloader = DataLoader(
            train_labelled_dataset, batch_size=bs_train_fn, pin_memory=True, num_workers=4,
            shuffle=True, collate_fn=train_labelled_dataset.collate_batch,
            drop_last=True, timeout=0
        )
        
        fn_regressor = FNRegressor().cuda()
        optimizer = optim.Adam(fn_regressor.parameters(), lr=0.1, weight_decay=0.001)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=learning_rate_decay_interval, gamma=0.2)
        criterion = torch.nn.MSELoss().cuda()

        labelled_dataloader_iter = iter(train_labelled_dataloader)
        labelled_it_each_epoch = len(train_labelled_dataloader)
        
        with tqdm.trange(0, total_training_epochs_fnp, desc='epochs for training FN network', dynamic_ncols=True, leave=(self.rank == 0)) as tbar:
            for _ in tbar:
                for cur_it in range(labelled_it_each_epoch):
                    try:
                        labelled_batch = next(labelled_dataloader_iter)
                    except StopIteration:
                        labelled_dataloader_iter = iter(train_labelled_dataloader)
                        labelled_batch = next(labelled_dataloader_iter)

                    load_data_to_gpu(labelled_batch)

                    global_feat = np.stack(labelled_batch['global_feat'])
                    detected_feat = np.stack([np.mean(detected_feat, axis=0) for detected_feat in labelled_batch['detected_feat']])
                    feat = torch.from_numpy(np.concatenate((global_feat, detected_feat), axis=1)).float().cuda()
                    fn_number = labelled_batch['fn_number']
                    del global_feat
                    del detected_feat

                    optimizer.zero_grad()
                    fn_regressor = fn_regressor.train()
                    pred = fn_regressor(feat)
                    loss = criterion(pred, fn_number.unsqueeze(1))
                    loss.backward()
                    optimizer.step()
            
                scheduler.step()
            
        val_dataloader_iter = iter(self.unlabelled_loader)
        total_it_each_epoch = len(self.unlabelled_loader)
        
        if self.rank == 0:
            pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                                desc='going_through_unlabelled_set_epoch_%d_required_metrics' % cur_epoch, dynamic_ncols=True)
        fn_regressor.eval()
        self.model.eval()

        logger.info('current record epoch: %d' % record_epoch)
        for cur_it in range(total_it_each_epoch):
            try:
                unlabelled_batch = next(val_dataloader_iter)
            except StopIteration:
                unlabelled_dataloader_iter = iter(self.unlabelled_loader)
                unlabelled_batch = next(unlabelled_dataloader_iter)
            with torch.no_grad():
                load_data_to_gpu(unlabelled_batch)
                unlabelled_batch['cur_epoch'] = record_epoch

                unlabelled_batch_del = copy.deepcopy(unlabelled_batch)

                pred_dicts, _ = self.model(unlabelled_batch)
                frame_ids = unlabelled_batch['frame_id']
                sp_tensor = unlabelled_batch['sp_tensor']
                
                roi_list, record_dict_original = [], []
                global_feat_list, detected_feat_list = [], []

                for batch_idx in range(len(pred_dicts)):
                    frame_id = frame_ids[batch_idx]
                    confidence_mask_shrink = pred_dicts[batch_idx]['pred_scores_shrink'] > box_threshold
                    pred_labels_shrink = pred_dicts[batch_idx]['pred_labels_shrink'][confidence_mask_shrink]
                    embeddings_shrink = pred_dicts[batch_idx]['embeddings_shrink'][confidence_mask_shrink]

                    if pred_labels_shrink.shape[0] != 0:
                        for semantic_cls in torch.unique(pred_labels_shrink):
                            cls_mask = pred_labels_shrink == semantic_cls
                            cls_embeddings = embeddings_shrink[cls_mask, :]
                            old_classwise_prototypes = classwise_prototype_dict[int(semantic_cls)]  # List([C,])
                            updated_classwise_prototypes = self.update_classwise_prototype(old_classwise_prototypes,
                                                                                           cls_embeddings,
                                                                                           max_num_prototype,
                                                                                           sim_threshold)  # List([C,])
                            classwise_prototype_dict[int(semantic_cls)] = updated_classwise_prototypes

                            classwise_embedding_dict[int(semantic_cls)].append(cls_embeddings)
                            classwise_frameid_dict[int(semantic_cls)].append(frame_id)

                        value, counts = torch.unique(pred_labels_shrink, return_counts=True)
                        pred_labels_distribution = self.category2count(value, counts, num_class)
                        pred_labels_distribution_list.append(pred_labels_distribution)
                        index_list.append(frame_id)

                    mask = sp_tensor.coordinates[:, 0] == batch_idx
                    sp_feat, sp_coord = sp_tensor.features[mask], sp_tensor.coordinates[mask]
                    global_feature = torch.mean(sp_feat, dim=0)
                    pred_boxes_shrink = pred_dicts[batch_idx]['pred_boxes_shrink'][confidence_mask_shrink]
                    roi_shrink = pred_dicts[batch_idx]['roi_shrink'][confidence_mask_shrink]
                    pred_semantic_scores_shrink = pred_dicts[batch_idx]['pred_semantic_scores_shrink'][confidence_mask_shrink]
                    detected_feature_list = self.get_feature_from_boxes(pred_boxes_shrink, sp_coord, sp_feat)
                    global_feat_list.append(global_feature)
                    detected_feat_list.append(torch.stack(detected_feature_list).mean(dim=0))

                    if pred_semantic_scores_shrink.shape[0] == 0:
                        entropy = pred_semantic_scores_shrink.new_zeros(1)

                    else:
                        entropy = -(F.softmax(pred_semantic_scores_shrink, dim=1) * F.log_softmax(pred_semantic_scores_shrink, dim=1)).sum(dim=1)
                        select_dic_instance_entropy[frame_id] = entropy

                    if roi_shrink.shape[0] == 0:
                        record_dict = {
                        'pred_boxes_original': pred_boxes_shrink,
                    }
                        roi_list.append(unlabelled_batch['pred_bbox_list'][batch_idx][0][:1, :])
                        record_dict_original.append(record_dict)
                    else:
                        record_dict = {
                        'pred_boxes_original': pred_boxes_shrink,
                    }
                        roi_list.append(roi_shrink)
                        record_dict_original.append(record_dict)
                    
                global_feat = torch.stack(global_feat_list)
                detected_feat = torch.stack(detected_feat_list)
                feat = torch.cat((global_feat, detected_feat), dim=1)
                del global_feat_list, global_feat
                del detected_feat_list, detected_feat

                pred_fn_number = fn_regressor(feat).view(-1)
                pred_fn_number = self.normalize_fn_number(pred_fn_number)
                for batch_idx in range(len(pred_dicts)):
                    frame_id = frame_ids[batch_idx]
                    select_dic_fn_number[frame_id] = pred_fn_number[batch_idx].clone().detach().cpu()

                pts = unlabelled_batch['points']
                new_pts_del = []
                for bs in range(len(pred_dicts)):
                    bs_mask = (pts[:, 0] == bs)
                    pt = pts[bs_mask]
                    pred_box = record_dict_original[bs]['pred_boxes_original']
                    if pred_box.shape[0] == 0:
                        fake_number_rows_to_preserve = int(pt.shape[0] * preserve_rate)
                        new_pts_del.append(pt[:fake_number_rows_to_preserve, :])
                    else:
                        pt_del = self.augment_del(pt, preserve_rate)
                        new_pts_del.append(pt_del)

                new_pts_del = torch.cat(new_pts_del, dim=0)
                unlabelled_batch_del['points'] = new_pts_del
                rois, batch_size = self.reorder_rois(roi_list)
                unlabelled_batch_del['predict_roi_mode'] = False
                unlabelled_batch_del['rois'] = rois
                unlabelled_batch_del['batch_size'] = batch_size
                pred_dicts_del, _ = self.model(unlabelled_batch_del)
                for batch_idx in range(len(pred_dicts_del)):
                    frame_id = unlabelled_batch_del['frame_id'][batch_idx]
                    aug_del_pred_boxes = pred_dicts_del[batch_idx]['pred_boxes']
                    ori_pred_boxes = record_dict_original[batch_idx]['pred_boxes_original']
                    if ori_pred_boxes.shape[0] == 0:
                        aug_del_iou3d = ori_pred_boxes.new_zeros(1)
                    else:
                        aug_del_iou3d = torch.diag(boxes_iou3d_gpu(aug_del_pred_boxes, ori_pred_boxes))
                    select_dic_instance_iou[frame_id] = aug_del_iou3d
            if self.rank == 0:
                pbar.update()
                pbar.refresh()

        if self.rank == 0:
            pbar.close()

        ##============================================================================================================================================================================
        # Uncertainty Selection
        candidate_num = int(delta * self.cfg.ACTIVE_TRAIN.SELECT_NUMS)
        select_dic_instance_iou_power = {key: torch.pow(select_dic_instance_iou[key], iou_power) for key in select_dic_instance_iou}
        select_dic_instance_iou_power = {key: value / value.sum() for key, value in select_dic_instance_iou_power.items()}
        dic_reweight_avg_entropy = {key: torch.sum(select_dic_instance_entropy[key] * select_dic_instance_iou_power[key]).clone().detach().cpu() for key in select_dic_instance_entropy}
        dic_reweight_avg_entropy_normalized = self.normalize_entropy(dic_reweight_avg_entropy, scale=scale)
        dic_uncertainty_metric = {key: (1 + select_dic_fn_number[key]) * dic_reweight_avg_entropy[key] for key in dic_reweight_avg_entropy}
        dic_uncertainty_metric = dict(sorted(dic_uncertainty_metric.items(), key=lambda item: -item[1]))
        selected_candidates = list(dic_uncertainty_metric.keys())[:candidate_num]

        # Diversity Selection
        frame_prototype_number_dict = {}
        for frame_id in index_list:
            frame_prototype_number_dict[frame_id] = []
        for semantic_cls in classwise_embedding_dict:
            cls_embedding_list = classwise_embedding_dict[semantic_cls]
            cls_frameid_list = classwise_frameid_dict[semantic_cls]
            classwise_prototype = classwise_prototype_dict[semantic_cls]
            cls_prototype_number_dict = self.get_cls_prototype_number_dict(logger, cls_embedding_list, cls_frameid_list,
                                                                           max_num_prototype, semantic_cls,
                                                                           classwise_prototype)
            frame_prototype_number_dict = self.update_frame_prototype_number(frame_prototype_number_dict,
                                                                             cls_prototype_number_dict,
                                                                             max_num_prototype,
                                                                             semantic_cls)
        prototype_number_list = self.get_total_prototype_number(frame_prototype_number_dict, selected_candidates)
        dict_frame2distribution = {}
        for idx in range(len(index_list)):
            frame = index_list[idx]
            distribution = pred_labels_distribution_list[idx]
            dict_frame2distribution[frame] = distribution
        pred_labels_distribution_candidate_list = []
        for candidate_frame in selected_candidates:
            pred_labels_distribution_candidate_list.append(dict_frame2distribution[candidate_frame])
        pred_labels_distribution_candidate_tensor = torch.stack(pred_labels_distribution_candidate_list, 0)
        pred_labels_distribution_candidate_tensor = pred_labels_distribution_candidate_tensor.view(-1, num_class)
        del pred_labels_distribution_candidate_list
        
        unselected_frames = copy.deepcopy(selected_candidates)
        start_time = time.time()
        centroid, label, _ = k_means(pred_labels_distribution_candidate_tensor.cpu().numpy(), n_clusters=num_cluster, random_state=519)
        logger.info("--- k-means++ running time: %s seconds for scene-type clustering---" % (time.time() - start_time))
        selected_frames = []
        for cluster_idx in range(num_cluster):
            cluster_mask = (label == cluster_idx)
            index_cluster_list = [selected_candidates[i] for i in range(len(selected_candidates)) if cluster_mask[i]]
            prototype_number_cluster_list = [prototype_number_list[i] for i in range(len(prototype_number_list)) if cluster_mask[i]]
            prototype_total_number = sum([torch.sum(prototype_number) for prototype_number in prototype_number_cluster_list])
            logger.info('Number of scenes for scene type %d: %d' % (cluster_idx, len(index_cluster_list)))
            logger.info('Number of different boxes for scene type %d: %d' % (cluster_idx, prototype_total_number))
            select_num = selected_frames_per_cluster[cluster_idx]
            selected_frames_cluster, random_frames_cluster = self.non_overlap_max_unique_prototype_number_selection(
                logger, prototype_number_cluster_list, index_cluster_list, select_num, cluster_idx
            )

            for frame in selected_frames_cluster:
                unselected_frames.remove(frame)
            if random_frames_cluster > 0:
                logger.info('We only have %d active samples, thus need %d more random samples.' % (len(selected_frames_cluster), random_frames_cluster))
                random.shuffle(unselected_frames)
                selected_frames_cluster.extend(unselected_frames[:random_frames_cluster])
                for frame in unselected_frames[:random_frames_cluster]:
                    unselected_frames.remove(frame)
                    tmp_idx = selected_candidates.index(frame)
                    del selected_candidates[tmp_idx]
                    del prototype_number_list[tmp_idx]
                    label = np.delete(label, tmp_idx)
            selected_frames.extend(selected_frames_cluster)
        return selected_frames

    def category2count(self, categories, counts, total_category):
        result = torch.zeros(total_category, dtype=torch.long, device=categories.device)
        categories = categories.to(torch.long)
        result[categories] = counts
        return result

    def examine_inside_pt(self, pts, boxes):
        dev = pts.device
        all_indices = []
        for box_id in range(boxes.shape[0]):
            box = boxes[box_id]
            _, _, _, _, _, _, a = box
            pts_translate = pts[:, 1:4] - box[0:3]
            cos_a = torch.cos(a)
            sin_a = torch.sin(a)
            R = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]], device=dev)
            pts_rotate = torch.matmul(pts_translate, R.T)
            within_box = torch.all(torch.abs(pts_rotate) <= box[3:6]/2, dim=1)
            # wo_box = torch.any(torch.abs(pts_rotate) > box[3:6]/2, dim=1)
            within_indices = torch.where(within_box)[0]
            # wo_indices = torch.where(wo_box)[0] 

            all_indices.append(within_indices)
        
        all_indices = torch.cat(all_indices).unique()
        return all_indices
    
    def augment_add(self, inside_indices, pts, noise_std):
        N = inside_indices.shape[0]
        num_rows_to_add_noise = N//3

        pts_new = pts.clone().detach()
        dev = pts_new.device
        
        random_indices = torch.randperm(N)[:num_rows_to_add_noise]
        random_inside_indices = inside_indices[random_indices]
        noise = torch.normal(0, noise_std, size=(num_rows_to_add_noise, 3), device=dev)

        pts_new[random_inside_indices, 1:4] += noise
        return pts_new
        
    def augment_del(self, pts, preserve_rate):
        N = pts.shape[0]
        num_rows_to_preserve = int(N * preserve_rate)

        # Randomly select 80% of the rows
        selected_indices = torch.randperm(N)[:num_rows_to_preserve]

        # Get the 80% rows
        preserved_pts = pts[selected_indices]

        return preserved_pts

    
    def reorder_rois(self, roi_list):
        batch_size = len(roi_list)
        num_max_rois = max([roi.shape[0] for roi in roi_list])
        
        roi = roi_list[0]
        rois = roi.new_zeros((batch_size, num_max_rois, roi.shape[-1]))

        for bs_idx in range(batch_size):
            num_boxes = len(roi_list[bs_idx])
            rois[bs_idx, :num_boxes, :] = roi_list[bs_idx]

        # rois[..., 6] *= -1
        
        return rois, batch_size

    def normalize_acquistion_metric(self, acquistion_metric):
        N, dev = acquistion_metric.shape[0], acquistion_metric.device
        aug_acquistion_metric = (acquistion_metric - torch.mean(acquistion_metric)) / (6 * torch.std(acquistion_metric))
        aug_acquistion_metric += 1/2
        aug_acquistion_metric = torch.maximum(aug_acquistion_metric, torch.zeros(N, device=dev))

        return aug_acquistion_metric

    def normalize_instance_metric(self, select_dic_instance_metric):
        all_values = torch.cat(list(select_dic_instance_metric.values()))
        mean, std = torch.mean(all_values), torch.std(all_values)

        select_dic_instance_metric_normalized = {key: torch.maximum(((value - mean) / (6 * std)) + 1/2, torch.zeros(value.shape[0], device=value.device)) \
                            for key, value in select_dic_instance_metric.items()}
        
        return select_dic_instance_metric_normalized

    def divide_numbers(self, total, length):
        candidate1 = int(total // length)
        candidate2 = candidate1 + 1

        candidate2_repeat = total % length
        candidate1_repeat = length - candidate2_repeat

        result = [candidate1] * candidate1_repeat + [candidate2] * candidate2_repeat
        return result

    def get_cls_prototype_number_dict(self, logger, cls_embedding_list, cls_frameid_list, max_num_prototype,
                                      semantic_cls, classwise_prototype):
        # cls_embedding_list: List[Tensor(k, d)]
        # cls_frameid_list: List[]
        # classwise_prototype: List[Tensor(C,)]

        num_prototype = len(classwise_prototype)
        logger.info("--- for class %d prototype, its prototype number is: %d, and it has %d scenes---" % (
            semantic_cls, num_prototype, len(cls_embedding_list)))
        cluster_init = torch.stack(classwise_prototype)  # [N, C]

        cls_prototype_number_dict = {}

        cls_embedding_number_list = [embedding.shape[0] for embedding in cls_embedding_list]
        for idx in range(1, len(cls_embedding_number_list)):
            cls_embedding_number_list[idx] += cls_embedding_number_list[idx - 1]
        cls_embedding_tensor = torch.cat(cls_embedding_list, 0)

        start_time = time.time()

        device = cls_embedding_list[0].device
        _, assignment, _ = k_means(cls_embedding_tensor.cpu().numpy(), n_clusters=num_prototype,
                                   random_state=519, init=cluster_init.cpu().numpy())
        assignment = torch.tensor(assignment, device=device, dtype=torch.long)
        logger.info("--- k-means++ running time: %s seconds for class %d prototype selection ---" % (
            time.time() - start_time, semantic_cls))
        logger.info('cls_frameid_list_len: %d, cls_embedding_number_list_len: %d'% (len(cls_frameid_list),
                                                                                    len(cls_embedding_number_list)))
        logger.info('assignment_len: %d, cls_embedding_number_list_last: %d' % (assignment.shape[0],
                                                                                cls_embedding_number_list[-1]))

        for idx in range(len(cls_frameid_list)):
            frame_id = cls_frameid_list[idx]

            if idx == 0:
                start, end = 0, cls_embedding_number_list[idx]
            else:
                start, end = cls_embedding_number_list[idx - 1], cls_embedding_number_list[idx]

            prototype_idx, prototype_counts = torch.unique(assignment[start: end], return_counts=True)
            cls_prototype_number = self.category2count(prototype_idx, prototype_counts, max_num_prototype)

            # cls_prototype_number = torch.zeros(num_prototype, dtype=torch.long, device=device)
            # cls_prototype_number[prototype_idx] = prototype_counts

            cls_prototype_number_dict[frame_id] = cls_prototype_number

        return cls_prototype_number_dict


    def update_frame_prototype_number(self, frame_prototype_number_dict, cls_prototype_number_dict, num_prototype, semantic_cls):
        # frame_prototype_number_dict: Dict{frame_id: List()}
        # cls_prototype_number_dict: Dict{frame_id: Tensor(num_prototype,)}

        device = list(cls_prototype_number_dict.values())[0].device

        for frame_id in frame_prototype_number_dict:
            if frame_id in cls_prototype_number_dict:
                frame_cls_prototype_number = cls_prototype_number_dict[frame_id]
            else:
                frame_cls_prototype_number = torch.zeros(num_prototype, dtype=torch.long, device=device)

            frame_prototype_number_dict[frame_id].append(frame_cls_prototype_number)
        
        return frame_prototype_number_dict

    def update_classwise_prototype(self, old_classwise_prototypes, cls_embeddings, max_num_prototype_per_category, sim_threshold):
        # old_class_wise_prototypes: List([C,])
        # cls_embeddings: [N, C]
        # max_num_prototype_per_category, sim_threshold: hyperparam
        classwise_object_num = cls_embeddings.shape[0]
        cur_prototpye_num = len(old_classwise_prototypes)
        if cur_prototpye_num >= max_num_prototype_per_category:
            return old_classwise_prototypes
        else:
            for obj_idx in range(classwise_object_num):
                cur_prototpye_num = len(old_classwise_prototypes)
                if cur_prototpye_num >= max_num_prototype_per_category:
                    return old_classwise_prototypes
                else:
                    object_embedding = cls_embeddings[obj_idx].unsqueeze(0) # [1, C]
                    old_classwise_prototypes_tensor = torch.stack(old_classwise_prototypes, dim=0) # [N, C]
                    cosine_similarity = F.cosine_similarity(object_embedding, old_classwise_prototypes_tensor, dim=1) # [N]
                    if torch.max(cosine_similarity) < sim_threshold:
                        # set as new prototype
                        old_classwise_prototypes.append(cls_embeddings[obj_idx])
                    else:
                        max_idx = torch.argmax(cosine_similarity)
                        similar_prototype_embedding = old_classwise_prototypes[max_idx]
                        # online update
                        similar_prototype_embedding = cur_prototpye_num / (cur_prototpye_num+1) * similar_prototype_embedding + 1 / (cur_prototpye_num+1) * cls_embeddings[obj_idx]
                        old_classwise_prototypes[max_idx] = similar_prototype_embedding

        return old_classwise_prototypes


    def get_total_prototype_number(self, frame_prototype_number_dict, index_list):
        prototype_number_list = []
        for idx in index_list:
            frame_cls_prototype_number_list = frame_prototype_number_dict[idx] # List[Tensor(10,), ..., Tensor(10,)]
            frame_total_prototype_number = torch.cat(frame_cls_prototype_number_list)
            prototype_number_list.append(frame_total_prototype_number)

        return prototype_number_list

    def max_prototype_number_selection(self, logger, prototype_number_cluster_list, index_cluster_list, selected_frames_per_cluster, cluster_idx):
        num_scene_cluster = len(index_cluster_list)
        
        if num_scene_cluster >= selected_frames_per_cluster:
            prototype_unique_number_cluster_list = [torch.sum(prototype_number) for prototype_number in prototype_number_cluster_list]
            
            selected_dic_unique_prototype_number = dict(zip(index_cluster_list, prototype_unique_number_cluster_list))
            selected_dic_unique_prototype_number = dict(sorted(selected_dic_unique_prototype_number.items(), key=lambda item: -item[1]))
            
            selected_frames_cluster = list(selected_dic_unique_prototype_number.keys())[:selected_frames_per_cluster]
            random_frames_cluster = 0
        else:
            selected_frames_cluster = index_cluster_list
            logger.info('In class %d, there are only %d scenes!' % (cluster_idx, num_scene_cluster))
            random_frames_cluster = selected_frames_per_cluster - num_scene_cluster
    
        return selected_frames_cluster, random_frames_cluster
        

    def max_unique_prototype_number_selection(self, logger, prototype_number_cluster_list, index_cluster_list, selected_frames_per_cluster, cluster_idx):
        num_scene_cluster = len(index_cluster_list)
        
        if num_scene_cluster >= selected_frames_per_cluster:
            prototype_unique_number_cluster_list = [torch.count_nonzero(prototype_number) for prototype_number in prototype_number_cluster_list]
            
            selected_dic_unique_prototype_number = dict(zip(index_cluster_list, prototype_unique_number_cluster_list))
            selected_dic_unique_prototype_number = dict(sorted(selected_dic_unique_prototype_number.items(), key=lambda item: -item[1]))
            
            selected_frames_cluster = list(selected_dic_unique_prototype_number.keys())[:selected_frames_per_cluster]
            random_frames_cluster = 0
        else:
            selected_frames_cluster = index_cluster_list
            logger.info('In class %d, there are only %d scenes!' % (cluster_idx, num_scene_cluster))
            random_frames_cluster = selected_frames_per_cluster - num_scene_cluster
    
        return selected_frames_cluster, random_frames_cluster

    

    def non_overlap_max_prototype_number_selection(self, logger, prototype_number_cluster_list, index_cluster_list, selected_frames_per_cluster, cluster_idx):
        num_scene_cluster = len(index_cluster_list)
        

        if num_scene_cluster >= selected_frames_per_cluster:
            selected_frames_cluster = []
            random_frames_cluster = 0

            num_total_prototype, device = prototype_number_cluster_list[0].shape[0], prototype_number_cluster_list[0].device
            accumulated_prototype = torch.zeros(num_total_prototype, dtype=torch.long, device=device)
            for active_round in range(selected_frames_per_cluster):
                zero_mask = (accumulated_prototype == 0)
                _, indices = torch.topk(accumulated_prototype, num_total_prototype//4, largest=False)  # 25% quantile
                quantile_mask = torch.zeros(num_total_prototype, dtype=torch.bool, device=device)
                quantile_mask[indices] = True
                prototype_mask = zero_mask | quantile_mask


                prototype_number_cluster_nonoverlap_list = [torch.sum(prototype_number[prototype_mask]) for prototype_number in prototype_number_cluster_list]
                max_index = prototype_number_cluster_nonoverlap_list.index(max(prototype_number_cluster_nonoverlap_list))
                frameid_non_overlap_max_prototype_number, prototype_number_non_overlap_max_prototype_number = index_cluster_list[max_index], prototype_number_cluster_list[max_index]
                
                selected_frames_cluster.append(frameid_non_overlap_max_prototype_number)
                accumulated_prototype += prototype_number_non_overlap_max_prototype_number

                del prototype_number_cluster_list[max_index]
                del index_cluster_list[max_index]
            
        else:
            selected_frames_cluster = index_cluster_list
            logger.info('In class %d, there are only %d scenes!' % (cluster_idx, num_scene_cluster))
            random_frames_cluster = selected_frames_per_cluster - num_scene_cluster

        return selected_frames_cluster, random_frames_cluster

    def non_overlap_max_unique_prototype_number_selection(self, logger, prototype_number_cluster_list, index_cluster_list, selected_frames_per_cluster, cluster_idx):
        num_scene_cluster = len(index_cluster_list)
        

        if num_scene_cluster >= selected_frames_per_cluster:
            selected_frames_cluster = []
            random_frames_cluster = 0

            num_total_prototype, device = prototype_number_cluster_list[0].shape[0], prototype_number_cluster_list[0].device
            accumulated_prototype = torch.zeros(num_total_prototype, dtype=torch.long, device=device)
            for active_round in range(selected_frames_per_cluster):
                zero_mask = (accumulated_prototype == 0)
                _, indices = torch.topk(accumulated_prototype, num_total_prototype//4, largest=False)  # 25% quantile
                quantile_mask = torch.zeros(num_total_prototype, dtype=torch.bool, device=device)
                quantile_mask[indices] = True
                prototype_mask = zero_mask | quantile_mask


                prototype_number_cluster_nonoverlap_list = [torch.count_nonzero(prototype_number[prototype_mask]) for prototype_number in prototype_number_cluster_list]
                max_index = prototype_number_cluster_nonoverlap_list.index(max(prototype_number_cluster_nonoverlap_list))
                frameid_non_overlap_max_unique_prototype_number, prototype_number_non_overlap_max_unique_prototype_number = index_cluster_list[max_index], prototype_number_cluster_list[max_index]
                
                selected_frames_cluster.append(frameid_non_overlap_max_unique_prototype_number)
                accumulated_prototype += prototype_number_non_overlap_max_unique_prototype_number

                del prototype_number_cluster_list[max_index]
                del index_cluster_list[max_index]
            
        else:
            selected_frames_cluster = index_cluster_list
            logger.info('In class %d, there are only %d scenes!' % (cluster_idx, num_scene_cluster))
            random_frames_cluster = selected_frames_per_cluster - num_scene_cluster

        return selected_frames_cluster, random_frames_cluster


    def get_fn_number(self, gt_boxes, gt_labels, pred_boxes_shrink, pred_labels_shrink):
        fn_number = 0
        threshold = 0.3

        for gt_label in torch.unique(gt_labels):
            gt_label_mask = gt_labels==gt_label
            pred_label_mask = pred_labels_shrink==gt_label
            

            gt_boxes_filter = gt_boxes[gt_label_mask]
            pred_boxes_filter = pred_boxes_shrink[pred_label_mask]

            if pred_boxes_filter.shape[0] > 0:
                iou = boxes_iou3d_gpu(gt_boxes_filter, pred_boxes_filter)
                fn_tag = torch.any(iou>threshold, dim=1)
                fn_count = torch.sum(fn_tag == False).item()
                fn_number += fn_count
            else:
                fn_number += gt_boxes_filter.shape[0]

        return fn_number

    def normalize_fn_number(self, pred_fn_number, upper_bound=1.):
        return torch.where(pred_fn_number>upper_bound, torch.tensor(upper_bound, device=pred_fn_number.device), pred_fn_number)

    def normalize_entropy(self, dic_reweight_avg_entropy, scale, eps=1e-5):
        keys = list(dic_reweight_avg_entropy.keys())
        vals = torch.tensor(list(dic_reweight_avg_entropy.values()), dtype=torch.float)
        mean = vals.mean()
        var = vals.var() * scale + eps
        normed = (vals - mean) / var + 0.5
        clamped = torch.clamp(normed, min=0.0)
        dic_reweight_avg_entropy_normalized = dict(zip(keys, clamped.tolist()))
        
        return dic_reweight_avg_entropy_normalized

    def get_feature_from_boxes(self, pred_boxes_shrink, sp_coord, sp_feat):
        voxel_size = self.cfg.MODEL.VOXEL_SIZE
        feat_len, dev = sp_feat.shape[-1], sp_feat.device
        detected_feature_list = []

        if pred_boxes_shrink.shape[0] == 0:
            detected_feature_list.append(torch.zeros(feat_len, device=dev))
            return detected_feature_list
        
        pred_boxes = copy.deepcopy(pred_boxes_shrink)
        pred_boxes[:, 0:6] = pred_boxes[:, 0:6] / voxel_size
        

        for box_id in range(pred_boxes.shape[0]):
            box = pred_boxes[box_id]
            _, _, _, _, _, _, a = box
            sp_coord_translate = sp_coord[:, 1:4] - box[0:3]
            cos_a = torch.cos(a)
            sin_a = torch.sin(a)
            R = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]], device=dev)
            sp_coord_rotate = torch.matmul(sp_coord_translate, R.T)
            within_box = torch.all(torch.abs(sp_coord_rotate) <= box[3:6]/2, dim=1)
            within_indices = torch.where(within_box)[0]

            if within_indices.shape[0] == 0:
                sp_feat_within_box = torch.zeros(feat_len, device=dev)
            else:
                sp_feat_within_box = torch.mean(sp_feat[within_indices], dim=0)
            
            detected_feature_list.append(sp_feat_within_box)
            
        return detected_feature_list

    def check_stats(self, data):
        device = data.device
        percentiles = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
        
        mean = data.mean().item()
        std_dev = data.std().item()
        min_val = data.min().item()
        max_val = data.max().item()
        median = data.median().item()

        # Calculate the specified percentiles
        percentile_values = torch.quantile(data, torch.tensor(percentiles, device=device))

        # Print the results
        for p, value in zip(percentiles, percentile_values):
            print(f"{int(p * 100)}th percentile: {value.item()}")

        print(f"Mean: {mean}")
        print(f"Standard Deviation: {std_dev}")
        print(f"Minimum: {min_val}")
        print(f"Maximum: {max_val}")
        print(f"Median: {median}")

        return 0

    def check_stats_normalize(self, data):
        device = data.device
        percentiles = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
        
        mean = data.mean().item()
        std_dev = data.std().item()
        min_val = data.min().item()
        max_val = data.max().item()
        median = data.median().item()

        new_data = (data - mean) / (std_dev * 3) + 1/2

        # Calculate the specified percentiles
        percentile_values = torch.quantile(new_data, torch.tensor(percentiles, device=device))

        # calculate new stats
        mean = new_data.mean().item()
        std_dev = new_data.std().item()
        min_val = new_data.min().item()
        max_val = new_data.max().item()
        median = new_data.median().item()

        # Print the results
        for p, value in zip(percentiles, percentile_values):
            print(f"{int(p * 100)}th percentile: {value.item()}")

        print(f"Mean: {mean}")
        print(f"Standard Deviation: {std_dev}")
        print(f"Minimum: {min_val}")
        print(f"Maximum: {max_val}")
        print(f"Median: {median}")

        return 0
    
    def normalize_dict(self, data, ratio, shift):
        values = list(data.values())
        mean = sum(values) / len(values)
        std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5


        normalized_data = {key: min(max((value - mean) / (std * ratio) + shift, 0.), 1.) for key, value in data.items()}

        return normalized_data

    



            