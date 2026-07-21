"""
Segmentation Solver for MAPSeg training
Uses pre-trained MAE encoder with MoESegModel
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import os
import numpy as np
from collections import OrderedDict

from utils.visualizer import Visualizer
from utils import util


class SegmentationTrainer(nn.Module):
    '''
    Solver for MAPSeg (Masked Pseudo Labeling) Segmentation
    Uses pre-trained MAE encoder with student-teacher framework
    '''

    def __init__(self, model, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.model.cuda()
        self.is_teacher_init = False

        # Optimizer and scheduler will be set externally
        self.optimizer = None
        self.scheduler = None

        self.visualizer = Visualizer(cfg)
        self.vis_dir = os.path.join(
            cfg.system.ckpt_dir, cfg.system.project, cfg.system.exp_name, 'visulization')
        util.mkdirs(self.vis_dir)
        self.val_dir = os.path.join(
            cfg.system.ckpt_dir, cfg.system.project, cfg.system.exp_name, 'validation')
        util.mkdirs(self.val_dir)

        # Track loss/val score here
        self.src_seg_loss = []
        self.src_seg_loss_masked = []
        self.src_seg_loss_aux = []
        self.src_seg_loss_aux_masked = []
        self.src_cos_reg = []
        self.src_cos_reg_masked = []
        self.tgt_pse_seg_loss = []
        self.tgt_pse_seg_loss_aux = []
        self.tgt_cos_reg = []
        self.tgt_cmc_loss = []
        self.val_score = []
        self.val_dice = []
        self.cumulative_no_improve = []
        self.total_step = 0

    def _get_epoch(self):
        return len(self.src_seg_loss)

    def _init_epoch(self):
        self.tmp_src_seg_loss = 0
        self.tmp_src_seg_loss_masked = 0
        self.tmp_src_seg_loss_aux = 0
        self.tmp_src_seg_loss_aux_masked = 0
        self.tmp_src_cos_reg = 0
        self.tmp_src_cos_reg_masked = 0
        self.tmp_tgt_pse_seg_loss = 0
        self.tmp_tgt_pse_seg_loss_aux = 0
        self.tmp_tgt_cos_reg = 0
        self.tmp_tgt_cmc_loss = 0

    def _log_internal_epoch_res(self, steps):
        self.src_seg_loss.append(self.tmp_src_seg_loss/steps)
        self.src_seg_loss_masked.append(self.tmp_src_seg_loss_masked/steps)
        self.src_seg_loss_aux.append(self.tmp_src_seg_loss_aux/steps)
        self.src_seg_loss_aux_masked.append(
            self.tmp_src_seg_loss_aux_masked/steps)
        self.src_cos_reg.append(self.tmp_src_cos_reg/steps)
        self.src_cos_reg_masked.append(self.tmp_src_cos_reg_masked/steps)
        self.tgt_pse_seg_loss.append(self.tmp_tgt_pse_seg_loss/steps)
        self.tgt_pse_seg_loss_aux.append(self.tmp_tgt_pse_seg_loss_aux/steps)
        self.tgt_cos_reg.append(self.tmp_tgt_cos_reg/steps)
        self.tgt_cmc_loss.append(self.tmp_tgt_cmc_loss/steps)

    def _get_internal_loss(self):
        return {
            'src_seg_loss': self.src_seg_loss[-1],
            'src_seg_loss_masked': self.src_seg_loss_masked[-1],
            'src_seg_loss_aux': self.src_seg_loss_aux[-1],
            'src_seg_loss_aux_masked': self.src_seg_loss_aux_masked[-1],
            'src_cos_reg': self.src_cos_reg[-1],
            'src_cos_reg_masked': self.src_cos_reg_masked[-1],
            'tgt_pse_seg_loss': self.tgt_pse_seg_loss[-1],
            'tgt_pse_seg_loss_aux': self.tgt_pse_seg_loss_aux[-1],
            'tgt_cos_reg': self.tgt_cos_reg[-1],
            'tgt_cmc_loss': self.tgt_cmc_loss[-1],
        }

    def train_step(self, data, epoch):
        """Training step following MAPSeg solver pattern"""
        torch.cuda.empty_cache()
        self.total_step += 1
        if epoch > self.cfg.train.warmup:
            self.model._update_ema(self.total_step)
        self.model.train()

        img_src = data['imgB'].float().cuda()
        img_src[img_src < 0] = 0
        global_src = data['downB'].float().cuda()
        global_src[global_src < 0] = 0
        label_src = data['labelB'].long().cuda()
        label_src_aux = data['label_B_aux'].long().cuda()
        cord_src = data['cord_B']
        img_tgt = data['imgA'].float().cuda()
        img_tgt[img_tgt < 0] = 0
        global_tgt = data['downA'].float().cuda()
        global_tgt[global_tgt < 0] = 0
        cord_tgt = data['cord_A']
        modality_src = data['modality_B']  # Source domain modality
        modality_tgt = data['modality_A']  # Target domain modality
        age_group_src = data.get('age_group_B', None)
        age_group_tgt = data.get('age_group_A', None)

        if epoch <= self.cfg.train.warmup:
            # Warmup phase - train only on source
            seg_loss, seg_loss_masked, seg_loss_aux, seg_loss_aux_masked, cos_feat, cos_feat_masked, pred_seg, pred_seg_masked, pred_aux, mask_seg = \
                self.model.train_source(cord_src, img_src, label_src,
                                      global_src, label_src_aux, self.cfg.train.mask_ratio, modality_src,
                                      age_group_src=age_group_src)
            self.loss_dict = dict(zip(['src_seg_loss', 'src_seg_loss_masked', 'src_seg_loss_aux', 'src_seg_loss_aux_masked', 'src_cos_reg', 'src_cos_reg_masked'],
                                    [seg_loss.item(), seg_loss_masked.item(), seg_loss_aux.item(), seg_loss_aux_masked.item(), cos_feat.item(), cos_feat_masked.item()]))
            self.tmp_src_seg_loss += seg_loss.item()
            self.tmp_src_seg_loss_masked += seg_loss_masked.item()
            self.tmp_src_seg_loss_aux += seg_loss_aux.item()
            self.tmp_src_seg_loss_aux_masked += seg_loss_aux_masked.item()
            self.tmp_src_cos_reg += cos_feat.item()
            self.tmp_src_cos_reg_masked += cos_feat_masked.item()
            loss = (seg_loss + seg_loss_masked) * 0.5 + (seg_loss_aux + seg_loss_aux_masked) * 0.5 * 0.1 + (
                cos_feat + cos_feat_masked) * 0.5 * 0.05

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.visual_dict = dict(zip(['src_local', 'src_global', 'src_local_label', 'src_global_label',
                                       'src_local_pred', 'src_local_pred_masked', 'src_global_pred',
                                       'src_masked_map'],
                                      [img_src.detach(), global_src.detach(), label_src.detach(), label_src_aux.detach(),
                                       pred_seg.detach(), pred_seg_masked.detach(), pred_aux.detach(),
                                       mask_seg.detach()]))
        else:
            # MPL phase - train on source + pseudo target
            if not self.is_teacher_init:
                self.model._init_ema_weights()
                self.is_teacher_init = True

            seg_loss, seg_loss_masked, seg_loss_aux, seg_loss_aux_masked, cos_feat, cos_feat_masked, pred_seg, pred_seg_masked, pred_aux, mask_seg = \
                self.model.train_source(cord_src, img_src, label_src,
                                      global_src, label_src_aux, self.cfg.train.mask_ratio, modality_src,
                                      age_group_src=age_group_src)
            pseudo_label_loc_logit, pseudo_label_global_logit = self.model.get_pseudo_label(
                img_tgt, global_tgt, cord_tgt, modality=modality_tgt, age_group=age_group_tgt)
            pseudo_label_loc = self.model.get_pseudo_label_and_weight(
                pseudo_label_loc_logit)
            pseudo_label_global = self.model.get_pseudo_label_and_weight(
                pseudo_label_global_logit)
            del pseudo_label_loc_logit, pseudo_label_global_logit

            # Train on pseudo dataset
            pse_seg_loss, pse_seg_pred, pse_seg_loss_aux, pse_seg_pred_aux, pse_seg_mask, pse_cos_feat = \
                self.model.train_pseudo(cord_tgt, img_tgt, pseudo_label_loc.long().cuda(), global_tgt, pseudo_label_global.long().cuda(),
                                      self.cfg.train.mask_ratio, modality_tgt, age_group_tgt=age_group_tgt)
            loss = (seg_loss + seg_loss_masked) * 0.5 + (seg_loss_aux + seg_loss_aux_masked) * 0.5 * 0.1 + \
                (cos_feat + cos_feat_masked) * 0.5 * 0.05 + \
                (pse_seg_loss + 0.1 * pse_seg_loss_aux + pse_cos_feat * 0.05)

            # CMC loss: cross-modality consistency via KL divergence.
            # imgA2 is either a real paired modality or a noisy copy of imgA —
            # so every sample contributes (no gating needed).
            cmc_weight = getattr(self.cfg.train, 'cmc_weight', 0.0)
            if cmc_weight > 0:
                img_tgt2 = data['imgA2'].float().cuda()
                img_tgt2[img_tgt2 < 0] = 0
                global_tgt2 = data['downA2'].float().cuda()
                global_tgt2[global_tgt2 < 0] = 0
                modality_tgt2 = data['modality_A2']
                # Student logits for both modalities of the same subject at the same location
                logits_A = self.model.get_student_logits_masked(
                    cord_tgt, img_tgt, global_tgt, self.cfg.train.mask_ratio,
                    modality_tgt, age_group_tgt)
                logits_A2 = self.model.get_student_logits_masked(
                    cord_tgt, img_tgt2, global_tgt2, self.cfg.train.mask_ratio,
                    modality_tgt2, age_group_tgt)
                # KL(P_A2 || P_A): push paired modality toward the primary modality's distribution
                loss_cmc = F.kl_div(
                    F.log_softmax(logits_A2, dim=1),
                    F.softmax(logits_A.detach(), dim=1),
                    reduction='mean')
                loss = loss + cmc_weight * loss_cmc
            else:
                loss_cmc = torch.tensor(0.0)

            self.loss_dict = dict(zip(['src_seg_loss', 'src_seg_loss_masked', 'src_seg_loss_aux', 'src_seg_loss_aux_masked', 'src_cos_reg', 'src_cos_reg_masked',
                                     'tgt_pse_seg_loss', 'tgt_pse_seg_loss_aux', 'tgt_cos_reg', 'tgt_cmc_loss'],
                                    [seg_loss.item(), seg_loss_masked.item(), seg_loss_aux.item(), seg_loss_aux_masked.item(), cos_feat.item(), cos_feat_masked.item(),
                                     pse_seg_loss.item(), pse_seg_loss_aux.item(), pse_cos_feat.item(), loss_cmc.item()]))
            self.tmp_src_seg_loss += seg_loss.item()
            self.tmp_src_seg_loss_masked += seg_loss_masked.item()
            self.tmp_src_seg_loss_aux += seg_loss_aux.item()
            self.tmp_src_seg_loss_aux_masked += seg_loss_aux_masked.item()
            self.tmp_src_cos_reg += cos_feat.item()
            self.tmp_src_cos_reg_masked += cos_feat_masked.item()
            self.tmp_tgt_pse_seg_loss += pse_seg_loss.item()
            self.tmp_tgt_pse_seg_loss_aux += pse_seg_loss_aux.item()
            self.tmp_tgt_cos_reg += pse_cos_feat.item()
            self.tmp_tgt_cmc_loss += loss_cmc.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.visual_dict = dict(zip(['src_local', 'src_global', 'src_local_label', 'src_global_label',
                                       'src_local_pred', 'src_local_pred_masked', 'src_global_pred',
                                       'src_masked_map', 'tgt_local', 'tgt_global', 'tgt_local_pse_label', 'tgt_global_pse_label',
                                       'tgt_local_pred', 'tgt_global_pred',
                                       'tgt_masked_map'],
                                      [img_src.detach(), global_src.detach(), label_src.detach(), label_src_aux.detach(),
                                      pred_seg.detach(), pred_seg_masked.detach(), pred_aux.detach(),
                                      mask_seg.detach(), img_tgt.detach(), global_tgt.detach(
                                      ), pseudo_label_loc.detach(), pseudo_label_global.detach(),
              pse_seg_pred.detach(), pse_seg_pred_aux.detach(), pse_seg_mask.detach()]))

    def get_cur_loss(self):
        return self.loss_dict

    def print_cur_loss(self, epoch, epoch_iter, start_time):
        errors = {k: v if not isinstance(
            v, int) else v for k, v in self.loss_dict.items()}
        t = (time.time() - start_time) / self.cfg.train.print_freq
        self.visualizer.print_current_errors(epoch, epoch_iter, errors, t)

    def save_visualization(self, epoch):
        vis = OrderedDict()
        slc_num = self.cfg.data.patch_size[-1]//2
        for k, v in self.visual_dict.items():
            if 'masked_map' in k:
                vis[k] = util.tensor2label(v[0, :, :, :, slc_num], 2)
                self.visual_dict[k] = v.squeeze().float()
            elif 'pred' in k:
                v = F.softmax(v, dim=1)
                v = torch.argmax(v, dim=1)
                self.visual_dict[k] = v.squeeze().float()
                vis[k] = util.tensor2label(
                    v.squeeze()[:, :, slc_num].unsqueeze(0), self.cfg.train.cls_num)
            elif 'label' in k:
                vis[k] = util.tensor2label(
                    v.squeeze()[:, :, slc_num].unsqueeze(0), self.cfg.train.cls_num)
                self.visual_dict[k] = v.squeeze().float()
            else:
                vis[k] = util.tensor2im(v[0, :, :, :, slc_num])
                self.visual_dict[k] = v.squeeze().float()

        if self.cfg.system.save_nii:
            util.save_nii(self.visual_dict, self.vis_dir, epoch)

        self.visualizer.display_current_results(vis, epoch)

    def scheduler_step(self):
        self.scheduler.step()
        print('Current Learning Rate changed to {}'.format(
            self.optimizer.param_groups[0]['lr']))

    @torch.no_grad()
    def infer_single_scan(self, tmp_scans, modality=None, age_group=None):
        import torchio as tio

        pad_flag = False
        self.model.eval()
        x, y, z = self.cfg.data.patch_size
        if self.cfg.data.normalize:
            tmp_scans = util.norm_img(tmp_scans, self.cfg.data.norm_perc)
        if min(tmp_scans.shape) < min(x, y, z):
            x_ori_size, y_ori_size, z_ori_size = tmp_scans.shape
            pad_flag = True
            x_diff = x-x_ori_size
            y_diff = y-y_ori_size
            z_diff = z-z_ori_size
            tmp_scans = np.pad(tmp_scans, ((max(0, int(x_diff/2)), max(0, x_diff-int(x_diff/2))), (max(0, int(
                y_diff/2)), max(0, y_diff-int(y_diff/2))), (max(0, int(z_diff/2)), max(0, z_diff-int(z_diff/2)))), constant_values=1e-4)

        pred = np.zeros((self.cfg.train.cls_num,) + tmp_scans.shape)
        tmp_norm = np.zeros((self.cfg.train.cls_num,) + tmp_scans.shape)

        scan_patches, _, tmp_idx = util.patch_slicer(tmp_scans, tmp_scans, self.cfg.data.patch_size,
                                                     (x - 16, y -
                                                      16, z - 16),
                                                     remove_bg=self.cfg.data.remove_bg, test=True, ori_path=None)
        bound = util.get_bounds(torch.from_numpy(tmp_scans))
        global_scan = torch.unsqueeze(torch.from_numpy(
            tmp_scans).to(dtype=torch.float), dim=0)

        # Sliding window implementation
        for idx, patch in enumerate(scan_patches):
            ipt = torch.from_numpy(patch).to(dtype=torch.float).cuda()
            ipt = ipt.reshape((1, 1,) + ipt.shape)

            patch_idx = tmp_idx[idx]
            location = torch.zeros_like(
                torch.from_numpy(tmp_scans)).float()
            location = torch.unsqueeze(location, 0)
            location[:, patch_idx[0]:patch_idx[1], patch_idx[2]:patch_idx[3], patch_idx[4]:patch_idx[5]] = 1

            # When brain bounding box is smaller than patch in a dimension, use the
            # local patch extent for the global crop so spatial correspondence is correct
            gx0 = patch_idx[0] if (bound[1] - bound[0] < x) else bound[0]
            gx1 = patch_idx[1] if (bound[1] - bound[0] < x) else bound[1]
            gy0 = patch_idx[2] if (bound[3] - bound[2] < y) else bound[2]
            gy1 = patch_idx[3] if (bound[3] - bound[2] < y) else bound[3]
            gz0 = patch_idx[4] if (bound[5] - bound[4] < z) else bound[4]
            gz1 = patch_idx[5] if (bound[5] - bound[4] < z) else bound[5]

            sbj = tio.Subject(one_image=tio.ScalarImage(
                tensor=global_scan[:, gx0:gx1, gy0:gy1, gz0:gz1]),
                a_segmentation=tio.LabelMap(
                    tensor=location[:, gx0:gx1, gy0:gy1, gz0:gz1]))
            transforms = tio.transforms.Resize(target_shape=(x, y, z))
            sbj = transforms(sbj)
            down_scan = sbj['one_image'].data
            loc = sbj['a_segmentation'].data
            tmp_coor = util.get_bounds(loc)
            coordinates_A = np.array([np.floor(tmp_coor[0] / 4),
                                      np.ceil(tmp_coor[1] / 4),
                                      np.floor(tmp_coor[2] / 4),
                                      np.ceil(tmp_coor[3] / 4),
                                      np.floor(tmp_coor[4] / 4),
                                      np.ceil(tmp_coor[5] / 4)
                                      ]).astype(int)
            coordinates_A = torch.unsqueeze(
                torch.from_numpy(coordinates_A), 0)
            tmp_pred, _ = self.model(ipt, down_scan.cuda().reshape([1, 1, x, y, z]),
                                     coordinates_A, modality=modality, age_group=age_group)

            patch_idx = (slice(0, self.cfg.train.cls_num),) + (
                slice(patch_idx[0], patch_idx[1]), slice(
                    patch_idx[2], patch_idx[3]),
                slice(patch_idx[4], patch_idx[5]))
            pred[patch_idx] += torch.squeeze(
                tmp_pred).detach().cpu().numpy()
            tmp_norm[patch_idx] += 1

        pred[tmp_norm > 0] = (pred[tmp_norm > 0]) / \
            tmp_norm[tmp_norm > 0]
        sf = torch.nn.Softmax(dim=0)
        pred_vol = sf(torch.from_numpy(pred)).numpy()
        pred_vol = np.argmax(pred_vol, axis=0)
        if pad_flag:
            pred_vol = pred_vol[max(0, int(x_diff/2)): max(0, int(x_diff/2))+x_ori_size,
                                max(0, int(y_diff/2)): max(0, int(y_diff/2))+y_ori_size,
                                max(0, int(z_diff/2)): max(0, int(z_diff/2))+z_ori_size]
            assert pred_vol.shape == (
                x_ori_size, y_ori_size, z_ori_size), 'pred_vol shape must be the same as the original scan shape'
        return pred_vol

    def validation(self, epoch):
        import nibabel as nib
        import re
        from data.preprocess import preprocess_image

        if not self.cfg.train.test_time:
            from data.lifespan_datasets import get_modality_from_filename, parse_age_to_years, age_years_to_group
            import pandas as _pd

            # Build age lookup from val demographics
            val_age_lookup = {}
            val_demo_path = getattr(self.cfg.data, 'val_demographics', '')
            if val_demo_path and os.path.isfile(val_demo_path):
                _df = _pd.read_csv(val_demo_path)
                for _, _row in _df.iterrows():
                    _sub = str(_row.get('sub_id', '')).strip()
                    _ses_raw = _row.get('ses_id', '')
                    _ses = '' if (_ses_raw is None or (isinstance(_ses_raw, float) and np.isnan(_ses_raw))) else str(_ses_raw).strip()
                    _age = age_years_to_group(parse_age_to_years(_row.get('age')))
                    val_age_lookup[(_sub, _ses)] = _age
                    # also store without ses as fallback (for filenames that have no ses-)
                    if _ses:
                        val_age_lookup.setdefault((_sub, ''), _age)

            # Collect all _img/_label folder pairs from val_data root
            val_data_root = getattr(self.cfg.data, 'val_data', None)
            if val_data_root and os.path.isdir(val_data_root):
                img_folders = sorted([
                    d for d in os.listdir(val_data_root)
                    if d.endswith('_img') and os.path.isdir(os.path.join(val_data_root, d))
                ])
            else:
                # Fallback to legacy single-folder config
                img_folders = [os.path.basename(self.cfg.data.val_img)]
                val_data_root = os.path.dirname(self.cfg.data.val_img)

            supported_mods = set(self.cfg.model.modalities)
            cur_dsc = 0
            total_files = 0

            for img_folder in img_folders:
                img_dir   = os.path.join(val_data_root, img_folder)
                label_dir = os.path.join(val_data_root, img_folder.replace('_img', '_label'))
                if not os.path.isdir(label_dir):
                    print(f'[WARNING] No label folder for {img_folder}, skipping')
                    continue

                # Modality from folder name (e.g. BCP_FA_img -> FA)
                val_modality = get_modality_from_filename(img_folder, supported_mods) or 'T1w'

                val_lst = [f for f in os.listdir(img_dir) if f.endswith(self.cfg.data.extension)]
                for val_file in val_lst:
                    img_path   = os.path.join(img_dir, val_file)
                    label_file = val_file.replace('.nii.gz', '_tissue.nii.gz')
                    label_path = os.path.join(label_dir, label_file)
                    if not os.path.isfile(label_path):
                        continue

                    tmp_scans = preprocess_image(img_path, target_spacing=(1.0, 1.0, 1.0))
                    tmp_label = preprocess_image(label_path, target_spacing=(1.0, 1.0, 1.0), is_label=True)
                    tmp_scans = np.squeeze(tmp_scans)
                    tmp_label = np.squeeze(np.round(tmp_label))
                    tmp_scans[tmp_scans < 0] = 0

                    if 14 in np.unique(tmp_label):
                        tmp_label[tmp_label == 14] = 0
                    tmp_label[tmp_label == 15] = 14

                    # Age group from demographics
                    sub_m = re.search(r'(sub-[^_.\s]+)', val_file)
                    ses_m = re.search(r'(ses-[^_.\s]+)', val_file)
                    sub_key = sub_m.group(1) if sub_m else val_file.replace('.nii.gz', '')
                    ses_key = ses_m.group(1) if ses_m else ''
                    age_group = val_age_lookup.get((sub_key, ses_key),
                                 val_age_lookup.get((sub_key, ''), 'adult'))

                    tmp_pred = self.infer_single_scan(tmp_scans, modality=val_modality, age_group=age_group)
                    ind_dsc = []
                    for cls_idx in range(1, self.cfg.train.cls_num):
                        ind_dsc.append(util.cal_dice(tmp_pred, tmp_label, cls_idx))
                    cur_dsc += np.mean(ind_dsc)
                    total_files += 1

                    if self.cfg.system.save_nii:
                        _label_nii = nib.Nifti1Image(tmp_label, np.eye(4))
                        _pred_nii  = nib.Nifti1Image(tmp_pred.astype(np.uint8), np.eye(4))
                        prefix = f'{epoch}_{img_folder}_{val_file.split(".")[0]}'
                        nib.save(_label_nii, os.path.join(self.val_dir, prefix + '_label.nii.gz'))
                        nib.save(_pred_nii,  os.path.join(self.val_dir, prefix + '_pred.nii.gz'))

            cur_dsc /= max(total_files, 1)
            self.val_dice.append(cur_dsc)

            tmp_val_score = cur_dsc * 1 - self.tgt_pse_seg_loss[-1]*0.5

        else:
            self.val_dice.append(0)
            tmp_val_score = - self.tgt_pse_seg_loss[-1]
        if len(self.val_score) == 0:
            self.val_score.append(tmp_val_score)
            self.cumulative_no_improve.append(0)
            save_best = True
        else:
            if tmp_val_score > max(self.val_score):
                self.cumulative_no_improve.append(0)
                save_best = True
            else:
                save_best = False
                self.cumulative_no_improve.append(
                    self.cumulative_no_improve[-1]+1)
            self.val_score.append(tmp_val_score)

        return save_best
