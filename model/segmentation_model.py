import torch
import torch.nn as nn
from .blocks import DeepLabHead
from .loss import DC_and_CE_loss
from model.lifespan_moe_mae import LifespanMoEMAE
import nibabel as nib
import numpy as np
import os

class Masked_seg(nn.Module):
    """ Masked Autoencoder with ResNet encoder + DeepLab segmentation header
    """

    def __init__(self, cfg):
        super().__init__()

        # --------------------------------------------------------------------------
        # Use LifespanMoEMAE as encoder
        self.cfg = cfg
        embed_dim = self.cfg.model.embed_dim

        # encoder - using LifespanMoEMAE
        self.encoder = LifespanMoEMAE(cfg)

        self.CE = nn.CrossEntropyLoss()
        self.seg_decoder = DeepLabHead(in_channels=embed_dim * 2, aspp_channel=embed_dim, num_classes=cfg.train.cls_num,
                                       ratio=4)

    def patchify(self, imgs, p):
        """

        imgs: (N, 1, H, W, D)
        x: (N, H*W*D/P***3, patch_size**3)
        """
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0 and imgs.shape[4] % p == 0
        h, w, d = [i // p for i in self.cfg.data.patch_size]

        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p, d, p))
        x = torch.einsum('nchpwqdr->nhwdpqrc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p ** 3))
        return x

    def unpatchify(self, x, p):
        """

        x: (N, H*W*D/P***3, patch_size**3)
        imgs: (N, 1, H, W, D)
        """
        h, w, d = [i // p for i in self.cfg.data.patch_size]

        assert h * w * d == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p))
        x = torch.einsum('nhwdpqr->nhpwqdr', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, w * p, d * p))
        return imgs

    def random_masking(self, x, mask_ratio, p):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        x = self.patchify(x, p)

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask_ = torch.zeros_like(x_masked)
        # generate the binary mask: 0 is keep, 1 is remove

        x_empty = torch.zeros((N, L - len_keep, D)).cuda()
        mask = torch.ones_like(x_empty)
        x_ = torch.cat([x_masked, x_empty], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        mask_ = torch.cat([mask_, mask], dim=1)
        mask_ = torch.gather(
            mask_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        x_masked = self.unpatchify(x_, p)

        mask = self.unpatchify(mask_, p)

        return x_masked, mask

    def forward_encoder(self, x, modality, mask_ratio, p, age_group=None):
        """
        Forward encoder using LifespanMoEMAE with modality information

        Args:
            x: Input tensor [B, 1, H, W, D]
            modality: Modality string ('T1w', 'T2w', 'FA', etc.)
            mask_ratio: Masking ratio for MAE-style training
            p: Patch size for masking
            age_group: Age group string ('fetal', 'infant', 'child', 'adult', 'elderly')
        """
        # Use LifespanMoEMAE's forward_encoder_with_moe for MoE-enhanced features
        features, mask = self.encoder.forward_encoder_with_moe(x, modality, age_group, mask_ratio, p)

        if mask_ratio > 0:
            return features, mask
        else:
            return features

    def seg_loss(self, label, pred):
        # this version has confidence mask

        loss = DC_and_CE_loss(
            {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
        loss_seg = loss(pred, label)
        return loss_seg

    def cos_regularization(self, pred, tar):
        loss = nn.CosineEmbeddingLoss()

        return loss(pred.flatten(start_dim=2).squeeze(), tar.flatten(start_dim=2).squeeze(),
                    target=torch.ones((pred.shape[1])).cuda())
    # THIS IS THE MAIN FORWARD FUNCTION to generate pseudo label

    def forward(self, coordinates, local_patch, local_label, global_img, global_label, mask_ratio=0,
                pseudo=False, real_label=True, modality=None, age_group=None):

        # TODO: WARNING: The current implementation only supports batch size of 1
        # the way to extract feature via these coordinates can't be applied to multi batch

        if len(coordinates.shape) == 2 and coordinates.shape[0] == 1:
            coordinates = coordinates[0]
        if not pseudo:
            if real_label:
                # no masked out
                local_latent_1 = self.forward_encoder(
                    local_patch, modality, mask_ratio=0, p=self.cfg.train.local_mae_patch, age_group=age_group)
                global_latent_1 = self.forward_encoder(global_img, modality, mask_ratio=0,
                                                       p=self.cfg.train.global_mae_patch, age_group=age_group)
                global_latent_1_zoomed = global_latent_1[:, :, coordinates[0]:coordinates[1],
                                                         coordinates[2]:coordinates[3],
                                                         coordinates[4]:coordinates[5]].clone()
                upsample = nn.Upsample(
                    size=global_latent_1.shape[2:], mode='trilinear', align_corners=True)
                global_latent_1_zoomed = upsample(global_latent_1_zoomed)

                pred_1 = self.seg_decoder(torch.concat(
                    [local_latent_1, global_latent_1_zoomed], dim=1))
                
                # # Save pred_1 as .nii file for testing
                # pred_1_np = pred_1.detach().cpu().numpy()
                # # Take argmax if pred_1 has multiple channels (class predictions)
                # if pred_1_np.shape[1] > 1:
                #     pred_1_np = np.argmax(pred_1_np, axis=1).astype(np.int16)
                # else:
                #     pred_1_np = pred_1_np.squeeze(1).astype(np.float32)
                # # Remove batch dimension if batch size is 1
                # if pred_1_np.shape[0] == 1:
                #     pred_1_np = pred_1_np[0]
                # # Create output directory if it doesn't exist
                # os.makedirs('test_predictions', exist_ok=True)
                # # Save as NIfTI file
                # pred_1_nii = nib.Nifti1Image(pred_1_np, affine=np.eye(4))
                # nib.save(pred_1_nii, 'test_predictions/pred_1.nii.gz')
                # print(f"Saved pred_1 with shape {pred_1_np.shape} to test_predictions/pred_1.nii.gz")


                loss_1 = self.seg_loss(local_label, pred_1)
                pred_aux_1 = self.seg_decoder(torch.concat(
                    [global_latent_1, global_latent_1], dim=1))
                loss_aux_1 = self.seg_loss(global_label, pred_aux_1)
                loss_feat_1 = self.cos_regularization(
                    local_latent_1, global_latent_1_zoomed)
                if mask_ratio > 0:
                    # with masked out:
                    local_latent_2, mask_local = self.forward_encoder(local_patch, modality, mask_ratio=mask_ratio,
                                                                      p=self.cfg.train.local_mae_patch, age_group=age_group)
                    global_latent_2, _ = self.forward_encoder(global_img, modality, mask_ratio=mask_ratio,
                                                              p=self.cfg.train.global_mae_patch, age_group=age_group)
                    global_latent_2_zoomed = global_latent_2[:, :, coordinates[0]:coordinates[1],
                                                             coordinates[2]:coordinates[3],
                                                             coordinates[4]:coordinates[5]].clone()
                    global_latent_2_zoomed = upsample(global_latent_2_zoomed)
                    pred_2 = self.seg_decoder(torch.concat(
                        [local_latent_2, global_latent_2_zoomed], dim=1))
                    loss_2 = self.seg_loss(local_label, pred_2)

                    pred_aux_2 = self.seg_decoder(torch.concat(
                        [global_latent_2, global_latent_2], dim=1))
                    loss_aux_2 = self.seg_loss(global_label, pred_aux_2)

                    loss_feat_2 = self.cos_regularization(
                        local_latent_2, global_latent_2_zoomed)
                    return loss_1, loss_2, loss_aux_1, loss_aux_2, loss_feat_1, loss_feat_2, pred_1, pred_2, pred_aux_1, mask_local  # , \
                else:
                    return loss_1, loss_aux_1, loss_feat_1, pred_1, pred_aux_1  # , \
            else:
                if mask_ratio > 0:

                    local_latent_2, mask_local = self.forward_encoder(local_patch, modality, mask_ratio=mask_ratio,
                                                                      p=int(self.cfg.train.local_mae_patch), age_group=age_group)
                    global_latent_1, _ = self.forward_encoder(global_img, modality, mask_ratio=mask_ratio,
                                                              p=int(self.cfg.train.global_mae_patch), age_group=age_group)

                    global_latent_1_zoomed = global_latent_1[:, :, coordinates[0]:coordinates[1],
                                                             coordinates[2]:coordinates[3],
                                                             coordinates[4]:coordinates[5]].clone()
                    upsample = nn.Upsample(
                        size=global_latent_1.shape[2:], mode='trilinear', align_corners=True)
                    global_latent_1_zoomed = upsample(global_latent_1_zoomed)
                    pred_2 = self.seg_decoder(torch.concat(
                        [local_latent_2, global_latent_1_zoomed], dim=1))

                    loss_2 = self.seg_loss(local_label, pred_2)
                    loss_feat_1 = self.cos_regularization(
                        local_latent_2, global_latent_1_zoomed)
                    pred_aux = self.seg_decoder(torch.concat(
                        [global_latent_1, global_latent_1], dim=1))
                    loss_aux = self.seg_loss(
                        global_label, pred_aux)

                    return loss_2, pred_2, loss_aux, pred_aux, mask_local, loss_feat_1
                else:

                    local_latent_2 = self.forward_encoder(local_patch, modality, mask_ratio=mask_ratio,
                                                          p=int(self.cfg.train.local_mae_patch), age_group=age_group)
                    global_latent_1 = self.forward_encoder(global_img, modality, mask_ratio=mask_ratio,
                                                           p=int(self.cfg.train.global_mae_patch), age_group=age_group)

                    global_latent_1_zoomed = global_latent_1[:, :, coordinates[0]:coordinates[1],
                                                             coordinates[2]:coordinates[3],
                                                             coordinates[4]:coordinates[5]].clone()
                    upsample = nn.Upsample(
                        size=global_latent_1.shape[2:], mode='trilinear', align_corners=True)
                    global_latent_1_zoomed = upsample(global_latent_1_zoomed)
                    pred_2 = self.seg_decoder(torch.concat(
                        [local_latent_2, global_latent_1_zoomed], dim=1))

                    loss_2 = self.seg_loss(local_label, pred_2)
                    loss_feat_1 = self.cos_regularization(
                        local_latent_2, global_latent_1_zoomed)
                    pred_aux = self.seg_decoder(torch.concat(
                        [global_latent_1, global_latent_1], dim=1))
                    loss_aux = self.seg_loss(
                        global_label, pred_aux)

                    return loss_2, pred_2, loss_aux, pred_aux, loss_feat_1

        elif pseudo:
            local_latent_1 = self.forward_encoder(
                local_patch, modality, mask_ratio=0, p=self.cfg.train.local_mae_patch, age_group=age_group)
            global_latent_1 = self.forward_encoder(
                global_img, modality, mask_ratio=0, p=self.cfg.train.global_mae_patch, age_group=age_group)
            global_latent_1_zoomed = global_latent_1[:, :, coordinates[0]:coordinates[1],
                                                     coordinates[2]:coordinates[3],
                                                     coordinates[4]:coordinates[5]].clone()
            upsample = nn.Upsample(
                size=global_latent_1.shape[2:], mode='trilinear', align_corners=True)
            global_latent_1_zoomed = upsample(global_latent_1_zoomed)
            pred_1 = self.seg_decoder(torch.concat(
                [local_latent_1, global_latent_1_zoomed], dim=1))
            # pred_1 = self.seg_decoder(local_latent_1)
            pred_aux = self.seg_decoder(torch.concat(
                [global_latent_1, global_latent_1], dim=1))
            return pred_1, pred_aux
    def get_logits_masked(self, coordinates, local_patch, global_img, mask_ratio, modality=None, age_group=None):
        """Student masked forward — returns local logits only, no loss.
        Used for the CMC pair modality where no label is available."""
        if len(coordinates.shape) == 2 and coordinates.shape[0] == 1:
            coordinates = coordinates[0]
        local_latent, _ = self.forward_encoder(local_patch, modality, mask_ratio=mask_ratio,
                                               p=self.cfg.train.local_mae_patch, age_group=age_group)
        global_latent, _ = self.forward_encoder(global_img, modality, mask_ratio=mask_ratio,
                                                p=self.cfg.train.global_mae_patch, age_group=age_group)
        global_latent_zoomed = global_latent[:, :, coordinates[0]:coordinates[1],
                                             coordinates[2]:coordinates[3],
                                             coordinates[4]:coordinates[5]].clone()
        upsample = nn.Upsample(size=global_latent.shape[2:], mode='trilinear', align_corners=True)
        global_latent_zoomed = upsample(global_latent_zoomed)
        return self.seg_decoder(torch.concat([local_latent, global_latent_zoomed], dim=1))

    def forward_only1(self, coordinates, local_patch, local_label, global_img, global_label, mask_ratio=0,
                      pseudo=False, real_label=True, modality=None, age_group=None):

        # TODO: WARNING: The current implementation only supports batch size of 1
        # the way to extract feature via these coordinates can't be applied to multi batch

        if len(coordinates.shape) == 2 and coordinates.shape[0] == 1:
            coordinates = coordinates[0]
        if not pseudo:
            if real_label:
                # no masked out
                local_latent_2, mask_local = self.forward_encoder(local_patch, modality, mask_ratio=mask_ratio,
                                                                  p=self.cfg.train.local_mae_patch, age_group=age_group)
                global_latent_2, _ = self.forward_encoder(global_img, modality, mask_ratio=mask_ratio,
                                                          p=self.cfg.train.global_mae_patch, age_group=age_group)
                upsample = nn.Upsample(
                    size=global_latent_2.shape[2:], mode='trilinear', align_corners=True)
                global_latent_2_zoomed = global_latent_2[:, :, coordinates[0]:coordinates[1],
                                                         coordinates[2]:coordinates[3],
                                                         coordinates[4]:coordinates[5]].clone()
                global_latent_2_zoomed = upsample(global_latent_2_zoomed)
                pred_2 = self.seg_decoder(torch.concat(
                    [local_latent_2, global_latent_2_zoomed], dim=1))
                loss_2 = self.seg_loss(local_label, pred_2)

                pred_aux_2 = self.seg_decoder(torch.concat(
                    [global_latent_2, global_latent_2], dim=1))
                loss_aux_2 = self.seg_loss(global_label, pred_aux_2)

                loss_feat_2 = self.cos_regularization(
                    local_latent_2, global_latent_2_zoomed)
                return loss_2, loss_aux_2, loss_feat_2, pred_2, pred_aux_2


class MoESegModel(nn.Module):
    def __init__(self, cfg, pretrained_encoder_path=None):
        """
        MAPSeg model with MoE encoder (following synthesis_model.py pattern)

        Args:
            cfg: Config object
            pretrained_encoder_path: Path to pre-trained MAE checkpoint
        """
        super().__init__()

        self.cfg = cfg

        # Create MoE encoder (following synthesis_model.py pattern)
        from model.lifespan_moe_mae import LifespanMoEMAE
        self.encoder = LifespanMoEMAE(cfg)

        # Load pretrained encoder weights if provided
        if pretrained_encoder_path is not None:
            self._load_pretrained_encoder(pretrained_encoder_path)

        # Create student and teacher heads (decoder parts only)
        # For now, still using Masked_seg which has its own encoder
        # TODO: Refactor to separate encoder and decoder to use self.encoder
        self.teacher = Masked_seg(cfg=cfg)
        self.student = Masked_seg(cfg=cfg)

        # Store references for easy access
        self.teacher_head = self.teacher
        self.student_head = self.student

    def _load_pretrained_encoder(self, pretrained_path):
        """Load pretrained MAE encoder weights (following synthesis_model.py)"""
        print(f"Loading pre-trained MAE encoder from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

        # Load encoder weights with strict=False to allow partial loading
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        missing_keys, unexpected_keys = self.encoder.load_state_dict(
            state_dict, strict=False)

        if missing_keys:
            print(f"Missing keys when loading encoder: {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"Unexpected keys when loading encoder: {len(unexpected_keys)} keys")

        print("✓ Pre-trained MAE encoder loaded successfully")

    def set_freeze_encoder(self, freeze=True):
        """
        Set encoder parameters to trainable or frozen (following synthesis_model.py)
        Args:
            freeze: If True, freeze encoder; if False, make trainable
        """
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
        print(f"Encoder {'frozen' if freeze else 'trainable'}")

    def initialize_load(self):
        """Legacy method for loading pretrained weights into student"""
        self.student.load_state_dict(
            torch.load(self.cfg.model.pretrain_model),
            strict=False)
        print('pretrained weights loaded, %s' % self.cfg.model.pretrain_model)

    def _init_ema_weights(self):

        for param in self.teacher.parameters():
            param.detach_()
        mp = list(self.student.parameters())
        mcp = list(self.teacher.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()
        print('EMA weights initialized')

    @torch.no_grad()
    def _update_ema(self, iter):
        if self.cfg.model.large_scale:
            # for the model that was pretrained on large-scale data > 1000
            if iter < self.cfg.train.warmup*100 + 1000:
                # for the first 10 epochs after warmup
                # iteration per epoch is 100 and is never tuned
                # if that was altered, this part should be changed and the performance might be affected
                alpha_teacher = 0.999
            else:
                alpha_teacher = 0.9999
        else:
            # for the model that was pretrained on small-batch data: dozens to hundreds
            if iter < self.cfg.train.warmup*100 + 1000:  # for the first 10 epochs after warmup
                alpha_teacher = 0.99
            # for the 10-30th epochs after warmup
            elif iter >= self.cfg.train.warmup*100 + 1000 and iter < self.cfg.train.warmup*100 + 3000:
                alpha_teacher = 0.999
            else:
                alpha_teacher = 0.9999

        for ema_param, param in zip(self.teacher.parameters(),
                                    self.student.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]
    # TO GET THE PSEUDO LABEL

    @torch.no_grad()
    def get_pseudo_label(self, local_patch, global_img, coordinates, modality=None, age_group=None):
        pseudo, pseudo_aux = self.teacher(local_patch=local_patch, local_label=None, global_img=global_img,
                                          global_label=None, coordinates=coordinates, pseudo=True,
                                          modality=modality, age_group=age_group)
        return pseudo, pseudo_aux

    @torch.no_grad()
    def get_pseudo_label_and_weight(self, logits):
        ema_softmax = torch.softmax(logits.detach(), dim=1)
        _, pseudo_label = torch.max(ema_softmax, dim=1)

        # Below is a simple way to get the pseudo label with a certain threshold on confidence (prob.)
        # pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        # ps_large_p = pseudo_prob.ge(
        #     0.95).long() == 1
        # pseudo_label *= ps_large_p
        return pseudo_label

    # this is the training loop for source domain

    def train_source(self, cord_src, img_src, label_src, global_src, label_src_aux, src_mask_ratio, modality_src, age_group_src=None):
        if src_mask_ratio > 0:
            seg_loss, seg_loss_masked, seg_loss_aux, seg_loss_aux_masked, cos_feat, cos_feat_masked, pred_seg, pred_seg_masked, pred_aux, mask_seg = \
                self.student(cord_src, img_src, label_src, global_src,
                             label_src_aux, src_mask_ratio, modality=modality_src, age_group=age_group_src)

            return seg_loss, seg_loss_masked, seg_loss_aux, seg_loss_aux_masked, cos_feat, cos_feat_masked, pred_seg, pred_seg_masked, pred_aux, mask_seg
        else:
            seg_loss, seg_loss_aux, cos_feat, pred_seg, pred_aux = \
                self.student(cord_src, img_src, label_src, global_src,
                             label_src_aux, src_mask_ratio, modality=modality_src, age_group=age_group_src)
            return seg_loss, seg_loss_aux, cos_feat, pred_seg, pred_aux

    def train_source_only1(self, cord_src, img_src, label_src, global_src, label_src_aux, src_mask_ratio):

        seg_loss, seg_loss_aux, cos_feat, pred_seg, pred_aux = \
            self.student.forward_only1(cord_src, img_src, label_src, global_src,
                                       label_src_aux, src_mask_ratio)
        return seg_loss, seg_loss_aux, cos_feat, pred_seg, pred_aux

    # THIS IS THE TRAINIGN LOOP FOR TARGET DOMAIN

    def train_pseudo(self, cord_tgt, img_tgt, pseudo_label_loc, global_tgt, pseudo_label_global, trg_mask_ratio, modality_tgt, age_group_tgt=None):
        if trg_mask_ratio > 0:
            pse_seg_loss, pse_seg_pred, pse_seg_loss_aux, pse_seg_pred_aux, pse_seg_mask, pse_cos_feat = \
                self.student(cord_tgt, img_tgt, pseudo_label_loc, global_tgt, pseudo_label_global, trg_mask_ratio,
                             real_label=False, modality=modality_tgt, age_group=age_group_tgt)

            return pse_seg_loss, pse_seg_pred, pse_seg_loss_aux, pse_seg_pred_aux, pse_seg_mask, pse_cos_feat
        else:
            pse_seg_loss, pse_seg_pred, pse_seg_loss_aux, pse_seg_pred_aux, pse_cos_feat = \
                self.student(cord_tgt, img_tgt, pseudo_label_loc, global_tgt, pseudo_label_global, trg_mask_ratio,
                             real_label=False, modality=modality_tgt, age_group=age_group_tgt)

            return pse_seg_loss, pse_seg_pred, pse_seg_loss_aux, pse_seg_pred_aux, pse_cos_feat

    def get_student_logits_masked(self, cord, local_patch, global_img, mask_ratio, modality=None, age_group=None):
        """Student masked forward on a modality with no label — returns logits for CMC loss."""
        return self.student.get_logits_masked(cord, local_patch, global_img, mask_ratio,
                                              modality=modality, age_group=age_group)

    def forward(self, local_patch, global_img, coordinates, modality=None, age_group=None):
        pseudo, pseudo_aux = self.student(local_patch=local_patch, local_label=None, global_img=global_img,
                                          global_label=None, coordinates=coordinates, pseudo=True,
                                          modality=modality, age_group=age_group)
        return pseudo, pseudo_aux
