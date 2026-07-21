"""
MAE Pre-training Script for Lifespan Foundation Model

This script handles the first phase of training:
- Pre-train MAE on individual modalities (T1w, T2w, FA, MD) across age groups
  (fetal, neonatal, infant, child, adult, elderly) using the Duo MoE architecture.
- Shared encoder learns general brain anatomy features
- Duo experts (modality x age_group) learn fine-grained specializations
"""

import torch
import torch.optim as optim
import argparse
import os
import time
import wandb
from typing import Dict
from collections import OrderedDict

from model.lifespan_moe_mae import LifespanMoEMAE
from data.get_dataloader import get_dataloader
from cfg.lifespan_config import get_cfg_defaults
from utils.visualizer import Visualizer
from utils import util


# Remove the MAEPretrainingDataset class as it's now handled in data/lifespan_datasets.py


def _build_combo_trackers(modalities, age_groups):
    """Build zero-initialized dicts for (modality, age_group) combo tracking."""
    combos = [f'{m}_{a}' for m in modalities for a in age_groups]
    return {c: 0.0 for c in combos}, {c: 0 for c in combos}


def _update_combo_metrics(combo_losses, combo_counts, modality_ids, age_group_ids,
                          loss_val):
    """Accumulate per-combo loss metrics from a batch."""
    if isinstance(modality_ids, (list, tuple)):
        for mod, ag in zip(modality_ids, age_group_ids):
            key = f'{mod}_{ag}'
            if key in combo_losses:
                n = len(modality_ids)
                combo_losses[key] += loss_val / n
                combo_counts[key] += 1 / n
    else:
        mod = modality_ids if isinstance(modality_ids, str) else str(modality_ids)
        ag = age_group_ids if isinstance(age_group_ids, str) else str(age_group_ids)
        key = f'{mod}_{ag}'
        if key in combo_losses:
            combo_losses[key] += loss_val
            combo_counts[key] += 1


def train_mae_epoch(model: LifespanMoEMAE, dataloader,
                    optimizer: optim.Optimizer, device: torch.device,
                    mask_ratio: float = 0.75, epoch: int = 0,
                    print_freq: int = 100) -> Dict[str, float]:
    """Train MAE for one epoch with Duo MoE (modality x age_group tracking)."""
    model.train()
    total_loss = 0.0
    total_local_loss = 0.0
    total_global_loss = 0.0
    modalities = model.modalities
    age_groups = model.age_groups
    combo_losses, combo_counts = _build_combo_trackers(modalities, age_groups)

    visual_dict = None

    print_start_time = time.time()
    for i, batch in enumerate(dataloader):
        local_patch = batch['local_patch'].to(device)      # [B, 1, H, W, D]
        global_img = batch['global_images'].to(device)     # [B, 1, H, W, D]
        modality_ids = batch['modality']
        age_group_ids = batch['age_group']

        optimizer.zero_grad()

        local_loss, global_loss, local_pred, global_pred, local_mask, global_mask = \
            model.forward_train(
                local_patch=local_patch,
                global_img=global_img,
                modality_ids=modality_ids,
                age_group_ids=age_group_ids,
                mask_ratio=mask_ratio
            )

        loss = local_loss + global_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_local_loss += local_loss.item()
        total_global_loss += global_loss.item()
        _update_combo_metrics(combo_losses, combo_counts,
                              modality_ids, age_group_ids, loss.item())

        visual_dict = {
            'local_patch': batch['local_patch'],
            'local_mask': local_mask.detach(),
            'local_pred': local_pred.detach(),
            'global_scan': batch['global_images'],
            'global_mask': global_mask.detach(),
            'global_pred': global_pred.detach()
        }

        if i % print_freq == 0:
            elapsed_time = time.time() - print_start_time
            print(f"(epoch: {epoch}, iters: {i}, time: {elapsed_time:.3f}) "
                  f"local_MSE: {local_loss.item():.3f} global_MSE: {global_loss.item():.3f}")
            print_start_time = time.time()

    avg_loss = total_loss / len(dataloader)
    avg_combo_losses = {
        k: combo_losses[k] / (combo_counts[k] + 1e-8)
        for k in combo_losses
    }

    return {
        'total_loss': avg_loss,
        'local_loss': total_local_loss / len(dataloader),
        'global_loss': total_global_loss / len(dataloader),
        'combo_losses': avg_combo_losses,
        'visual_dict': visual_dict
    }


def save_visualization(visual_dict, cfg, visualizer, epoch):
    """Save visualization using proper visualizer - matching original solver"""
    if visual_dict is None:
        return

    # Create visualization directory (needed for nifti saving)
    vis_dir = os.path.join(cfg.system.ckpt_dir, cfg.system.project, cfg.system.exp_name, 'visulization')
    os.makedirs(vis_dir, exist_ok=True)

    try:
        # Save nifti first (like original solver)
        if cfg.system.save_nii:
            util.save_nii(visual_dict, vis_dir, epoch)

        # Process visualization data following original solver approach
        vis = OrderedDict()
        slc_num = cfg.data.patch_size[-1] // 2

        for k, v in visual_dict.items():
            if v is not None and hasattr(v, 'shape') and len(v.shape) == 5:  # [B, C, H, W, D]
                if 'mask' in k:
                    # For masks, use tensor2label (binary: 0/1 -> 0/255)
                    vis[k] = util.tensor2label(v[0, :, :, :, slc_num], 2)
                else:
                    # For images/predictions, use tensor2im (normalize to 0-255)
                    vis[k] = util.tensor2im(v[0, :, :, :, slc_num])

        # Use the proper visualizer to save images and create HTML
        visualizer.display_current_results(vis, epoch)

    except Exception as e:
        print(f"Warning: Could not save visualizations: {e}")


def validate_mae(model: LifespanMoEMAE, dataloader,
                 device: torch.device, mask_ratio: float = 0.75) -> Dict[str, float]:
    """Validate MAE with Duo MoE."""
    model.eval()
    total_loss = 0.0
    total_local_loss = 0.0
    total_global_loss = 0.0
    combo_losses, combo_counts = _build_combo_trackers(model.modalities, model.age_groups)

    with torch.no_grad():
        for batch in dataloader:
            local_patch = batch['local_patch'].to(device)
            global_img = batch['global_images'].to(device)
            modality_ids = batch['modality']
            age_group_ids = batch['age_group']

            local_loss, global_loss, _, _, _, _ = model.forward_train(
                local_patch=local_patch,
                global_img=global_img,
                modality_ids=modality_ids,
                age_group_ids=age_group_ids,
                mask_ratio=mask_ratio
            )

            loss = local_loss + global_loss
            total_loss += loss.item()
            total_local_loss += local_loss.item()
            total_global_loss += global_loss.item()
            _update_combo_metrics(combo_losses, combo_counts,
                                  modality_ids, age_group_ids, loss.item())

    avg_combo_losses = {
        k: combo_losses[k] / (combo_counts[k] + 1e-8)
        for k in combo_losses
    }

    return {
        'total_loss': total_loss / len(dataloader),
        'local_loss': total_local_loss / len(dataloader),
        'global_loss': total_global_loss / len(dataloader),
        'combo_losses': avg_combo_losses
    }


def main():
    parser = argparse.ArgumentParser(description='MAE Pre-training')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (optional)')
    parser.add_argument('--data_root', type=str, default='/labs/wanglab/projects/lifespan-T1T2FA',
                       help='Path to data root directory')
    parser.add_argument('--output_dir', type=str, default='./MAE_checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default='/opt/localdata/data/usr-envs/ruiying/Code/foundation/MoE_foundation_v3/checkpoint/MAE_checkpoints/latest.pth',
                       help='Resume from checkpoint')
    
    args = parser.parse_args()

    # Load config - directly use the YAML config file
    cfg = get_cfg_defaults()
    # Always use the lifespan_mae.yaml config
    yaml_config = '/opt/localdata/data/usr-envs/ruiying/Code/foundation/MoE_foundation_v3/cfg/lifespan_mae.yaml'
    if os.path.exists(yaml_config):
        cfg.merge_from_file(yaml_config)
    elif args.config is not None:
        cfg.merge_from_file(args.config)

    # Override data root
    cfg.data.mae_root = args.data_root
    args.output_dir = cfg.system.ckpt_dir
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize wandb
    if cfg.system.wandb:
        wandb.init(project="lifespan-mae-pretraining", config=cfg)

    # Initialize visualizer
    visualizer = Visualizer(cfg)

    # Create model
    model = LifespanMoEMAE(cfg)
    model = model.to(device)

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create datasets and dataloaders
    train_loader = get_dataloader(cfg, task='mae', split='train')
    val_loader = get_dataloader(cfg, task='mae', split='val')

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=cfg.train.betas
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.epochs
    )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    best_val_loss = float('inf')
    mask_ratio = cfg.train.mask_ratio

    for epoch in range(start_epoch, cfg.train.epochs):
        epoch_start_time = time.time()

        # Train
        train_metrics = train_mae_epoch(
            model, train_loader, optimizer, device, mask_ratio,
            epoch=epoch+1, print_freq=cfg.train.print_freq
        )

        # Validate
        val_metrics = validate_mae(model, val_loader, device, mask_ratio)

        # Conditional scheduler step - only after niter epochs (like original)
        if epoch > cfg.train.niter:
            scheduler.step()
            print('Current Learning Rate changed to {}'.format(optimizer.param_groups[0]['lr']))

        # Save visualization
        if 'visual_dict' in train_metrics and train_metrics['visual_dict'] is not None:
            save_visualization(train_metrics['visual_dict'], cfg, visualizer, epoch+1)

        # End of epoch logging
        epoch_time = time.time() - epoch_start_time
        print(f"End of epoch {epoch+1} / {cfg.train.epochs}\t Time Taken: {int(epoch_time)} sec")

        if cfg.system.wandb:
            log_dict = {
                'epoch': epoch,
                'train/total_loss': train_metrics['total_loss'],
                'train/local_loss': train_metrics['local_loss'],
                'train/global_loss': train_metrics['global_loss'],
                'val/total_loss': val_metrics['total_loss'],
                'val/local_loss': val_metrics['local_loss'],
                'val/global_loss': val_metrics['global_loss'],
                'learning_rate': optimizer.param_groups[0]['lr']
            }

            # Add per-combo (modality x age_group) losses
            for combo in train_metrics['combo_losses']:
                log_dict[f'train/{combo}_loss'] = train_metrics['combo_losses'][combo]
                log_dict[f'val/{combo}_loss'] = val_metrics['combo_losses'][combo]

            wandb.log(log_dict)

        # Additional wandb logging for learning rate (like original)
        if cfg.system.wandb:
            wandb.log({'lr': optimizer.param_groups[0]['lr']})
            wandb.log({'epoch': epoch})

        # Save checkpoint
        is_best = val_metrics['total_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['total_loss']

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'config': cfg
        }

        # Save latest
        torch.save(checkpoint, os.path.join(args.output_dir, 'latest.pth'))

        # Save best
        if is_best:
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_mae.pth'))
            print(f"New best model saved with val loss: {best_val_loss:.4f}")

        # Save periodic
        if (epoch + 1) % cfg.train.save_epoch_freq == 0:
            torch.save(checkpoint, os.path.join(args.output_dir, f'epoch_{epoch}.pth'))

    print(f"MAE pre-training completed. Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()