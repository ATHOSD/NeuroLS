"""
MAPSeg Training Script for Lifespan Foundation Model

This script handles MAPSeg (Masked Pseudo-Labeling) segmentation using pre-trained MAE encoder.
Main function structure follows train_synthesis.py for consistency.
Uses SegmentationTrainer solver from solver/seg_solver.py for the training loop.
"""

import torch
import torch.optim as optim
import argparse
import os
import time
import wandb
import numpy as np
import random
import torch.backends.cudnn as cudnn
from datetime import datetime

from model.segmentation_model import MoESegModel
from solver.seg_solver import SegmentationTrainer
from data.get_dataloader import get_dataloader
from cfg.lifespan_config import get_cfg_defaults


def set_random_seed(seed):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def set_encoder_trainable(model, trainable=True):
    """
    Set encoder parameters to trainable or frozen
    Args:
        model: MoESegModel
        trainable: If True, encoder is trainable; if False, frozen
    """
    model.set_freeze_encoder(freeze=not trainable)
    print(f"Encoder trainable: {trainable}")


def main():
    parser = argparse.ArgumentParser(description='MAPSeg Segmentation Training')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (optional)')
    parser.add_argument('--pretrained', type=str, required=False,
                       help='Path to pre-trained MAE checkpoint')
    parser.add_argument('--encoder_mode', type=str, default='frozen',
                       choices=['frozen', 'warmup', 'finetune'],
                       help='Encoder training mode: frozen (freeze), warmup (low lr), or finetune (full lr)')
    parser.add_argument('--warmup_lr_factor', type=float, default=0.1,
                       help='Learning rate factor for encoder warmup (default: 0.1)')
    parser.add_argument('--output_dir', type=str, default='./segmentation_checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')

    args = parser.parse_args()

    # Load config
    cfg = get_cfg_defaults()

    # Set defaults (following train_synthesis.py structure)
    args.encoder_mode = 'finetune'
    args.pretrained = '/opt/localdata/data/usr-envs/ruiying/Code/foundation/MoE_foundation_final/MAE_checkpoint/best_mae.pth'
    args.output_dir = '/opt/localdata/data/usr-envs/ruiying/Code/foundation/MoE_foundation_final/segmentation_checkpoints'
    yaml_config = '/opt/localdata/data/usr-envs/ruiying/Code/foundation/MoE_foundation_final/cfg/lifespan_segmentation.yaml'
    #args.resume = '/opt/localdata/data/usr-envs/ruiying/Code/foundation/MoE_foundation_final/segmentation_choa_finetune_checkpoints_v2/PROJ/Lifespan_Segmentation/solver_latest.pth'  # Do not hardcode; allow auto-resume from latest checkpoint

    if os.path.exists(yaml_config):
        cfg.merge_from_file(yaml_config)
        print('loaded configuration file {}'.format(yaml_config))
    elif args.config is not None:
        cfg.merge_from_file(args.config)
        print('loaded configuration file {}'.format(args.config))
    else:
        print('using default configuration')

    cfg.freeze()
    if cfg.system.seed is not None:
        set_random_seed(cfg.system.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the checkpoint folder and save the config file
    ckpt_fld = os.path.join(cfg.system.ckpt_dir,
                            cfg.system.project, cfg.system.exp_name)

    if not os.path.exists(ckpt_fld):
        os.makedirs(ckpt_fld)

    if not os.path.exists(os.path.join(ckpt_fld, 'train_cfg.yaml')):
        with open(os.path.join(ckpt_fld, 'train_cfg.yaml'), 'w') as f:
            f.write(cfg.dump())
            f.close()
    else:
        time_now = datetime.now()
        cfg_fname = os.path.join(
            ckpt_fld, 'train_cfg_' + time_now.strftime('%Y%m%d_%H%M%S') + '.yaml')
        with open(cfg_fname, 'w') as f:
            f.write(cfg.dump())
            f.close()

    # Create output directory
    args.output_dir = ckpt_fld
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize wandb
    if cfg.system.wandb:
        wandb.init(
            project=cfg.system.project,
            name=cfg.system.exp_name,
            config=cfg
        )

    # ============================================
    # Create MAPSeg model with pre-trained MAE encoder (following train_synthesis.py pattern)
    # ============================================
    # MoESegModel creates encoder internally and loads pretrained weights
    model = MoESegModel(cfg, pretrained_encoder_path=args.pretrained)
    model = model.to(device)

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Set encoder trainable mode (following train_synthesis.py)
    encoder_frozen = (args.encoder_mode == 'frozen')
    set_encoder_trainable(model, trainable=not encoder_frozen)

    # ============================================
    # Create Optimizer (following train_synthesis.py pattern)
    # ============================================
    if args.encoder_mode == 'frozen':
        # Only optimize student head (encoder frozen)
        optimizer = optim.AdamW(
            model.student_head.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            betas=cfg.train.betas
        )
    elif args.encoder_mode == 'warmup':
        # Encoder with lower learning rate + student head with full learning rate
        optimizer = optim.AdamW(
            [
                {'params': model.encoder.parameters(), 'lr': cfg.train.lr * args.warmup_lr_factor},
                {'params': model.student_head.parameters(), 'lr': cfg.train.lr}
            ],
            weight_decay=cfg.train.weight_decay,
            betas=cfg.train.betas
        )
    else:  # finetune
        # All parameters with same learning rate
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            betas=cfg.train.betas
        )

    # ============================================
    # Create Scheduler
    # ============================================
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-8
    )

    # ============================================
    # Create Solver
    # ============================================
    cudnn.benchmark = True
    train_solver = SegmentationTrainer(model, cfg)
    train_solver.optimizer = optimizer
    train_solver.scheduler = scheduler

    # To detect if there is existing checkpoint (following train.py line 76-79)
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        train_solver.model.load_state_dict(resume_checkpoint['model_state_dict'])
        train_solver.optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        train_solver.scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
        train_solver.is_teacher_init = resume_checkpoint.get('is_teacher_init', False)
        # Restore solver state
        for key in ['src_seg_loss', 'src_seg_loss_masked', 'val_dice', 'val_score', 'cumulative_no_improve']:
            if key in resume_checkpoint:
                setattr(train_solver, key, resume_checkpoint[key])
        print('Loaded solver checkpoint from {}'.format(args.resume))
        print('Previous epochs: ' + str(train_solver._get_epoch()))
    elif os.path.exists(os.path.join(ckpt_fld, 'solver_latest.pth')):
        resume_checkpoint = torch.load(os.path.join(ckpt_fld, 'solver_latest.pth'), map_location='cpu', weights_only=False)
        train_solver.model.load_state_dict(resume_checkpoint['model_state_dict'])
        train_solver.optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        train_solver.scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
        train_solver.is_teacher_init = resume_checkpoint.get('is_teacher_init', False)
        for key in ['src_seg_loss', 'src_seg_loss_masked', 'val_dice', 'val_score', 'cumulative_no_improve']:
            if key in resume_checkpoint:
                setattr(train_solver, key, resume_checkpoint[key])
        print('Loaded the latest solver checkpoint')
        print('Previous epochs: ' + str(train_solver._get_epoch()))

    # Get data loader (following train.py line 87)
    train_loader = get_dataloader(cfg, task='segmentation', split='train')

    # Set up validation parameters (following train.py line 92-94)
    run_val = False
    if cfg.train.type == 'mpl':
        run_val = True

    start_epoch = 1 + train_solver._get_epoch()

    print('Start training with this config:')
    print(cfg)
    print(f"Encoder mode: {args.encoder_mode}")
    print(f"Pretrained MAE: {args.pretrained}")
    print(f"Warmup epochs: {cfg.train.warmup} (train only on source)")
    print(f"Total epochs: {cfg.train.niter + cfg.train.niter_decay}")

    # Training loop (following train.py line 102-166)
    for epoch in range(start_epoch, cfg.train.niter + cfg.train.niter_decay + 1):
        epoch_start_time = time.time()
        print_start_time = time.time()

        # Determine training phase
        phase = "Warmup (Source Only)" if epoch <= cfg.train.warmup else "MPL (Source + Pseudo Target)"
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{cfg.train.niter + cfg.train.niter_decay} - Phase: {phase}")
        print(f"{'='*60}")

        # Training (following train.py line 107-120)
        # First to initialize the internal log of loss
        train_solver._init_epoch()

        for i, data in enumerate(train_loader):
            train_solver.train_step(data, epoch)

            if i % cfg.train.print_freq == 0:
                train_solver.print_cur_loss(epoch, i, print_start_time)
                print_start_time = time.time()

        # Summarize this epoch's results (following train.py line 120)
        train_solver._log_internal_epoch_res(len(train_loader))

        if cfg.system.wandb:
            wandb.log(
                {k+'_epoch': v for k, v in train_solver._get_internal_loss().items()})

        # Validation (following train.py line 126-143)
        if run_val:
            save_best = train_solver.validation(epoch)
            if cfg.system.wandb:
                wandb.log({'validation dice': train_solver.val_dice[-1],
                           'validation score': train_solver.val_score[-1]})
            print(
                f"Epoch: {epoch}, Validation Dice: {train_solver.val_dice[-1]}, Validation score: {train_solver.val_score[-1]}, target pseudo loss: {train_solver.tgt_pse_seg_loss[-1]}")

            if save_best:
                torch.save(train_solver.model.state_dict(),
                           os.path.join(ckpt_fld, 'best_model.pth'))

            print('Current cumulative epochs of no improvement: ' +
                  str(train_solver.cumulative_no_improve[-1]))
            if train_solver.cumulative_no_improve[-1] > cfg.train.patience:
                print('Early stopping')
                break

        # Get the visualization (following train.py line 146)
        train_solver.save_visualization(epoch)

        # Save the model (following train.py line 148-150)
        if epoch % cfg.train.save_epoch_freq == 0:
            torch.save(train_solver.model.state_dict(),
                       os.path.join(ckpt_fld, f'model_epoch_{epoch}.pth'))

        # Save the latest solver status (following train.py line 152-154)
        checkpoint_dict = {
            'model_state_dict': train_solver.model.state_dict(),
            'optimizer_state_dict': train_solver.optimizer.state_dict(),
            'scheduler_state_dict': train_solver.scheduler.state_dict(),
            'is_teacher_init': train_solver.is_teacher_init,
            'src_seg_loss': train_solver.src_seg_loss,
            'src_seg_loss_masked': train_solver.src_seg_loss_masked,
            'tgt_pse_seg_loss': train_solver.tgt_pse_seg_loss,
            'val_dice': train_solver.val_dice,
            'val_score': train_solver.val_score,
            'cumulative_no_improve': train_solver.cumulative_no_improve,
        }
        torch.save(checkpoint_dict, os.path.join(ckpt_fld, 'solver_latest.pth'))

        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, cfg.train.niter + cfg.train.niter_decay, time.time() - epoch_start_time))

        # Scheduler step (following train.py line 160-163)
        if epoch > cfg.train.niter:
            train_solver.scheduler_step()

        if cfg.system.wandb:
            wandb.log({'lr': train_solver.optimizer.param_groups[0]['lr']})
            wandb.log({'epoch': epoch})

    # Save final model (following train.py line 167-168)
    torch.save(train_solver.model.state_dict(),
               os.path.join(ckpt_fld, 'model_final.pth'))

    print(f"\n{'='*60}")
    print(f"MAPSeg training completed!")
    print(f"Best Dice Score: {max(train_solver.val_dice) if train_solver.val_dice else 0.0:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
