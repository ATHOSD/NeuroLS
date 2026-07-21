"""
DataLoader factory for Lifespan Foundation Model
"""

from .lifespan_datasets import (
    LifespanMAEDataset,
    LifespanSegmentationDataset,
    LifespanPredictionDataset,
    LifespanSynthesisDataset,
    LifespanAgeDataset
)
import torch


def get_dataloader(cfg, task='mae', split='train', synthesis_type='t1w_to_t2w'):
    """
    Get dataloader for different tasks and splits

    Args:
        cfg: Configuration object
        task: Task type ('mae', 'segmentation', 'age', 'sex', 'synthesis')
        split: Data split ('train', 'val')

    Returns:
        DataLoader for the specified task and split
    """

    if task == 'mae':
        dataset = LifespanMAEDataset(cfg, split=split)
        batch_size = cfg.train.batch_size
        shuffle = (split == 'train')
        num_workers = cfg.system.n_threads if hasattr(cfg.system, 'n_threads') else 8

    elif task == 'segmentation':
        dataset = LifespanSegmentationDataset(cfg, split=split)
        batch_size = cfg.train.batch_size if split == 'train' else 1
        shuffle = (split == 'train')
        num_workers = cfg.system.n_threads if hasattr(cfg.system, 'n_threads') else 4

    elif task in ['age', 'sex']:
        dataset = LifespanPredictionDataset(cfg, task=task, split=split)
        batch_size = cfg.train.batch_size if split == 'train' else 1
        shuffle = (split == 'train')
        num_workers = cfg.system.n_threads if hasattr(cfg.system, 'n_threads') else 4

    elif task == 'synthesis':
        dataset = LifespanSynthesisDataset(cfg, split=split, synthesis_type=synthesis_type)
        batch_size = cfg.train.batch_size if split == 'train' else 1
        shuffle = (split == 'train')
        num_workers = cfg.system.n_threads if hasattr(cfg.system, 'n_threads') else 4

    elif task == 'age_stage1':
        dataset = LifespanAgeDataset(cfg, split=split)
        batch_size = cfg.train.batch_size if split == 'train' else 1
        shuffle = (split == 'train')
        num_workers = cfg.system.n_threads if hasattr(cfg.system, 'n_threads') else 4

    else:
        raise ValueError(f"Unsupported task: {task}. Supported tasks: 'mae', 'segmentation', 'age', 'sex', 'synthesis', 'age_stage1'")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=(split == 'train')  # Drop last batch only for training
    )

    return dataloader