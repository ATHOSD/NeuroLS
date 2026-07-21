'''
Configuration file for Lifespan Foundation Model
'''
from yacs.config import CfgNode as CN

# create the config according to ./options/base_options.py
_c = CN()

_c.system = CN()
_c.system.project = 'LifespanFoundation'
_c.system.exp_name = 'lifespan_moe_mae'
# wheter to use wandb
_c.system.wandb = True
# Number of threads for data loading
_c.system.n_threads = 4
# random seed
_c.system.seed = 0
# whether to save nifti during training
_c.system.save_nii = True
# display image size
_c.system.display_winsize = 128
# save image output into HTML
_c.system.no_html = False
# checkpoint directory
_c.system.ckpt_dir = './checkpoints'

_c.train = CN()
# training type: 'mae', 'segmentation', 'age', 'sex', 'synthesis'
_c.train.type = 'mae'
# learning rate
_c.train.lr = 1e-4
# batch size
_c.train.batch_size = 4
# number of epochs
_c.train.epochs = 100
# number of epochs with scheduler
_c.train.niter = 0
# number of epochs with decay (for learning rate decay after niter)
_c.train.niter_decay = 0
# warmup epochs (for segmentation, train only on source during warmup)
_c.train.warmup = 0
# test_time training mode
_c.train.test_time = False
# mask ratio for MAE training
_c.train.mask_ratio = 0.75
# class number for segmentation (including background)
_c.train.cls_num = 8
# patch size for MAE patches
_c.train.local_mae_patch = 8
_c.train.global_mae_patch = 4
# default optimizer
_c.train.optimizer = 'AdamW'
# weight decay
_c.train.weight_decay = 0.05
# betas
_c.train.betas = (0.9, 0.95)
# number of steps to print the loss
_c.train.print_freq = 50
# number of epochs to save the current checkpoints
_c.train.save_epoch_freq = 50
# how many of epochs without improvement will stop the training
_c.train.patience = 50
# weight for cross-modality consistency (CMC) KL loss in MPL phase (0 disables CMC)
_c.train.cmc_weight = 0.01

_c.data = CN()
# file extension for the data
_c.data.extension = '.nii.gz'
# whether to normalize the data
_c.data.normalize = True
# normalization percentile for data
_c.data.norm_perc = 99.5
# whether to remove background
_c.data.remove_bg = True
# whether to do data augmentation
_c.data.aug = True
# probability of data augmentation
_c.data.aug_prob = 0.35
# size of 3D patch
_c.data.patch_size = (96, 96, 96)

# For MAE TRAINING
# root for masked autoencoding training
'''
    data structure:
    mae_root
        |--- Site1
            |--- sub-*
                |--- ses-*
                    |--- anat
                        |--- T1w.nii.gz
                        |--- T2w.nii.gz
                        |--- FA.nii.gz
        |--- Site2
             |--- sub-*
                |--- ses-*
                    |--- anat
                        |--- T1w.nii.gz
                        |--- T2w.nii.gz
                        |--- FA.nii.gz
'''
_c.data.mae_root = './data/mae_pretraining'
# optional: domain file listing different datasets
_c.data.mae_domain_file = './cfg/mae_domains.txt'
# domain file for validation (separate set of datasets)
_c.data.mae_val_domain_file = './cfg/mae_val_domains.txt'
# optional: test exclusion list
_c.data.mae_test_list = './cfg/test_exclusions.csv'
# Age group assignment is loaded automatically from each domain's demographics.csv
# (or Demographics/demographics-site*.csv for ABCD).
# Subjects not found in demographics default to 'adult'.

# For SEGMENTATION
# root for segmentation data
'''
    data structure:
    seg_root
        |--- train
            |--- T2w
                |--- subject1.nii.gz
                |--- subject2.nii.gz
            |--- labels
                |--- subject1.nii.gz
                |--- subject2.nii.gz
        |--- val
            |--- T2w
            |--- labels
'''
_c.data.seg_root = './data/segmentation'
_c.data.seg_modality = 'T2w'  # Primary modality for segmentation
# MAPSeg-style data paths
_c.data.val_img = './data/segmentation/val/images'
_c.data.val_label = './data/segmentation/val/labels'
_c.data.val_data = './data/segmentation/val_data'   # Val data root (multi-dataset)
_c.data.src_data = './data/segmentation/src_data'  # Source domain (labeled)
_c.data.tgt_data = './data/segmentation/tgt_data'  # Target domain (unlabeled)
_c.data.src_demographics = ''
_c.data.tgt_demographics = ''
_c.data.val_demographics = ''

# For PREDICTION TASKS (age/sex)
# root for prediction data
'''
    data structure:
    pred_root
        |--- age
            |--- train_metadata.csv  # subject_id, age
            |--- val_metadata.csv
            |--- train
                |--- T1w
                |--- T2w
                |--- FA
            |--- val
                |--- T1w
                |--- T2w
                |--- FA
        |--- sex
            |--- train_metadata.csv  # subject_id, sex (0/1)
            |--- val_metadata.csv
            |--- train
            |--- val
'''
_c.data.pred_root = './data/prediction'

# For SYNTHESIS
# root for synthesis data (following MAE structure)
'''
    data structure:
    synthesis_root
        |--- HCP-A
            |--- Resampled
                |--- sub-*
                    |--- ses-*
                        |--- anat
                            |--- *_T1w.nii.gz
                            |--- *_T2w.nii.gz
                            |--- *_FA.nii.gz
'''
_c.data.synthesis_root = './data/synthesis'
_c.data.synthesis_domain_file = './cfg/synthesis_domains.txt'
_c.data.synthesis_train_csv = './cfg/synthesis_train.csv'
_c.data.synthesis_val_csv = './cfg/synthesis_val.csv'

# For AGE PREDICTION (Stage 1: Classification)
# root for age prediction data (following synthesis/MAE structure)
'''
    data structure:
    age_root
        |--- HCP-A
            |--- Resampled
                |--- sub-*
                    |--- ses-*
                        |--- anat
                            |--- *_T1w.nii.gz
                            |--- *_T2w.nii.gz
                            |--- *_FA.nii.gz
        |--- ABCD
            |--- Resampled
                |--- site-*
                    |--- sub-*
                        |--- ses-*
                            |--- anat
'''
_c.data.age_root = './data/age_prediction'
_c.data.age_domain_file = './cfg/age_predict/age_domains.txt'
_c.data.age_train_csv = './cfg/age_predict/age_train.csv'
_c.data.age_val_csv = './cfg/age_predict/age_val.csv'

_c.model = CN()
# supported modalities (now includes MD)
_c.model.modalities = ['T1w', 'T2w', 'FA', 'MD']
# age groups for duo MoE: fetal (prenatal), neonatal (0-3mo, wk(neonatal)), infant (3mo-2y), child (2-18y), adult (>18y)
_c.model.age_groups = ['fetal', 'neonatal', 'infant', 'child', 'adult', 'elderly']
# MoE architecture type:
#   'hierarchical' - two-layer: 4 modality experts → 4 age experts (8 total, recommended)
#   'flat'         - one expert per (modality × age_group) = 16 total
_c.model.moe_type = 'hierarchical'
# dimension of intermediate embedding
_c.model.embed_dim = 512
# depth of the model (for ResNet encoder)
_c.model.depth = 5
# number of classes for segmentation
_c.model.num_classes = 15
# number of classes for age prediction (lifespan stages)
_c.model.num_age_classes = 6
# whether to load pretrained model
_c.model.load_pretrain = False
# path to pretrained model
_c.model.pretrain_model = './checkpoints/pretrained.pth'
# whether the model was pretrained on large-scale data (affects EMA alpha)
_c.model.large_scale = False


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for lifespan foundation model."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _c.clone()