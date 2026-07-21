"""
Datasets for Lifespan Foundation Model Training

Supports both MAE pre-training and downstream task fine-tuning
with flexible modality handling
"""

import torch.utils.data as data
import os
from .data_utils import *
import torchio as tio
import torch
import numpy as np
import glob
import pandas as pd
from .preprocess import preprocess_image


def parse_age_to_years(age_val) -> float:
    """
    Parse the age field from demographics CSVs to decimal years.
    Returns None if unparseable; negative value signals fetal/gestational.

    Formats seen across datasets:
      '37.43 wk'            dHCP-Fetal, FeTA  → fetal (gestational weeks)
      '0.29 wk(neonatal)'   dHCP-Infant       → infant (weeks after birth)
      '1.2 mo'              HBCD              → infant (months)
      '1.08', '33.0', '91'  BCP, ABIDE, ADNI  → years (float / int)
      '', NaN, 'Young'                         → None (unknown)
    """
    if age_val is None:
        return None
    if isinstance(age_val, float) and np.isnan(age_val):
        return None
    s = str(age_val).strip()
    if not s:
        return None

    if 'wk(neonatal)' in s:
        # Weeks after birth (neonatal) → convert to years (stays < 2y → infant)
        try:
            return float(s.split('wk')[0].strip()) / 52.0
        except ValueError:
            return 0.01  # newborn fallback

    if 'wk' in s:
        # Gestational weeks → fetal (return negative as sentinel)
        return -1.0

    if 'mo' in s:
        # Months after birth → years
        try:
            return float(s.split('mo')[0].strip()) / 12.0
        except ValueError:
            return None

    # Handle age group name strings directly
    _group_to_years = {
        'fetal': -1.0, 'neonatal': 0.1, 'infant': 1.0, 'child': 10.0, 'adult': 30.0, 'elderly': 70.0
    }
    if s.lower() in _group_to_years:
        return _group_to_years[s.lower()]

    try:
        return float(s)
    except ValueError:
        return None  # 'Young', free-text, etc.


def age_years_to_group(age_years) -> str:
    """Map decimal years (or None) to an age group string.
    fetal:    < 0     (gestational)
    neonatal: 0 - 3mo (wk(neonatal), < 0.25y)
    infant:   3mo - 2y
    child:    2 - 18y
    adult:    18 - 65y
    elderly:  > 65y
    """
    if age_years is None:
        return 'adult'   # unknown → default to adult
    if age_years < 0:
        return 'fetal'
    elif age_years < 0.25:  # < 3 months
        return 'neonatal'
    elif age_years < 2:
        return 'infant'
    elif age_years < 18:
        return 'child'
    elif age_years < 65:
        return 'adult'
    else:
        return 'elderly'


def get_modality_from_filename(filename: str, supported_modalities: set):
    """
    Extract modality from a NIfTI filename by looking for modality tokens.
    Uses '_MOD' or '_MOD.' patterns to avoid false matches (e.g. 'MD' in a subject ID).
    Returns the matched modality string, or None if no supported modality is found.

    Examples:
      sub-100206_T1w_brain.nii.gz      → 'T1w'
      sub-CC00856XX15_ses-3530_FA.nii.gz → 'FA'
      100206_MD.nii.gz                  → 'MD'
      100206_AD.nii.gz                  → None  (AD not in supported_modalities)
    """
    stem = filename.replace('.nii.gz', '').replace('.nii', '')
    parts = stem.replace('-', '_').split('_')
    for part in parts:
        if part in supported_modalities:
            return part
    return None


def load_domain_age_lookup(mae_root: str, domain: str) -> dict:
    """
    Build {(sub_id, ses_id): age_group} from a domain's demographics CSV(s).

    Special case:
      ABCD uses Demographics/demographics-site*.csv (multiple per-site files).
      All other domains use demographics.csv in the domain root.
    """
    lookup = {}

    if domain == 'ABCD':
        csv_files = sorted(glob.glob(
            os.path.join(mae_root, domain, 'Demographics', 'demographics-site*.csv')))
    else:
        demo_path = os.path.join(mae_root, domain, 'demographics.csv')
        csv_files = [demo_path] if os.path.exists(demo_path) else []

    for csv_path in csv_files:
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if not {'sub_id', 'ses_id', 'age'}.issubset(df.columns):
            continue
        for _, row in df.iterrows():
            age_years = parse_age_to_years(row['age'])
            if age_years is None:
                continue  # skip subjects with missing age
            key = (str(row['sub_id']), str(row['ses_id']))
            lookup[key] = age_years_to_group(age_years)

    return lookup


class LifespanMAEDataset(data.Dataset):
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split

        domain_file = cfg.data.mae_val_domain_file if split == 'val' else cfg.data.mae_domain_file
        with open(domain_file) as f:
            self.domains = f.read().splitlines()

        # Load (sub_id, ses_id) → age_group from each domain's demographics CSV
        self.age_lookup = {}
        for domain in self.domains:
            domain_lookup = load_domain_age_lookup(cfg.data.mae_root, domain)
            self.age_lookup.update(domain_lookup)
            print(f"  [age] {domain}: {len(domain_lookup)} subjects loaded")

        self.path_dic = {}
        self.subject_count = {}
        self.image_count = {}

        total_subjects = set()
        total_images = 0

        exclude_pairs = set()
        if os.path.exists(cfg.data.mae_test_list):
            df_exclude = pd.read_csv(cfg.data.mae_test_list)
            for _, row in df_exclude.iterrows():
                exclude_pairs.add((row["sub_id"], row["ses_id"]))

        supported_modalities = set(cfg.model.modalities)

        for idx, domain in enumerate(self.domains):
            domain_path = os.path.join(cfg.data.mae_root, domain)
            nii_paths = []

            # Collect .nii.gz files from Skull-stripped folder
            if domain in ["ABCD", "ADHD", "ABIDE"]:
                search_dirs = [
                    os.path.join(domain_path, "Skull-stripped", "*", "sub-*", "ses-*", "anat", "*.nii.gz"),
                ]
            else:
                search_dirs = [
                    os.path.join(domain_path, "Skull-stripped", "sub-*", "ses-*", "anat", "*.nii.gz"),
                ]

            # Gather files — keep only those with a supported modality, skip masks
            for pattern in search_dirs:
                for f in glob.glob(pattern):
                    fname = os.path.basename(f)
                    # Skip mask files
                    if 'mask' in fname.lower():
                        continue
                    # Skip files whose modality is not supported (e.g. AD, RD)
                    if get_modality_from_filename(fname, supported_modalities) is None:
                        continue
                    parts = f.split(os.sep)
                    sub_id = next((p for p in parts if p.startswith("sub-")), None)
                    ses_id = next((p for p in parts if p.startswith("ses-")), None)
                    if (sub_id, ses_id) in exclude_pairs:
                        continue
                    nii_paths.append(f)

            # Store image paths
            self.path_dic[str(idx)] = sorted(nii_paths)
            self.image_count[domain] = len(nii_paths)
            total_images += len(nii_paths)
            if len(nii_paths) == 0:
                print(f"  WARNING: {domain} has no files — will be skipped during sampling")

            # Track subjects
            domain_subjects = set()
            for path in nii_paths:
                parts = path.split(os.sep)
                sub_id = next((p for p in parts if p.startswith("sub-")), None)
                if sub_id:
                    domain_subjects.add(sub_id)
                    total_subjects.add(sub_id)
            self.subject_count[domain] = len(domain_subjects)

            # Domain summary
            print(f"📁 Domain: {domain}")
            print(f"  🧠 Subjects: {self.subject_count[domain]}")
            print(f"  📄 Images : {self.image_count[domain]}")

        print("====================================")
        print(f"✅ Total domains  : {len(self.domains)}")
        print(f"👤 Total subjects : {len(total_subjects)}")
        print(f"🖼️  Total images   : {total_images}")
        print("====================================")

        self.num_domain = len(self.domains)
        self.all_img = total_images
        self.all_subjects = len(total_subjects)

    
    def _get_age_group(self, path: str):
        """Look up age group from pre-loaded demographics. Returns None if missing."""
        parts = path.split(os.sep)
        sub_id = next((p for p in parts if p.startswith('sub-')), '')
        ses_id = next((p for p in parts if p.startswith('ses-')), '')
        return self.age_lookup.get((sub_id, ses_id), None)

    def __getitem__(self, index):
        # Retry until we find a sample with a valid age in demographics
        # Skip domains with no files (empty path list)
        non_empty_domains = [i for i in range(self.num_domain)
                             if len(self.path_dic[str(i)]) > 0]
        if not non_empty_domains:
            raise RuntimeError("All domains have empty file lists.")

        tmp_scans = None
        for _ in range(50):
            idx = non_empty_domains[np.random.randint(0, len(non_empty_domains))]
            tmp_path = self.path_dic[str(idx)]
            index = np.random.randint(0, len(tmp_path))
            age_group = self._get_age_group(tmp_path[index])
            if age_group is None:
                continue

            # Extract modality from filename
            filename = os.path.basename(tmp_path[index])
            modality = get_modality_from_filename(filename, set(self.cfg.model.modalities))

            # Randomly resample to 1mm isotropic or keep native spacing
            do_resample = np.random.uniform() <= self.cfg.data.aug_prob
            try:
                tmp_scans = preprocess_image(tmp_path[index],
                                             target_spacing=(1.0, 1.0, 1.0),
                                             resample=do_resample)
            except Exception as e:
                print(f"[WARN] Skipping corrupted file {tmp_path[index]}: {e}")
                continue
            break
        if tmp_scans is None:
            raise RuntimeError("Could not find a valid sample after 50 retries.")
        tmp_scans[tmp_scans < 0] = 0
        tmp_scans = random_flip(tmp_scans)

        # normalization
        # we do a little trick to normalize the scan at random percentile
        if self.cfg.data.normalize:
            if np.random.uniform() <= self.cfg.data.aug_prob:
                perc_dif = 100-self.cfg.data.norm_perc
                tmp_scans = norm_img(tmp_scans, np.random.uniform(
                    self.cfg.data.norm_perc-perc_dif, 100))
            else:
                tmp_scans = norm_img(tmp_scans, self.cfg.data.norm_perc)
            if tmp_scans is None:
                raise ValueError(f"[SKIP] Normalization failed for: {tmp_path[index]}")

        # whether to pad the image to match the patch size
        # and then cast to torch.tensor
        x, y, z = self.cfg.data.patch_size

        if min(tmp_scans.shape) < min(x, y, z):
            x_diff = x-tmp_scans.shape[0]
            y_diff = y-tmp_scans.shape[1]
            z_diff = z-tmp_scans.shape[2]
            tmp_scans = np.pad(tmp_scans, ((max(0, int(x_diff/2)), max(0, x_diff-int(x_diff/2))), (max(0, int(
                y_diff/2)), max(0, y_diff-int(y_diff/2))), (max(0, int(z_diff/2)), max(0, z_diff-int(z_diff/2)))), constant_values=1e-4)  # cant pad with 0s, otherwise the local and global patches wont be the same location
            tmp_scans = torch.unsqueeze(torch.from_numpy(tmp_scans), 0)
        else:
            tmp_scans = torch.unsqueeze(torch.from_numpy(tmp_scans), 0)
        _, x1, y1, z1 = tmp_scans.shape
        transforms = tio.RandomAffine(p=self.cfg.data.aug_prob, scales=(0.75, 1.5), degrees=40,
                                      isotropic=True,
                                      default_pad_value=0, image_interpolation='linear')
        tmp_scans = tio.ScalarImage(tensor=tmp_scans)
        tmp_scans = transforms(tmp_scans)

        # if remove_bg, the patch will only be sampled from the foreground (non-zero) region

        if self.cfg.data.remove_bg:
            bound = get_bounds(tmp_scans.data)
            if bound[1] - x > bound[0]:
                x_idx = np.random.randint(bound[0], bound[1] - x)
            else:
                if bound[1] - x >= 0:
                    x_idx = bound[1] - x
                else:
                    x_idx = int((x1-x)/2)
            if bound[3] - y > bound[2]:
                y_idx = np.random.randint(bound[2], bound[3] - y)
            else:
                if bound[3] - y >= 0:
                    y_idx = bound[3] - y
                else:
                    y_idx = int((y1-y)/2)

            if bound[5] - z > bound[4]:
                z_idx = np.random.randint(bound[4], bound[5] - z)
            else:
                if bound[5] - z >= 0:
                    z_idx = bound[5] - z
                else:
                    z_idx = int((z1-z)/2)
        else:
            # if not remove_bg, the patch will be sampled from the whole image
            bound = [0, x1, 0, y1, 0, z1]
            if x1 - x == 0:
                x_idx = 0
            else:
                x_idx = np.random.randint(0, x1 - x)
            if y1 - y == 0:
                y_idx = 0
            else:
                y_idx = np.random.randint(0, y1 - y)
            if z1 - z == 0:
                z_idx = 0
            else:
                z_idx = np.random.randint(0, z1 - z)
        # location indicates the sampled patch location
        location = torch.zeros_like(tmp_scans.data)
        location[:, x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z] = 1

        # When brain bounding box is smaller than patch in a dimension, use the
        # local patch extent for the global crop so spatial correspondence is correct
        gx0 = x_idx if (bound[1] - bound[0] < x) else bound[0]
        gx1 = x_idx + x if (bound[1] - bound[0] < x) else bound[1]
        gy0 = y_idx if (bound[3] - bound[2] < y) else bound[2]
        gy1 = y_idx + y if (bound[3] - bound[2] < y) else bound[3]
        gz0 = z_idx if (bound[5] - bound[4] < z) else bound[4]
        gz1 = z_idx + z if (bound[5] - bound[4] < z) else bound[5]

        sbj = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans.data[:, gx0:gx1, gy0:gy1, gz0:gz1]))
        transforms = tio.transforms.Resize(target_shape=(x, y, z))
        sbj = transforms(sbj)
        down_scan = sbj['one_image'].data

        input_dict = {'local_patch': tmp_scans.data[:, x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z],
                      'global_images': down_scan,
                      'modality': modality,
                      'age_group': age_group}

        return input_dict

    def __len__(self):
        # we used fixed 2000 as the number of samples in each epoch
        # one can choose max(2000, self.all_img)
        # return max(2000, self.all_img)
        return 2000




class LifespanPredictionDataset(data.Dataset):
    """
    Dataset for age/sex prediction tasks
    Can use any available modality
    """

    def __init__(self, cfg, task='age', split='train'):
        self.cfg = cfg
        self.task = task  # 'age' or 'sex'
        self.split = split

        # Load metadata
        metadata_file = os.path.join(cfg.data.pred_root, task, f'{split}_metadata.csv')
        self.metadata = pd.read_csv(metadata_file)

        # Get available modalities for each subject
        self.samples = []
        task_root = os.path.join(cfg.data.pred_root, task, split)

        for _, row in self.metadata.iterrows():
            subject_id = row['subject_id']
            sample = {'subject_id': subject_id}

            # Add target
            if task == 'age':
                sample['target'] = float(row['age'])
            elif task == 'sex':
                sample['target'] = int(row['sex'])  # 0: female, 1: male

            # Find available modalities for this subject
            for modality in cfg.model.modalities:
                img_path = os.path.join(task_root, modality, f"{subject_id}.nii.gz")
                if os.path.exists(img_path):
                    sample[modality] = img_path

            # Only include if at least one modality is available
            available_modalities = [mod for mod in cfg.model.modalities if mod in sample]
            if len(available_modalities) > 0:
                sample['available_modalities'] = available_modalities
                self.samples.append(sample)

        print(f"{task.capitalize()} {split} dataset: {len(self.samples)} samples")

    def __getitem__(self, index):
        sample = self.samples[index]

        # Randomly select one available modality
        available_mods = sample['available_modalities']
        selected_modality = np.random.choice(available_mods)

        img_path = sample[selected_modality]

        try:
            # Load and preprocess image
            img_data = preprocess_image(img_path, target_spacing=(1.0, 1.0, 1.0))
            img_data[img_data < 0] = 0

            # Normalization
            if self.cfg.data.normalize:
                img_data = norm_img(img_data, self.cfg.data.norm_perc)
                if img_data is None:
                    raise ValueError(f"Normalization failed for: {img_path}")

            # Pad to match patch size if needed
            x, y, z = self.cfg.data.patch_size
            if min(img_data.shape) < min(x, y, z):
                x_diff = x - img_data.shape[0]
                y_diff = y - img_data.shape[1]
                z_diff = z - img_data.shape[2]

                img_data = np.pad(img_data, (
                    (max(0, int(x_diff/2)), max(0, x_diff - int(x_diff/2))),
                    (max(0, int(y_diff/2)), max(0, y_diff - int(y_diff/2))),
                    (max(0, int(z_diff/2)), max(0, z_diff - int(z_diff/2)))
                ), constant_values=1e-4)

            img_data = torch.unsqueeze(torch.from_numpy(img_data.astype(np.float32)), 0)

            return {
                'image': img_data,
                'target': sample['target'],
                'modality': selected_modality,
                'subject_id': sample['subject_id']
            }

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return self.__getitem__((index + 1) % len(self.samples))

    def __len__(self):
        return len(self.samples)


class LifespanSynthesisDataset(data.Dataset):
    """
    Dataset for synthesis tasks (following MAE structure):
    - T1w -> T2w
    - T2w -> T1w
    - T1w+T2w -> FA

    Uses domain file and CSV list like MAE dataset
    """

    def __init__(self, cfg, split='train', synthesis_type='t1w_to_t2w'):
        self.cfg = cfg
        self.split = split
        self.synthesis_type = synthesis_type

        # Load domains from file
        with open(cfg.data.synthesis_domain_file) as f:
            self.domains = f.read().splitlines()

        # Load subject list from CSV and create inclusion set
        csv_file = cfg.data.synthesis_train_csv if split == 'train' else cfg.data.synthesis_val_csv
        df = pd.read_csv(csv_file)

        # Create set of (sub_id, ses_id) pairs to include
        include_pairs = set()
        for _, row in df.iterrows():
            include_pairs.add((row['sub_id'], row['ses_id']))

        self.path_dic = {}
        self.subject_count = {}
        self.image_count = {}

        total_subjects = set()
        total_images = 0

        for idx, domain in enumerate(self.domains):
            domain_path = os.path.join(cfg.data.synthesis_root, domain)
            nii_paths = []

            # Collect .nii.gz files from Resampled folders (same as MAE)
            search_dirs = []
            if domain in ["ABCD", "ADHD", "ABIDE"]:
                search_dirs += [
                    os.path.join(domain_path, "registered", "*", "sub-*", "ses-*", "anat", "*registered.nii.gz"),
                ]
            else:
                search_dirs += [
                    os.path.join(domain_path, "registered", "sub-*", "ses-*", "anat", "*registered.nii.gz"),
                ]

            # Gather files and organize by subject-session
            subject_modalities = {}  # {(sub_id, ses_id): {'T1w': path, 'T2w': path, 'FA': path}}

            for pattern in search_dirs:
                matches = glob.glob(pattern)
                for f in matches:
                    parts = f.split(os.sep)
                    sub_id = next((p for p in parts if p.startswith("sub-")), None)
                    ses_id = next((p for p in parts if p.startswith("ses-")), None)

                    # Only include if (sub_id, ses_id) is in CSV
                    if (sub_id, ses_id) in include_pairs:
                        key = (sub_id, ses_id)
                        if key not in subject_modalities:
                            subject_modalities[key] = {}

                        # Determine modality from filename
                        filename = os.path.basename(f)
                        for mod in ['T1w', 'T2w', 'FA']:
                            if mod in filename:
                                subject_modalities[key][mod] = f
                                nii_paths.append(f)
                                break

            # Store organized subject-session data
            self.path_dic[str(idx)] = subject_modalities
            self.image_count[domain] = len(nii_paths)
            total_images += len(nii_paths)

            # Track subjects
            domain_subjects = set()
            for (sub_id, ses_id) in subject_modalities.keys():
                domain_subjects.add(sub_id)
                total_subjects.add(sub_id)
            self.subject_count[domain] = len(domain_subjects)

            # Domain summary
            print(f"📁 Domain: {domain}")
            print(f"  🧠 Subjects: {self.subject_count[domain]}")
            print(f"  📄 Images : {self.image_count[domain]}")

        print("====================================")
        print(f"✅ Total domains  : {len(self.domains)}")
        print(f"👤 Total subjects : {len(total_subjects)}")
        print(f"🖼️  Total images   : {total_images}")
        print("====================================")

        self.num_domain = len(self.domains)
        self.all_img = total_images
        self.all_subjects = len(total_subjects)

    def __getitem__(self, index):
        """
        Returns a dictionary with source (image) and target (label) based on synthesis type:
        - t1w_to_t2w: image=T1w, label=T2w
        - t2w_to_t1w: image=T2w, label=T1w
        - t1t2_to_fa: image=[T1w, T2w], label=FA
        """
        # Random domain selection (like MAE)
        idx = int(np.random.random_sample() // (1 / self.num_domain))
        subject_dict = self.path_dic[str(idx)]

        # Random subject-session selection from domain
        if len(subject_dict) == 0:
            # Fallback if domain is empty
            idx = (idx + 1) % self.num_domain
            subject_dict = self.path_dic[str(idx)]

        subject_keys = list(subject_dict.keys())
        random_key = subject_keys[np.random.randint(0, len(subject_keys))]
        modality_paths = subject_dict[random_key]

        x, y, z = self.cfg.data.patch_size

        try:
            # Load all required modalities based on synthesis type
            loaded_data = {}

            if self.synthesis_type == 't1w_to_t2w':
                required_mods = ['T1w', 'T2w']
            elif self.synthesis_type == 't2w_to_t1w':
                required_mods = ['T2w', 'T1w']
            elif self.synthesis_type == 't1t2_to_fa':
                required_mods = ['T1w', 'T2w', 'FA']
            else:
                raise ValueError(f"Unknown synthesis type: {self.synthesis_type}")

            # Check if all required modalities are available
            for mod in required_mods:
                if mod not in modality_paths:
                    # Skip this sample if missing required modality
                    return self.__getitem__((index + 1) % self.__len__())

            # Load and preprocess each required modality
            for modality in required_mods:
                path = modality_paths[modality]

                # Load and preprocess
                tmp_scans = preprocess_image(path, target_spacing=(1.0, 1.0, 1.0))
                tmp_scans[tmp_scans < 0] = 0

                # Normalization
                if self.cfg.data.normalize:
                    if np.random.uniform() <= self.cfg.data.aug_prob:
                        perc_dif = 100-self.cfg.data.norm_perc
                        tmp_scans = norm_img(tmp_scans, np.random.uniform(
                            self.cfg.data.norm_perc-perc_dif, 100))
                    else:
                        tmp_scans = norm_img(tmp_scans, self.cfg.data.norm_perc)
                    if tmp_scans is None:
                        raise ValueError(f"[SKIP] Normalization failed for: {path}")

                loaded_data[modality] = tmp_scans

            # Ensure all loaded images have same shape (for paired synthesis)
            shapes = [data.shape for data in loaded_data.values()]
            if len(set(shapes)) > 1:
                print(f"Warning: Shape mismatch for {random_key}: {shapes}")
                return self.__getitem__((index + 1) % self.__len__())

            # Pad to match patch size
            first_data = list(loaded_data.values())[0]
            for i in range(3):
                if first_data.shape[i] < self.cfg.data.patch_size[i]:
                    pad_needed = self.cfg.data.patch_size[i] - first_data.shape[i]
                    pad_before = pad_needed // 2
                    pad_after = pad_needed - pad_before

                    for modality in loaded_data:
                        if i == 0:
                            pad_width = ((pad_before, pad_after), (0, 0), (0, 0))
                        elif i == 1:
                            pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
                        else:
                            pad_width = ((0, 0), (0, 0), (pad_before, pad_after))

                        loaded_data[modality] = np.pad(loaded_data[modality], pad_width,
                                                       mode='constant', constant_values=0)

            # Random patch extraction (reject if too many zeros)
            first_data = list(loaded_data.values())[0]
            x1, y1, z1 = first_data.shape

            max_attempts = 50
            valid_patch = False

            for attempt in range(max_attempts):
                x_idx = np.random.randint(0, x1 - x) if x1 > x else 0
                y_idx = np.random.randint(0, y1 - y) if y1 > y else 0
                z_idx = np.random.randint(0, z1 - z) if z1 > z else 0

                # Check if patch has enough non-zero content
                if self.cfg.data.remove_bg:
                    patches_valid = True
                    for modality, data in loaded_data.items():
                        patch = data[x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z]
                        zero_ratio = (patch == 0).sum() / patch.size
                        if zero_ratio >= 0.8:
                            patches_valid = False
                            break

                    if patches_valid:
                        valid_patch = True
                        break
                else:
                    valid_patch = True
                    break

            # Extract patches for all modalities
            patches = {}
            for modality, data in loaded_data.items():
                patch = data[x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z]
                patches[modality] = torch.unsqueeze(torch.from_numpy(patch.astype(np.float32)), 0)

            # Organize as image (source) and label (target) based on synthesis type
            if self.synthesis_type == 't1w_to_t2w':
                return {
                    'T1w': patches['T1w'],    # Source: T1w
                    'T2w': patches['T2w'],    # Target: T2w
                    'path': modality_paths['T1w']
                }
            elif self.synthesis_type == 't2w_to_t1w':
                return {
                    'T1w': patches['T1w'],    # Source: T2w
                    'T2w': patches['T2w'],    # Target: T1w
                    'path': modality_paths['T2w']
                }
            elif self.synthesis_type == 't1t2_to_fa':
                return {
                    'T1w': patches['T1w'],    # Source 1: T1w (will handle T2w separately)
                    'T2w': patches['T2w'],    # Source 2: T2w
                    'FA': patches['FA'],      # Target: FA
                    'path': modality_paths['T1w']
                }

        except Exception as e:
            print(f"Error processing {random_key}: {e}")
            return self.__getitem__((index + 1) % self.__len__())

    def __len__(self):
        # Fixed epoch size like MAE
        return 2000
    

class LifespanSegmentationDataset(data.Dataset):
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split  # Keep for compatibility, but MAPSeg loads source+target together
        self._supported_modalities = set(cfg.model.modalities)

        # Load demographics for age lookup: (sub_id, ses_id) -> age_group
        # Also build dataset-level fallback: dataset_name -> most common age_group
        self._age_lookup = {}
        self._dataset_age_lookup = {}
        for demo_path in [
            getattr(cfg.data, 'src_demographics', None),
            getattr(cfg.data, 'tgt_demographics', None),
        ]:
            if demo_path and os.path.isfile(demo_path):
                df = pd.read_csv(demo_path)
                for _, row in df.iterrows():
                    sub = str(row.get('sub_id', '')).strip()
                    ses_raw = row.get('ses_id', '')
                    ses = '' if (ses_raw is None or (isinstance(ses_raw, float) and np.isnan(ses_raw))) else str(ses_raw).strip()
                    age_years = parse_age_to_years(row.get('age'))
                    grp = age_years_to_group(age_years)
                    self._age_lookup[(sub, ses)] = grp
                    # dataset-level: use first seen age group per dataset
                    dataset = str(row.get('dataset', '')).strip()
                    if dataset and dataset not in self._dataset_age_lookup:
                        self._dataset_age_lookup[dataset] = grp

        # get all image paths (source and target)
        # folder should end with '_train'

        # data from target domain, only img (folder name should end with '_train')

        tgt_dir, src_dir1 = list_finetune_domains(
            cfg.data.tgt_data, cfg.data.src_data)

        self.path_dic = {}
        for i in range(len(tgt_dir)):
            self.path_dic[str(i)] = sorted(
                list_scans(tgt_dir[i], self.cfg.data.extension))
        self.num_domain = len(tgt_dir)

        # data from source domain,  img + label (folder name should end with '_img' for img and '_label' for label)

        self.path_dic_B1 = {}
        self.path_dic_B2 = {}
        for i in range(len(src_dir1)):
            self.path_dic_B1[str(i)] = sorted(
                list_scans(src_dir1[i], self.cfg.data.extension))
            self.path_dic_B2[str(i)] = [
                fpath.replace('_img/', '_label/').replace(
                self.cfg.data.extension, f"_tissue{self.cfg.data.extension}")
                for fpath in self.path_dic_B1[str(i)]
                ]

        self.num_domain_B = len(src_dir1)

        # Build cross-modality index for CMC loss: {stem -> {modality: path}}
        # stem = filename with the modality token and extension removed
        # Subjects with 2+ modalities can serve as CMC pairs (any modality combo)
        self._subject_multimodal_index = {}
        for domain_paths in self.path_dic.values():
            for path in domain_paths:
                fname = os.path.basename(path)
                mod = get_modality_from_filename(fname, self._supported_modalities)
                if mod is None:
                    continue
                stem = fname.replace(f'_{mod}{self.cfg.data.extension}', '')
                if stem not in self._subject_multimodal_index:
                    self._subject_multimodal_index[stem] = {}
                self._subject_multimodal_index[stem][mod] = path

        n_paired = sum(1 for v in self._subject_multimodal_index.values() if len(v) > 1)
        print(f'CMC index: {len(self._subject_multimodal_index)} subjects, {n_paired} with 2+ modalities')

        print('num of target domian: ' + str(self.num_domain))
        print('num of source domain: ' + str(self.num_domain_B))

    def _find_cmc_pair(self, path, modality):
        """Return (paired_path, paired_modality) for CMC loss, or (None, None).

        Looks up the subject stem in the multi-modal index and randomly picks
        one of the other available modalities. Works for any pair (FA/MD, T1w/T2w, etc.)
        as long as the subject has 2+ modalities in the target split.
        """
        fname = os.path.basename(path)
        stem = fname.replace(f'_{modality}{self.cfg.data.extension}', '')
        others = {m: p for m, p in self._subject_multimodal_index.get(stem, {}).items()
                  if m != modality}
        if not others:
            return None, None
        paired_mod = others[np.random.choice(list(others.keys()))]  # random other modality path
        paired_mod_name = [m for m, p in others.items() if p == paired_mod][0]
        return paired_mod, paired_mod_name

    def __getitem__(self, index):
        idx = int(np.random.random_sample() // (1 / self.num_domain))
        tmp_path = self.path_dic[str(idx)]
        indexA = np.random.randint(0, len(tmp_path))

        idx = int(np.random.random_sample() // (1 / self.num_domain_B))
        tmp_path_B1 = self.path_dic_B1[str(idx)]
        tmp_path_B2 = self.path_dic_B2[str(idx)]

        indexB = np.random.randint(0, len(tmp_path_B1))
        x, y, z = self.cfg.data.patch_size

        # Extract modality_A early so we can look up the CMC pair before loading
        fname_A = os.path.basename(tmp_path[indexA])
        modality_A = get_modality_from_filename(fname_A, self._supported_modalities) or 'T1w'
        cmc_pair_path, modality_A2 = self._find_cmc_pair(tmp_path[indexA], modality_A)

        '''
        getitem for training/validation
        '''

        '''
        load non-labeled data
        '''
        tmp_scansA = preprocess_image(tmp_path[indexA],target_spacing=(1.0, 1.0, 1.0))
        tmp_scansA = np.squeeze(tmp_scansA)

        tmp_scansA[tmp_scansA < 0] = 0

        # normalization — save the percentile so the CMC pair uses the same value
        norm_perc_A = self.cfg.data.norm_perc
        if self.cfg.data.normalize:
            if np.random.uniform() <= self.cfg.data.aug_prob:
                perc_dif = 100 - self.cfg.data.norm_perc
                norm_perc_A = np.random.uniform(self.cfg.data.norm_perc - perc_dif, 100)
            tmp_scansA = norm_img(tmp_scansA, norm_perc_A)
        # padding
        if min(tmp_scansA.shape) < min(x, y, z):
            x_diff = 96-tmp_scansA.shape[0]
            y_diff = 96-tmp_scansA.shape[1]
            z_diff = 96-tmp_scansA.shape[2]
            tmp_scansA = np.pad(tmp_scansA, ((max(0, int(x_diff/2)), max(0, x_diff-int(x_diff/2))), (max(0, int(
                y_diff/2)), max(0, y_diff-int(y_diff/2))), (max(0, int(z_diff/2)), max(0, z_diff-int(z_diff/2)))))
            tmp_scansA = torch.unsqueeze(torch.from_numpy(tmp_scansA), 0)
        else:
            tmp_scansA = torch.unsqueeze(torch.from_numpy(tmp_scansA), 0)

        # Load CMC pair with identical preprocessing before augmentation
        tmp_scansA_pair = None
        if cmc_pair_path is not None:
            try:
                _pair = preprocess_image(cmc_pair_path, target_spacing=(1.0, 1.0, 1.0))
                _pair = np.squeeze(_pair)
                _pair[_pair < 0] = 0
                if self.cfg.data.normalize:
                    _pair = norm_img(_pair, norm_perc_A)
                if _pair is not None:
                    if min(_pair.shape) < min(x, y, z):
                        _xd = x - _pair.shape[0]
                        _yd = y - _pair.shape[1]
                        _zd = z - _pair.shape[2]
                        _pair = np.pad(_pair, (
                            (max(0, int(_xd/2)), max(0, _xd - int(_xd/2))),
                            (max(0, int(_yd/2)), max(0, _yd - int(_yd/2))),
                            (max(0, int(_zd/2)), max(0, _zd - int(_zd/2)))), constant_values=1e-4)
                    tmp_scansA_pair = torch.unsqueeze(
                        torch.from_numpy(_pair.astype(np.float32)), 0)
            except Exception as _e:
                print(f'[CMC] Failed to load pair {cmc_pair_path}: {_e}')
                cmc_pair_path = None

        # augmentation — apply the same random transform to imgA and its CMC pair
        _, x1, y1, z1 = tmp_scansA.shape
        # Ensure pair has the same spatial shape as the main image (different datasets
        # may produce off-by-one sizes after resampling to 1 mm isotropic).
        if tmp_scansA_pair is not None and tmp_scansA_pair.shape != tmp_scansA.shape:
            import torch.nn.functional as _F
            tmp_scansA_pair = _F.interpolate(
                tmp_scansA_pair.unsqueeze(0).float(),
                size=tmp_scansA.shape[1:],
                mode='trilinear', align_corners=True).squeeze(0)
        if self.cfg.data.aug:
            transforms = tio.Compose([tio.RandomAffine(p=self.cfg.data.aug_prob, scales=(0.7, 1.3), degrees=30,
                                                       isotropic=True,
                                                       default_pad_value=0, image_interpolation='linear',
                                                       label_interpolation='nearest')

                                      ])
            if tmp_scansA_pair is not None:
                _tio_sbj = tio.Subject(main=tio.ScalarImage(tensor=tmp_scansA),
                                       pair=tio.ScalarImage(tensor=tmp_scansA_pair))
                _tio_sbj = transforms(_tio_sbj)
                tmp_scans = _tio_sbj['main']
                tmp_scans_pair = _tio_sbj['pair']
            else:
                tmp_scans = tio.ScalarImage(tensor=tmp_scansA)
                tmp_scans = transforms(tmp_scans)
                tmp_scans_pair = None
        else:
            tmp_scans = tio.ScalarImage(tensor=tmp_scansA)
            tmp_scans_pair = tio.ScalarImage(tensor=tmp_scansA_pair) if tmp_scansA_pair is not None else None
        # randomly select patch
        if self.cfg.data.remove_bg:
            bound = get_bounds(tmp_scans.data)
            if bound[1] - x > bound[0]:
                x_idx = np.random.randint(bound[0], bound[1] - x)
            else:
                if bound[1] - x >= 0:
                    x_idx = bound[1] - x
                else:
                    if bound[0] + x < x1:
                        x_idx = bound[0]
                    else:
                        x_idx = int((x1 - x) / 2)
            if bound[3] - y > bound[2]:
                y_idx = np.random.randint(bound[2], bound[3] - y)
            else:
                if bound[3] - y >= 0:
                    y_idx = bound[3] - y
                else:
                    if bound[2] + y < y1:
                        y_idx = bound[2]
                    else:
                        y_idx = int((y1 - y) / 2)

            if bound[5] - z > bound[4]:
                z_idx = np.random.randint(bound[4], bound[5] - z)
            else:
                if bound[5] - z >= 0:
                    z_idx = bound[5] - z
                else:
                    if bound[4] + z < z1:
                        z_idx = bound[4]
                    else:
                        z_idx = int((z1 - z) / 2)
        else:
            bound = [0, x1, 0, y1, 0, z1]
            if x1 - x == 0:
                x_idx = 0
            else:
                x_idx = np.random.randint(0, x1 - x)
            if y1 - y == 0:
                y_idx = 0
            else:
                y_idx = np.random.randint(0, y1 - y)
            if z1 - z == 0:
                z_idx = 0
            else:
                z_idx = np.random.randint(0, z1 - z)

        location = torch.zeros_like(tmp_scans.data).float()
        location[:, x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z] = 1

        # When brain bounding box is smaller than patch in a dimension, use the
        # local patch extent for the global crop so spatial correspondence is correct
        gx0 = x_idx if (bound[1] - bound[0] < x) else bound[0]
        gx1 = x_idx + x if (bound[1] - bound[0] < x) else bound[1]
        gy0 = y_idx if (bound[3] - bound[2] < y) else bound[2]
        gy1 = y_idx + y if (bound[3] - bound[2] < y) else bound[3]
        gz0 = z_idx if (bound[5] - bound[4] < z) else bound[4]
        gz1 = z_idx + z if (bound[5] - bound[4] < z) else bound[5]

        sbj = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans.data[:, gx0:gx1, gy0:gy1, gz0:gz1]),
                          a_segmentation=tio.LabelMap(tensor=location[:, gx0:gx1, gy0:gy1, gz0:gz1]))
        transforms = tio.transforms.Resize(target_shape=(x, y, z))
        sbj = transforms(sbj)
        down_scan = sbj['one_image'].data
        locA = sbj['a_segmentation'].data

        tmp_coor = get_bounds(locA)
        coordinates_A = np.array([np.floor(tmp_coor[0] / 4),
                                  np.ceil(tmp_coor[1] / 4),
                                  np.floor(tmp_coor[2] / 4),
                                  np.ceil(tmp_coor[3] / 4),
                                  np.floor(tmp_coor[4] / 4),
                                  np.ceil(tmp_coor[5] / 4)
                                  ]).astype(int)

        patchA = tmp_scans.data[:, x_idx:x_idx + x,
                                y_idx:y_idx + y, z_idx:z_idx + z].float()
        downA = down_scan.float()

        # Extract CMC pair patch at the same spatial location
        has_cmc_pair = tmp_scans_pair is not None
        if has_cmc_pair:
            patchA2 = tmp_scans_pair.data[:, x_idx:x_idx + x,
                                          y_idx:y_idx + y, z_idx:z_idx + z].float()
            _sbj2 = tio.Subject(one_image=tio.ScalarImage(
                tensor=tmp_scans_pair.data[:, gx0:gx1, gy0:gy1, gz0:gz1]))
            _sbj2 = tio.transforms.Resize(target_shape=(x, y, z))(_sbj2)
            downA2 = _sbj2['one_image'].data.float()
        else:
            # No paired modality available: add small Gaussian noise to the same image.
            # CMC then acts as a self-consistency / perturbation regularizer.
            patchA2 = (patchA + torch.randn_like(patchA) * 0.01).clamp(min=0)
            downA2 = (downA + torch.randn_like(downA) * 0.01).clamp(min=0)
            modality_A2 = modality_A

        '''
        load annotated data
        '''

        # tmp_scans = nib.load(tmp_path_B1[indexB])
        # tmp_scans = np.squeeze(tmp_scans.get_fdata())

        tmp_scans = preprocess_image(tmp_path_B1[indexB],target_spacing=(1.0, 1.0, 1.0))
        tmp_scans = np.squeeze(tmp_scans)
        '''
        WARNING: HERE WE ONLY USE POSITIVE INTENSITY
        FOR CT, USE PREPROCESSING TO turn negatives to positives

        '''
        tmp_scans[tmp_scans < 0] = 0
        # tmp_label = np.squeeze(
        #     np.round(nib.load(tmp_path_B2[indexB]).get_fdata()))

        tmp_label = preprocess_image(tmp_path_B2[indexB],target_spacing=(1.0, 1.0, 1.0), is_label=True)
        tmp_label = np.squeeze(np.round(tmp_label))

        # Remap labels to remove CSF and make datasets consistent
        # If label 13 (CSF) exists, change it to 0 (background)
        # Then change label 14 to 13, and set 14 to 0
        if 14 in np.unique(tmp_label):
            tmp_label[tmp_label == 14] = 0  # Remove CSF
        tmp_label[tmp_label == 15] = 14  # Shift label 15 to 14
        # Note: label 14 is now 13, so no pixels have value 14 anymore

        assert tmp_scans.shape == tmp_label.shape, (
            f'scan and label must have the same shape\n'
            f'Scan file: {tmp_path_B1[indexB]}, shape: {tmp_scans.shape}\n'
            f'Label file: {tmp_path_B2[indexB]}, shape: {tmp_label.shape}'
        )

        if self.cfg.data.normalize:
            if np.random.uniform() <= self.cfg.data.aug_prob:
                perc_dif = 100 - self.cfg.data.norm_perc
                tmp_scans = norm_img(tmp_scans, np.random.uniform(
                    self.cfg.data.norm_perc - perc_dif, 100))
            else:
                tmp_scans = norm_img(tmp_scans, self.cfg.data.norm_perc)

        if min(tmp_scans.shape) < min(x, y, z):
            x_diff = x-tmp_scans.shape[0]
            y_diff = y-tmp_scans.shape[1]
            z_diff = z-tmp_scans.shape[2]
            tmp_scans = np.pad(tmp_scans, ((max(0, int(x_diff/2)), max(0, x_diff-int(x_diff/2))), (max(0, int(
                y_diff/2)), max(0, y_diff-int(y_diff/2))), (max(0, int(z_diff/2)), max(0, z_diff-int(z_diff/2)))), constant_values=1e-4)  # cant pad with 0s, otherwise the local and global patches wont be the same location
            tmp_label = np.pad(tmp_label, ((max(0, int(x_diff/2)), max(0, x_diff-int(x_diff/2))), (max(0, int(
                y_diff/2)), max(0, y_diff-int(y_diff/2))), (max(0, int(z_diff/2)), max(0, z_diff-int(z_diff/2)))), constant_values=0)  # pad with 0s bc it is label
            tmp_scans = torch.unsqueeze(torch.from_numpy(tmp_scans), 0)
            tmp_label = torch.unsqueeze(torch.from_numpy(tmp_label), 0)

        else:
            tmp_scans = torch.unsqueeze(
                torch.from_numpy(tmp_scans.copy()), 0)
            tmp_label = torch.unsqueeze(
                torch.from_numpy(tmp_label.copy()), 0)

        _, x1, y1, z1 = tmp_scans.shape
        tmp_scans = tio.ScalarImage(tensor=tmp_scans)
        tmp_label = tio.LabelMap(tensor=tmp_label)
        sbj = tio.Subject(one_image=tmp_scans, a_segmentation=tmp_label)
        if self.cfg.data.aug:
            transforms = tio.Compose([tio.RandomAffine(p=self.cfg.data.aug_prob, scales=(0.7, 1.4), degrees=30,
                                                       isotropic=True,
                                                       default_pad_value=0, image_interpolation='linear',
                                                       label_interpolation='nearest'),
                                      tio.RandomBiasField(
                                      p=self.cfg.data.aug_prob),
                                      tio.RandomGamma(
                                      p=self.cfg.data.aug_prob, log_gamma=(-0.4, 0.4))
                                      ])
            sbj = transforms(sbj)
        tmp_scans = sbj['one_image'].data.float()
        tmp_label = sbj['a_segmentation'].data.float()

        if self.cfg.data.remove_bg:
            bound = get_bounds(tmp_scans.data)
            if bound[1] - x > bound[0]:
                x_idx = np.random.randint(bound[0], bound[1] - x)
            else:
                if bound[1] - x >= 0:
                    x_idx = bound[1] - x
                else:
                    if bound[0] + x < x1:
                        x_idx = bound[0]
                    else:
                        x_idx = int((x1 - x) / 2)
            if bound[3] - y > bound[2]:
                y_idx = np.random.randint(bound[2], bound[3] - y)
            else:
                if bound[3] - y >= 0:
                    y_idx = bound[3] - y
                else:
                    if bound[2] + y < y1:
                        y_idx = bound[2]
                    else:
                        y_idx = int((y1 - y) / 2)

            if bound[5] - z > bound[4]:
                z_idx = np.random.randint(bound[4], bound[5] - z)
            else:
                if bound[5] - z >= 0:
                    z_idx = bound[5] - z
                else:
                    if bound[4] + z < z1:
                        z_idx = bound[4]
                    else:
                        z_idx = int((z1 - z) / 2)
        else:
            bound = [0, x1, 0, y1, 0, z1]
            if x1 - x == 0:
                x_idx = 0
            else:
                x_idx = np.random.randint(0, x1 - x)
            if y1 - y == 0:
                y_idx = 0
            else:
                y_idx = np.random.randint(0, y1 - y)
            if z1 - z == 0:
                z_idx = 0
            else:
                z_idx = np.random.randint(0, z1 - z)

        location_B = torch.zeros_like(tmp_scans.data).float()
        location_B[:, x_idx:x_idx + x,
                   y_idx:y_idx + y, z_idx:z_idx + z] = 1

        # When brain bounding box is smaller than patch in a dimension, use the
        # local patch extent for the global crop so spatial correspondence is correct
        gx0 = x_idx if (bound[1] - bound[0] < x) else bound[0]
        gx1 = x_idx + x if (bound[1] - bound[0] < x) else bound[1]
        gy0 = y_idx if (bound[3] - bound[2] < y) else bound[2]
        gy1 = y_idx + y if (bound[3] - bound[2] < y) else bound[3]
        gz0 = z_idx if (bound[5] - bound[4] < z) else bound[4]
        gz1 = z_idx + z if (bound[5] - bound[4] < z) else bound[5]

        sbj = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans[:, gx0:gx1, gy0:gy1, gz0:gz1]),
                          a_segmentation=tio.LabelMap(tensor=location_B[:, gx0:gx1, gy0:gy1, gz0:gz1]))
        transforms = tio.transforms.Resize(target_shape=(x, y, z))
        sbj = transforms(sbj)
        down_scan = sbj['one_image'].data.float()
        locB = sbj['a_segmentation'].data

        tmp_coor = get_bounds(locB)
        sbj = tio.Subject(one_image=tio.ScalarImage(tensor=tmp_scans[:, gx0:gx1, gy0:gy1, gz0:gz1]),
                          a_segmentation=tio.LabelMap(tensor=tmp_label[:, gx0:gx1, gy0:gy1, gz0:gz1]))
        sbj = transforms(sbj)
        aux_label = sbj['a_segmentation'].data

        coordinates_B = np.array([np.floor(tmp_coor[0] / 4),
                                  np.ceil(tmp_coor[1] / 4),
                                  np.floor(tmp_coor[2] / 4),
                                  np.ceil(tmp_coor[3] / 4),
                                  np.floor(tmp_coor[4] / 4),
                                  np.ceil(tmp_coor[5] / 4)
                                  ]).astype(int)
        # modality_A already extracted early; extract modality_B here
        fname_B = os.path.basename(tmp_path_B1[indexB])
        modality_B = get_modality_from_filename(fname_B, self._supported_modalities) or 'T1w'

        # Extract age group from demographics via sub_id + ses_id parsed from filename.
        # Falls back to dataset-level age group (from parent folder name) if subject not found.
        def _age_group_from_path(full_path):
            import re
            fname = os.path.basename(full_path)
            sub_m = re.search(r'(sub-[^_.\s]+)', fname)
            ses_m = re.search(r'(ses-[^_.\s]+)', fname)
            sub = sub_m.group(1) if sub_m else fname.replace('.nii.gz', '').replace('.nii', '')
            ses = ses_m.group(1) if ses_m else ''
            grp = self._age_lookup.get((sub, ses), self._age_lookup.get((sub, ''), None))
            if grp is not None:
                return grp
            # Fallback: extract dataset name from parent folder (e.g. BCP_T1w_train -> BCP)
            folder = os.path.basename(os.path.dirname(full_path))
            dataset = re.split(r'[_\-]', folder)[0]
            return self._dataset_age_lookup.get(dataset, 'adult')

        age_group_A = _age_group_from_path(tmp_path[indexA])
        age_group_B = _age_group_from_path(tmp_path_B1[indexB])

        input_dict = {'imgB': tmp_scans[:, x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z],
                      'labelB': torch.squeeze(tmp_label[:, x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z]),
                      'label_B_aux': torch.squeeze(aux_label),
                      'downB': down_scan,
                      'cord_B': coordinates_B,
                      'imgA': patchA,
                      'downA': downA,
                      'cord_A': coordinates_A,
                      'modality_A': modality_A,    # Target domain modality
                      'modality_B': modality_B,    # Source domain modality
                      'age_group_A': age_group_A,  # Target domain age group
                      'age_group_B': age_group_B,  # Source domain age group
                      # CMC pair (same subject, different modality); zeros + flag=False if unavailable
                      'imgA2': patchA2,
                      'downA2': downA2,
                      'modality_A2': modality_A2 if has_cmc_pair else modality_A,
                      'has_cmc_pair': has_cmc_pair}

        return input_dict

    def __len__(self):

        # we used fixed 100 steps for each epoch in finetuning
        # THIS PARAM WAS NEVER TUNED
        return 100


class LifespanAgeDataset(data.Dataset):
    """
    Dataset for lifespan age prediction (Stage 1: Classification)

    Similar to synthesis dataset structure with age/gender metadata from CSV.

    Age encoding rules in CSV:
    - Fetal: "X wk" (gestational weeks, e.g., "27.63 wk")
    - Neonatal: "X wk(neonatal)" (postnatal weeks, e.g., "8.5 wk(neonatal)")
    - Infant: "X mo" (months, e.g., "15 mo")
    - Child/Adolescent: "X" plain number, 2-17 years (e.g., "10")
    - Adult: "X" plain number, 18-65 years (e.g., "35")
    - Elderly: "X" plain number, >65 years (e.g., "75")

    Converts to 6 stage labels:
    0: Fetal, 1: Neonatal, 2: Infant, 3: Child/Adol, 4: Adult, 5: Elderly
    """

    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split

        # Load domains from file
        with open(cfg.data.age_domain_file) as f:
            self.domains = f.read().splitlines()

        # Load subject list from CSV
        csv_file = cfg.data.age_train_csv if split == 'train' else cfg.data.age_val_csv
        df = pd.read_csv(csv_file)

        # Create metadata dictionary: (sub_id, ses_id) -> {age, gender, dataset}
        self.metadata_dict = {}
        for _, row in df.iterrows():
            key = (row['sub_id'], row['ses_id'])
            age_value, age_unit = self._parse_age_string(row['age'])
            self.metadata_dict[key] = {
                'age': age_value,
                'age_unit': age_unit,
                'age_string': str(row['age']),  # Keep original string
                'gender': row['gender'],
                'dataset': row['dataset']
            }

        self.path_dic = {}
        self.subject_count = {}
        self.image_count = {}

        total_subjects = set()
        total_images = 0

        for idx, domain in enumerate(self.domains):
            domain_path = os.path.join(cfg.data.age_root, domain)
            nii_paths = []

            # Collect .nii.gz files from registered folders (same as synthesis)
            search_dirs = []
            if domain in ["ABCD", "ADHD", "ABIDE"]:
                search_dirs += [
                    os.path.join(domain_path, "Resampled", "*", "sub-*", "ses-*", "anat", "*.nii.gz"),
                ]
            else:
                search_dirs += [
                    os.path.join(domain_path, "Resampled", "sub-*", "ses-*", "anat", "*.nii.gz"),
                ]

            # Gather files and organize by subject-session
            subject_modalities = {}  # {(sub_id, ses_id): {'T1w': path, 'T2w': path, 'FA': path, 'age': ..., 'stage': ...}}

            for pattern in search_dirs:
                matches = glob.glob(pattern)
                for f in matches:
                    parts = f.split(os.sep)
                    sub_id = next((p for p in parts if p.startswith("sub-")), None)
                    ses_id = next((p for p in parts if p.startswith("ses-")), None)

                    # Only include if (sub_id, ses_id) is in CSV
                    if (sub_id, ses_id) in self.metadata_dict:
                        key = (sub_id, ses_id)
                        if key not in subject_modalities:
                            # Initialize with metadata
                            metadata = self.metadata_dict[key]
                            age_value = metadata['age']
                            age_unit = metadata['age_unit']
                            stage_label = self._age_to_stage(age_value, age_unit)

                            subject_modalities[key] = {
                                'age': age_value,
                                'age_unit': age_unit,
                                'age_string': metadata['age_string'],
                                'stage_label': stage_label,
                                'gender': metadata['gender'],
                                'dataset': metadata['dataset']
                            }

                        # Determine modality from filename
                        filename = os.path.basename(f)
                        for mod in ['T1w', 'T2w', 'FA']:
                            if mod in filename:
                                subject_modalities[key][mod] = f
                                nii_paths.append(f)
                                break

            # Store organized subject-session data
            self.path_dic[str(idx)] = subject_modalities
            self.image_count[domain] = len(nii_paths)
            total_images += len(nii_paths)

            # Track subjects
            domain_subjects = set()
            for (sub_id, ses_id) in subject_modalities.keys():
                domain_subjects.add(sub_id)
                total_subjects.add(sub_id)
            self.subject_count[domain] = len(domain_subjects)

            # Domain summary
            print(f"📁 Domain: {domain}")
            print(f"  🧠 Subjects: {self.subject_count[domain]}")
            print(f"  📄 Images : {self.image_count[domain]}")

        print("====================================")
        print(f"✅ Total domains  : {len(self.domains)}")
        print(f"👤 Total subjects : {len(total_subjects)}")
        print(f"🖼️  Total images   : {total_images}")
        print("====================================")

        self.num_domain = len(self.domains)
        self.all_img = total_images
        self.all_subjects = len(total_subjects)

        # Print stage distribution
        self._print_stage_distribution()

    def _parse_age_string(self, age_str):
        """
        Parse age string to extract numeric value and unit

        Examples:
        - "27.63 wk" -> (27.63, "wk") [Fetal]
        - "8.5 wk(neonatal)" -> (8.5, "wk(neonatal)") [Neonatal]
        - "15 mo" -> (15, "mo") [Infant]
        - "25" or "25.5" -> (25.5, "y") [Years: Child/Adult/Elderly]

        Returns:
            (float, str): (age_value, age_unit)
        """
        age_str = str(age_str).strip()

        # Check for "wk(neonatal)" pattern
        if "wk(neonatal)" in age_str.lower():
            value = float(age_str.split()[0])
            return (value, "wk(neonatal)")

        # Check for "wk" pattern (fetal)
        elif " wk" in age_str.lower():
            value = float(age_str.split()[0])
            return (value, "wk")

        # Check for "mo" pattern (infant)
        elif " mo" in age_str.lower():
            value = float(age_str.split()[0])
            return (value, "mo")

        # Plain number means years
        else:
            try:
                value = float(age_str)
                return (value, "y")
            except ValueError:
                print(f"Warning: Could not parse age string: {age_str}")
                return (None, None)

    def _age_to_stage(self, age_value, age_unit):
        """
        Convert age value and unit to lifespan stage

        Age encoding rules:
        - "wk" (without neonatal): Fetal (gestational weeks, typically 20-40)
        - "wk(neonatal)": Neonatal (postnatal weeks, 0-12 wk = 0-3 months)
        - "mo": Infant (months, typically 3-24 months)
        - "y" (plain number): Years
          - 2-17: Child/Adolescent
          - 18-65: Adult
          - >65: Elderly

        Returns:
            0: Fetal, 1: Neonatal, 2: Infant, 3: Child/Adol, 4: Adult, 5: Elderly
        """
        if age_value is None or age_unit is None:
            return None

        # Fetal: gestational weeks (unit = "wk")
        if age_unit == "wk":
            return 0

        # Neonatal: postnatal weeks (unit = "wk(neonatal)")
        elif age_unit == "wk(neonatal)":
            return 1

        # Infant: months (unit = "mo")
        elif age_unit == "mo":
            return 2

        # Years: Child/Adolescent/Adult/Elderly
        elif age_unit == "y":
            if age_value < 2:
                return 2  # Very young, likely infant
            elif age_value < 18:
                return 3  # Child/Adolescent
            elif age_value <= 65:
                return 4  # Adult
            else:
                return 5  # Elderly

        else:
            print(f"Warning: Unknown age unit: {age_unit}")
            return None

    def _print_stage_distribution(self):
        """Print distribution of samples across lifespan stages"""
        stage_names = ['Fetal', 'Neonatal', 'Infant', 'Child/Adol', 'Adult', 'Elderly']
        stage_counts = [0] * 6

        for domain_idx in range(self.num_domain):
            subject_dict = self.path_dic[str(domain_idx)]
            for (sub_id, ses_id), data in subject_dict.items():
                if data['stage_label'] is not None:
                    stage_counts[data['stage_label']] += 1

        total_samples = sum(stage_counts)
        print("\nLifespan Stage Distribution:")
        print("-" * 50)
        for i, (name, count) in enumerate(zip(stage_names, stage_counts)):
            pct = 100.0 * count / total_samples if total_samples > 0 else 0
            print(f"  {i}: {name:12s} - {count:5d} ({pct:5.1f}%)")
        print("-" * 50)

    def __getitem__(self, index):
        """
        Returns a single image with its stage label

        Pipeline:
        1. Randomly select a domain
        2. Randomly select a subject-session from that domain
        3. Randomly select an available modality
        4. Load image and normalize
        5. Pad/crop to 192^3
        6. Apply rotation augmentation (training only, with aug_prob probability)
        7. Downsample to 96^3

        Augmentation (training only):
        - Random rotation: ±15 degrees around each axis
        - Applied with probability = cfg.data.aug_prob (default: 0.3)
        """
        # Random domain selection (like synthesis)
        idx = int(np.random.random_sample() // (1 / self.num_domain))
        subject_dict = self.path_dic[str(idx)]

        # Random subject-session selection from domain
        if len(subject_dict) == 0:
            # Fallback if domain is empty
            idx = (idx + 1) % self.num_domain
            subject_dict = self.path_dic[str(idx)]

        subject_keys = list(subject_dict.keys())
        random_key = subject_keys[np.random.randint(0, len(subject_keys))]
        subject_data = subject_dict[random_key]

        # Skip if stage label is invalid
        if subject_data['stage_label'] is None:
            return self.__getitem__((index + 1) % self.__len__())

        # Find available modalities
        available_modalities = [mod for mod in ['T1w', 'T2w', 'FA'] if mod in subject_data]
        if len(available_modalities) == 0:
            # No modality available, skip
            return self.__getitem__((index + 1) % self.__len__())

        # Randomly select one modality
        selected_modality = available_modalities[np.random.randint(0, len(available_modalities))]
        img_path = subject_data[selected_modality]

        try:
            # Load image
            img_data = preprocess_image(img_path, target_spacing=(1.0, 1.0, 1.0), resample=False)
            img_data[img_data < 0] = 0

            # Normalization
            if self.cfg.data.normalize:
                if np.random.uniform() <= self.cfg.data.aug_prob and self.split == 'train':
                    perc_dif = 100 - self.cfg.data.norm_perc
                    img_data = norm_img(img_data, np.random.uniform(
                        self.cfg.data.norm_perc - perc_dif, 100))
                else:
                    img_data = norm_img(img_data, self.cfg.data.norm_perc)
                if img_data is None:
                    raise ValueError(f"Normalization failed for: {img_path}")

            # Pad or crop to 192^3 (better for fetal brains which are smaller)
            target_size = 192
            x, y, z = img_data.shape

            # Pad if needed
            if x < target_size or y < target_size or z < target_size:
                pad_x = max(0, target_size - x)
                pad_y = max(0, target_size - y)
                pad_z = max(0, target_size - z)
                img_data = np.pad(img_data, (
                    (pad_x // 2, pad_x - pad_x // 2),
                    (pad_y // 2, pad_y - pad_y // 2),
                    (pad_z // 2, pad_z - pad_z // 2)
                ), mode='constant', constant_values=0)

            # Crop if needed (center crop)
            x, y, z = img_data.shape
            if x > target_size or y > target_size or z > target_size:
                x_start = (x - target_size) // 2
                y_start = (y - target_size) // 2
                z_start = (z - target_size) // 2
                img_data = img_data[
                    x_start:x_start + target_size,
                    y_start:y_start + target_size,
                    z_start:z_start + target_size
                ]

            # Convert to tensor and add channel dimension
            img_tensor = torch.from_numpy(img_data.astype(np.float32)).unsqueeze(0)

            # Create TorchIO subject
            sbj = tio.Subject(image=tio.ScalarImage(tensor=img_tensor))

            # Apply augmentation (rotation) during training only
            if self.cfg.data.aug and self.split == 'train':
                if np.random.uniform() <= self.cfg.data.aug_prob:
                    # Random rotation: up to ±15 degrees around each axis
                    rotation_transform = tio.RandomAffine(
                        scales=0,  # No scaling
                        degrees=15,  # ±15 degrees rotation
                        translation=0,  # No translation
                        isotropic=False,  # Allow different rotations per axis
                        default_pad_value=0,
                        image_interpolation='linear'
                    )
                    sbj = rotation_transform(sbj)

            # Downsample 192^3 -> 96^3 using TorchIO
            resize_transform = tio.transforms.Resize(target_shape=self.cfg.data.patch_size)
            sbj = resize_transform(sbj)
            img_96 = sbj['image'].data  # Already (1, 96, 96, 96)

            return {
                'image': img_96,  # (1, 96, 96, 96) - tensor
                'stage_label': torch.tensor(subject_data['stage_label'], dtype=torch.long),  # 0-5 - tensor
                'modality': selected_modality,  # T1w/T2w/FA - string (needed for MoE)
            }

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
            # Return next sample
            return self.__getitem__((index + 1) % self.__len__())

    def __len__(self):
        # Fixed epoch size (like synthesis and MAE datasets)
        return 1000
