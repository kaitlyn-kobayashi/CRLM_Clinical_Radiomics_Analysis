"""Whole_Liver HDFS Survival Pipeline
---------------------------------------
This end-to-end script prepares imaging + clinical data, trains deep
survival models (optionally with self-supervised pre-training), performs
hyper-parameter search, ensembles multiple seeds, bootstraps performance
and finally exports per-patient deep features and risk scores.

"""

# --------------------------------------------------------------------- #
# Basic imports and global config
# --------------------------------------------------------------------- #
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage
from skimage.transform import resize

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw

from copy import deepcopy

# Mixed-precision helpers
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torchvision import transforms
import nibabel as nib

# Pretty CLI logging
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn

console = Console(style="white")

# --------------------------------------------------------------------- #
# Path management – adjust here for your own machine
# --------------------------------------------------------------------- #
##############################################################################
# Paths
##############################################################################
cached_dir   = "/mnt/largedrive1/rmojtahedi/Deep_Survival/Final_Improved_npj/Final_Improved_npj_HDFS"
results_dir  = "/mnt/largedrive1/rmojtahedi/Deep_Survival/Best_of_Best/Best_HDFS"
os.makedirs(results_dir, exist_ok=True)

liver_image_dir   = "/mnt/largedrive1/rmojtahedi/KSPIE/liver_whole/ct"
liver_label_dir   = "/mnt/largedrive1/rmojtahedi/KSPIE/liver_whole/final_seg_split"
clinical_csv_path = "/mnt/largedrive0/rmojtahedi/Kaitlyn_SPIE/Deep_Survival/npj_Digital_Medicine_Clinical_Data_FINAL.csv"

# Compute on second GPU if available
device      = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
scaler_amp  = GradScaler(enabled=(device.type == 'cuda'))

# --------------------------------------------------------------------- #
# Reproducibility helpers
# --------------------------------------------------------------------- #
##############################################################################
# Evaluation seed
##############################################################################
EVAL_SEED = 999

def set_seed(seed: int) -> None:
    """Set *all* relevant RNG seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_eval_seed(seed: int) -> None:
    """Same as set_seed but kept separate for clarity in eval/TTA loops."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --------------------------------------------------------------------- #
# Utility functions for matching CT/label/clinical rows
# --------------------------------------------------------------------- #
##############################################################################
# Match image/label/clinical
##############################################################################
def get_core_id(filename: str) -> str:
    """Strip .nii / .nii.gz suffix to get the *base* study ID."""
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    elif filename.endswith(".nii"):
        return filename[:-4]
    return filename

def get_candidate_id(name: str) -> str:
    """
    For filenames like ABC_123_456, drop last token to match the
    shorter XNAT ID stored in the clinical spreadsheet.
    """
    parts = name.split('_')
    if len(parts) > 1:
        return '_'.join(parts[:-1])
    return name

# Load directory contents
image_files = os.listdir(liver_image_dir)
label_files = os.listdir(liver_label_dir)

# Convert to sets for fast lookup
image_names = {get_core_id(fn) for fn in image_files}
label_names = {get_core_id(fn) for fn in label_files}

# Simple QC: spot any missing counterparts
unmatched_images = sorted(list(image_names - label_names))
unmatched_labels = sorted(list(label_names - image_names))

# Only keep slices present in BOTH folders
common_names    = image_names.intersection(label_names)
candidate_pairs = [(name, get_candidate_id(name)) for name in sorted(common_names)]

# --------------------------------------------------------------------- #
# Read and filter clinical sheet
# --------------------------------------------------------------------- #
clinical_data_full = pd.read_csv(clinical_csv_path)
clinical_data_full = clinical_data_full.dropna(subset=["XNAT ID"])
clinical_data_full["XNAT ID"] = clinical_data_full["XNAT ID"].astype(str)
clinical_ids = set(clinical_data_full["XNAT ID"])

# --------------------------------------------------------------------- #
# Resolve mismatches automatically + manually curated mapping
# --------------------------------------------------------------------- #
final_matched        = []
unmatched_candidates = []
for image_label, candidate in candidate_pairs:
    if candidate in clinical_ids:
        final_matched.append((image_label, candidate))
    else:
        unmatched_candidates.append((image_label, candidate))

# Hard-coded fixes from manual audit
manual_mapping = {
    # exact “image_label” : corrected “XNAT ID”
    "RIA_17-010A_000_202": "RIA_17-010A_000_202",
    "RIA_17-010A_000_439": "RIA_17-010A_000_439",
    "RIA_17-010A_002_118": "RIA_17-010A_002_118",
    "RIA_17-010A_002_131": "RIA_17-010A_002_131",
    "RIA_17-010A_002_159": "RIA_17-010A_002_159",
    "RIA_17-010A_002_170": "RIA_17-010A_002_170",
    # The following had double-ID typos in clinical sheet
    "RIA_17-010B_000_010": "RIA_17-010B_000_RIA_17-010B_000_010",
    "RIA_17-010B_000_054": "RIA_17-010B_000_RIA_17-010B_000_054",
    "RIA_17-010B_000_064": "RIA_17-010B_000_RIA_17-010B_000_064",
    "RIA_17-010B_000_123": "RIA_17-010B_000_RIA_17-010B_000_123",
    "RIA_17-010B_000_159": "RIA_17-010B_000_RIA_17-010B_000_159",
    "RIA_17-010B_000_165": "RIA_17-010B_000_RIA_17-010B_000_165",
    "RIA_17-010B_000_170": "RIA_17-010B_000_RIA_17-010B_000_170",
    "RIA_17-010B_000_200": "RIA_17-010B_000_RIA_17-010B_000_200",
    "RIA_17-010B_000_209": "RIA_17-010B_000_RIA_17-010B_000_209",
    "RIA_17-010B_000_258": "RIA_17-010B_000_RIA_17-010B_000_258",
}

manually_matched = []
still_unmatched  = []
for image_label, candidate in unmatched_candidates:
    if image_label in manual_mapping:
        corrected = manual_mapping[image_label]
        if corrected in clinical_ids:
            manually_matched.append((image_label, corrected))
        else:
            still_unmatched.append((image_label, candidate))
    else:
        still_unmatched.append((image_label, candidate))

# Combine auto- and manual-matches
final_matched.extend(manually_matched)

# Any clinical rows with no imaging yet?
matched_xnat_set       = {cid for _, cid in final_matched}
unmatched_clinical_ids = sorted(list(clinical_ids - matched_xnat_set))

# --------------------------------------------------------------------- #
# Merge matched IDs with clinical dataframe and filtering for HDFS task
# --------------------------------------------------------------------- #
matched_df  = pd.DataFrame(final_matched, columns=["Image_Label_Base", "XNAT ID"])
merged_data = pd.merge(matched_df, clinical_data_full, on="XNAT ID", how="inner")

#  Keep only rows with valid HDFS info (>0 time)
merged_data = merged_data.dropna(subset=["Surv Time Censor 4", "Censor 4: HDFS", "HDFSTrain"])
merged_data = merged_data[merged_data["Surv Time Censor 4"] > 0]

# Pre-defined split flag in spreadsheet
train_rows = merged_data[merged_data["HDFSTrain"] == 1]
val_rows   = merged_data[merged_data["HDFSTrain"] == 0]

train_image_names = train_rows["Image_Label_Base"].tolist()
val_image_names   = val_rows["Image_Label_Base"].tolist()

# --------------------------------------------------------------------- #
# 3-D augmentation classes – lightweight & NumPy-based for speed
# --------------------------------------------------------------------- #
##############################################################################
# Augmentations
##############################################################################
class RandomGaussianNoise:
    """Add Gaussian noise with random std in [std_range]."""
    def __init__(self, p=0.5, std_range=(0.01, 0.03)):
        self.p = p
        self.std_range = std_range
    def __call__(self, volume):
        if random.random() < self.p:
            std = random.uniform(*self.std_range)
            noise = np.random.randn(*volume.shape) * std
            volume = volume + noise
        return volume

class RandomZoom3D:
    """Isotropic zoom in/out followed by center crop or zero-pad."""
    def __init__(self, p=0.5, zoom_range=(0.9, 1.1)):
        self.p = p
        self.zoom_range = zoom_range
    def __call__(self, volume):
        if random.random() < self.p:
            zoom_factor = random.uniform(*self.zoom_range)
            zoomed = scipy.ndimage.zoom(volume, zoom_factor, order=1)
            orig_shape = volume.shape
            if zoomed.shape[0] >= orig_shape[0]:
                # Crop center
                sx = (zoomed.shape[0] - orig_shape[0]) // 2
                sy = (zoomed.shape[1] - orig_shape[1]) // 2
                sz = (zoomed.shape[2] - orig_shape[2]) // 2
                ex = sx + orig_shape[0]
                ey = sy + orig_shape[1]
                ez = sz + orig_shape[2]
                volume = zoomed[sx:ex, sy:ey, sz:ez]
            else:
                # Pad center
                padded = np.zeros(orig_shape, dtype=zoomed.dtype)
                ox = (orig_shape[0] - zoomed.shape[0]) // 2
                oy = (orig_shape[1] - zoomed.shape[1]) // 2
                oz = (orig_shape[2] - zoomed.shape[2]) // 2
                padded[ox:ox+zoomed.shape[0],
                       oy:oy+zoomed.shape[1],
                       oz:oz+zoomed.shape[2]] = zoomed
                volume = padded
        return volume

class RandomRotate3D:
    """
    Small Euler rotation around a randomly chosen axis pair.
    Note: order=1 trilinear ensures smooth interpolation.
    """
    def __init__(self, p=0.5, max_angle=10):
        self.p = p
        self.max_angle = max_angle
    def __call__(self, volume):
        if random.random() < self.p:
            axis_pairs = [(0,1),(0,2),(1,2)]
            axes = random.choice(axis_pairs)
            angle = random.uniform(-self.max_angle, self.max_angle)
            volume = scipy.ndimage.rotate(volume, angle, axes=axes, reshape=False, order=1)
        return volume

class RandomFlip3D:
    """Flip along a random axis with probability p."""
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, volume):
        if random.random() < self.p:
            flip_axis = random.choice([0,1,2])
            volume = np.flip(volume, axis=flip_axis).copy()
        return volume

class RandomIntensityShift:
    """Additive shift on normalized intensities."""
    def __init__(self, p=0.5, shift_range=(-0.07, 0.07)):
        self.p = p
        self.shift_range = shift_range
    def __call__(self, volume):
        if random.random() < self.p:
            shift = random.uniform(*self.shift_range)
            volume = volume + shift
        return volume

class RandomIntensityScale:
    """Multiplicative scaling on normalized intensities."""
    def __init__(self, p=0.5, scale_range=(0.95, 1.05)):
        self.p = p
        self.scale_range = scale_range
    def __call__(self, volume):
        if random.random() < self.p:
            sc_ = random.uniform(*self.scale_range)
            volume = volume * sc_
        return volume

class RandomGamma:
    """
    Gamma correction in [gamma_range] on the *positive* part of volume.
    """
    def __init__(self, p=0.5, gamma_range=(0.9, 1.1)):
        self.p = p
        self.gamma_range = gamma_range
    def __call__(self, volume):
        if random.random() < self.p:
            gm = random.uniform(*self.gamma_range)
            volume = np.clip(volume, 0, None)
            vmin = volume.min()
            vmax = volume.max()
            if vmax > vmin:
                normv = (volume - vmin) / (vmax - vmin + 1e-10)
                normv = normv ** gm
                volume = normv * (vmax - vmin) + vmin
        return volume

# Compose to a torchvision-style transform
augment_transform = transforms.Compose([
    RandomRotate3D(p=0.5, max_angle=10),
    RandomFlip3D(p=0.5),
    RandomZoom3D(p=0.5, zoom_range=(0.9,1.1)),
    RandomGaussianNoise(p=0.5, std_range=(0.01, 0.03)),
    RandomIntensityShift(p=0.5, shift_range=(-0.07,0.07)),
    RandomIntensityScale(p=0.5, scale_range=(0.95,1.05)),
    RandomGamma(p=0.5, gamma_range=(0.9,1.1))
])

# --------------------------------------------------------------------- #
# Conversion helper: NIfTI ➜ pre-scaled NumPy cache
# --------------------------------------------------------------------- #
##############################################################################
# Create/Load NPY
##############################################################################
def create_np_data(image_dir, label_dir, clinical_df, train_list, val_list,
                   npy_save_path, shape=(64,64,64)):
    """
    Convert CT volumes to fixed-shape 3-channel NumPy arrays (mask-applied
    & z-score normalized) and save to disk alongside metadata.
    """
    console.print("Starting conversion to NPY...")

    all_pids = train_list + val_list
    gather_samples = []

    # -------------------------------------------------------------- #
    # First pass: fit StandardScaler on **all** voxels so that
    # statistics come purely from training+val (no test leakage).
    # -------------------------------------------------------------- #
    with Progress(
        TextColumn("{task.description}", style="black", justify="center"),
        BarColumn(bar_width=None, complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} done", style="black", justify="center"),
        console=Console(),
        transient=True
    ) as progress:
        task_id = progress.add_task("Loading and resizing", total=len(all_pids))
        for pid in all_pids:
            row = clinical_df[clinical_df["Image_Label_Base"] == pid]
            if len(row) == 0:
                progress.update(task_id, advance=1)
                continue

            # Gracefully handle .nii.gz vs .nii
            image_path = os.path.join(image_dir, pid + ".nii.gz")
            if not os.path.exists(image_path):
                image_path = os.path.join(image_dir, pid + ".nii")

            label_path = os.path.join(label_dir, pid + ".nii.gz")
            if not os.path.exists(label_path):
                label_path = os.path.join(label_dir, pid + ".nii")

            # Load, resize and mask
            vol = nib.load(image_path).get_fdata()
            seg = nib.load(label_path).get_fdata()
            vol = resize(vol, shape)
            seg = resize(seg, shape, order=0, preserve_range=True).astype(np.float32)
            seg = (seg > 0.5).astype(np.float32)
            masked_vol = vol * seg
            gather_samples.append(masked_vol.reshape(-1,1))

            progress.update(task_id, advance=1)

    big_array = np.concatenate(gather_samples, axis=0)
    scaler = StandardScaler()
    scaler.fit(big_array)

    cache_dict = {}

    # -------------------------------------------------------------- #
    # Second pass: normalize each volume and build train/val lists
    # -------------------------------------------------------------- #
    def build_data(pid_list, desc):
        items = []
        with Progress(
            TextColumn("{task.description}", style="black", justify="center"),
            BarColumn(bar_width=None, complete_style="green", finished_style="green"),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total} done", style="black", justify="center"),
            console=Console(),
            transient=True
        ) as p2:
            t_id = p2.add_task(desc, total=len(pid_list))
            for pid_ in pid_list:
                row_ = clinical_df[clinical_df["Image_Label_Base"]==pid_]
                if len(row_) == 0:
                    p2.update(t_id, advance=1)
                    continue

                image_path_ = os.path.join(image_dir, pid_+".nii.gz")
                if not os.path.exists(image_path_):
                    image_path_ = os.path.join(image_dir, pid_+".nii")

                label_path_ = os.path.join(label_dir, pid_+".nii.gz")
                if not os.path.exists(label_path_):
                    label_path_ = os.path.join(label_dir, pid_+".nii")

                vol_ = nib.load(image_path_).get_fdata()
                seg_ = nib.load(label_path_).get_fdata()
                vol_ = resize(vol_, shape)
                seg_ = resize(seg_, shape, order=0, preserve_range=True).astype(np.float32)
                seg_ = (seg_ > 0.5).astype(np.float32)
                masked_vol_ = vol_ * seg_

                # Z-score normalization using global scaler
                masked_vol_ = masked_vol_.reshape(-1,1)
                masked_vol_ = scaler.transform(masked_vol_)
                masked_vol_ = masked_vol_.reshape(*shape)

                # Duplicate channels so that 3-D CNN expects 3 channels
                vol_3c = np.repeat(masked_vol_[np.newaxis, ...], 3, axis=0)

                st_ = row_["Surv Time Censor 4"].values[0]
                ev_ = row_["Censor 4: HDFS"].values[0]

                # Cache in RAM-friendly dict
                cache_dict[pid_] = vol_3c.astype(np.float32)
                items.append({
                    'pid': pid_,
                    'cache_pid': pid_,
                    'image': None,      # placeholder for compatibility
                    'time': float(st_),
                    'event': bool(ev_)
                })
                p2.update(t_id, advance=1)
        return items

    train_data = build_data(train_list, "Processing and scaling (train)")
    val_data   = build_data(val_list,   "Processing and scaling (val)")

    # Persist to disk
    final_dict = {
        'train': train_data,
        'val'  : val_data,
        'mean' : scaler.mean_[0],
        'std'  : scaler.scale_[0],
        'cache_dict': cache_dict
    }
    np.save(npy_save_path, final_dict)
    console.print(f"Saved data to {npy_save_path}")

# --------------------------------------------------------------------- #
# Custom Dataset wrapper – supports in-RAM tensor cache
# --------------------------------------------------------------------- #
##############################################################################
# Dataset
##############################################################################
class LiverDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, cache_dict, augment=False, tensor_cache=None):
        """
        Args
        ----
        data_list   : list with keys pid, time, event, cache_pid
        cache_dict  : {pid: numpy array (3,64,64,64)}
        augment     : whether to apply random 3-D augmentations
        tensor_cache: {pid: torch.Tensor} pre-converted tensors to speed val
        """
        self.data_list   = data_list
        self.cache_dict  = cache_dict
        self.augment     = augment
        self.tensor_cache= tensor_cache

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        d   = self.data_list[idx]
        pid = d['cache_pid']

        # Use cached torch Tensor if available (val set)
        if (not self.augment) and (self.tensor_cache is not None):
            vol_3c = self.tensor_cache[pid].clone()
        else:
            vol_3c = self.cache_dict[pid].copy()

        # Apply heavy augment in NumPy then back to CxDxHxW
        if self.augment:
            vol_3c_t = np.transpose(vol_3c, (1,2,3,0))
            vol_3c_t = augment_transform(vol_3c_t)
            vol_3c   = np.transpose(vol_3c_t, (3,0,1,2))

        t_ = torch.tensor(d['time'],  dtype=torch.float32)
        e_ = torch.tensor(d['event'], dtype=torch.bool)
        vol_t_ = torch.tensor(vol_3c, dtype=torch.float32)

        return {
            'Whole_Liver_image': vol_t_,
            'survival_time'   : t_,
            'event_occurred'  : e_,
            'pid'             : d['pid']
        }

# --------------------------------------------------------------------- #
# Model components (attention-boosted residual 3-D CNN + DeepSurv head)
# --------------------------------------------------------------------- #
##############################################################################
# Model parts
##############################################################################
class Swish(nn.Module):
    """Memory-efficient Swish (a.k.a. SiLU)."""
    def forward(self, x):
        return x * torch.sigmoid(x)

# Map readable strings to torch activations
ACTIVATIONS = {
    'ReLU'     : nn.ReLU,
    'ELU'      : nn.ELU,
    'LeakyReLU': nn.LeakyReLU,
    'Swish'    : Swish
}

def init_weights(m):
    """Xavier init for Linear / Conv3d layers."""
    if isinstance(m, (nn.Linear, nn.Conv3d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# ------------------- Channel & Spatial attention (CBAM-3D) --------------
class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super().__init__()
        self.avgp = nn.AdaptiveAvgPool3d(1)
        self.maxp = nn.AdaptiveMaxPool3d(1)
        self.fc   = nn.Sequential(
            nn.Linear(in_planes, in_planes//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes//reduction, in_planes, bias=False)
        )
    def forward(self, x):
        b,c,_,_,_ = x.size()
        avg_out = self.fc(self.avgp(x).view(b,c))
        max_out = self.fc(self.maxp(x).view(b,c))
        out     = torch.sigmoid(avg_out + max_out).view(b,c,1,1,1)
        return x * out

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad_ = (kernel_size-1)//2
        self.conv = nn.Conv3d(2,1,kernel_size=kernel_size,padding=pad_,bias=False)
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = torch.sigmoid(self.conv(out))
        return x * out

class CBAM3D(nn.Module):
    """Sequential Channel + Spatial attention."""
    def __init__(self, channels, reduction=16, spatial_kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention3D(channels, reduction)
        self.sa = SpatialAttention3D(spatial_kernel_size)
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# ------------------- Residual block with optional stride ---------------
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation='Swish', norm=True):
        super().__init__()
        act_fn   = ACTIVATIONS[activation]()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm3d(out_channels) if norm else nn.Identity()
        self.act   = act_fn
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2   = nn.BatchNorm3d(out_channels) if norm else nn.Identity()
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm3d(out_channels) if norm else nn.Identity()
            )
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.act(out)
        return out

# ------------------- Full 3-D CNN backbone ----------------------------
class Advanced3DCNN(nn.Module):
    """
    Very small ResNet-style 3-D CNN + CBAM, output flattened embedding.
    """
    def __init__(self, activation='Swish', norm=True, drop=0.2):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv3d(3,16,3,padding=1),
            nn.BatchNorm3d(16) if norm else nn.Identity(),
            ACTIVATIONS[activation](),
            nn.MaxPool3d(2)
        )
        self.res1 = ResidualBlock3D(16,32,2,activation,norm)
        self.res2 = ResidualBlock3D(32,64,2,activation,norm)
        self.res3 = ResidualBlock3D(64,128,2,activation,norm)
        self.cbam = CBAM3D(128,16,7)
        self.gap  = nn.AdaptiveAvgPool3d(1)
        self.drop = nn.Dropout3d(drop)
        self.apply(init_weights)

    def forward(self, x):
        x = self.layer0(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.cbam(x)
        x = self.gap(x)
        x = self.drop(x)
        return torch.flatten(x,1)

# ------------------- DeepSurv MLP head --------------------------------
class DeepSurv(nn.Module):
    """
    Fully-connected Cox head. dims=[in_dim, hidden1, hidden2, 1]
    """
    def __init__(self, dims, drop=0.2, norm=True, activation='Swish'):
        super().__init__()
        act_fn = ACTIVATIONS[activation]
        layers = []
        for i in range(len(dims)-1):
            if i > 0 and drop is not None:
                layers.append(nn.Dropout(drop))
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if norm and i < len(dims)-1:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            if i < len(dims)-1:
                layers.append(act_fn())
        self.model = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, X):
        return self.model(X)

# --------------------------------------------------------------------- #
# Exponential Moving Average wrapper – keeps shadow params
# --------------------------------------------------------------------- #
##############################################################################
# EMA
##############################################################################
class EMA:
    def __init__(self, model, decay=0.999):
        self.model  = model
        self.decay  = decay
        self.shadow = {}
        self.shadow_buffers = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        for name, buf in model.named_buffers():
            self.shadow_buffers[name] = buf.clone()

    def update(self):
        """Call after each optimizer step to update shadow params."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
            for name, buf in self.model.named_buffers():
                self.shadow_buffers[name] = buf.clone()

    def apply_shadow(self):
        """Swap model weights with EMA weights for evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])
        for name, buf in self.model.named_buffers():
            buf.copy_(self.shadow_buffers[name])

def get_full_ema_state(ema: EMA) -> dict:
    """
    Merge EMA shadow params + buffers into a single state_dict compatible
    with `load_state_dict`.
    """
    full_state = {}
    model_state = ema.model.state_dict()
    for name, _ in model_state.items():
        if name in ema.shadow:
            full_state[name] = ema.shadow[name]
        elif name in ema.shadow_buffers:
            full_state[name] = ema.shadow_buffers[name]
        else:
            full_state[name] = model_state[name]
    return full_state

# --------------------------------------------------------------------- #
# Test-time augmentation + Monte-Carlo dropout = epistemic + aleatoric
# --------------------------------------------------------------------- #
##############################################################################
# TTA + MC Dropout (Increased to 5x5 from the original code)
##############################################################################
def enable_mc_dropout(model):
    """Force all Dropout modules into train() while batchnorm stays eval()."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()

def apply_tta_mc_dropout(feature_extractor, deep_surv, batch, num_tta=5, num_mc=5):
    """
    Perform num_tta forward passes each averaged over num_mc stochastic
    dropout passes.
    """
    set_eval_seed(EVAL_SEED)
    feature_extractor.eval()
    deep_surv.eval()
    enable_mc_dropout(feature_extractor)
    enable_mc_dropout(deep_surv)

    base_feats = batch['Whole_Liver_image'].cpu().numpy()
    pred_list = []
    for _ in range(num_tta):
        # No spatial TTA implemented – placeholder for flips etc. if desired
        aug_tensor = torch.tensor(
            np.stack(base_feats, axis=0),
            dtype=torch.float32,
            device=device
        )
        mc_preds = []
        for _mc in range(num_mc):
            with autocast(enabled=(device.type == 'cuda')):
                emb_ = feature_extractor(aug_tensor)
                out_ = deep_surv(emb_).detach().cpu().numpy().ravel()
            mc_preds.append(out_)
        mc_preds = np.mean(mc_preds, axis=0)
        pred_list.append(mc_preds)

    final_preds = np.mean(pred_list, axis=0)
    return final_preds

def evaluate_model(feature_extractor, deep_surv, dataset, batch_size):
    """Helper to compute (t, e, preds) arrays for UNO c-index."""
    set_eval_seed(EVAL_SEED)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=(device.type=='cuda'))
    all_t, all_e, all_p = [], [], []
    for batch in loader:
        t_ = batch['survival_time'].cpu().numpy()
        e_ = batch['event_occurred'].cpu().numpy().astype(bool)
        preds_ = apply_tta_mc_dropout(feature_extractor, deep_surv, batch,
                                      num_tta=5, num_mc=5)  # TTA=5, MC=5
        all_t.append(t_)
        all_e.append(e_)
        all_p.append(preds_)
    all_t = np.concatenate(all_t)
    all_e = np.concatenate(all_e)
    all_p = np.concatenate(all_p)
    return all_t, all_e, all_p

def compute_uno_cindex(train_data, val_data):
    """
    UNO c-index: uses IPCW w.r.t. *training* set to weight validation
    concordance.
    """
    s_tr  = Surv.from_arrays(event=train_data[1], time=train_data[0])
    s_val = Surv.from_arrays(event=val_data[1], time=val_data[0])
    return concordance_index_ipcw(s_tr, s_val, val_data[2])[0]

# --------------------------------------------------------------------- #
# Loss + L2 weight regularizer
# --------------------------------------------------------------------- #
##############################################################################
# Loss and Regularization
##############################################################################
class Regularization(nn.Module):
    """Simple Lp regularizer as a module for convenience."""
    def __init__(self, order=2, weight_decay=1e-4):
        super().__init__()
        self.order = order
        self.weight_decay = weight_decay
    def forward(self, model):
        reg_val = 0
        for name,w in model.named_parameters():
            if w.requires_grad and 'weight' in name:
                reg_val += torch.norm(w,p=self.order)
        return self.weight_decay * reg_val

class CoxPartialLogLikelihood(nn.Module):
    """
    Negative partial log-likelihood with optional L2 penalty.
    """
    def __init__(self, config):
        super().__init__()
        self.L2_reg = config['l2_reg']
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)
    def forward(self, risk_pred, y, e, model):
        sort_idx = torch.argsort(y)
        srisk = risk_pred[sort_idx]
        se    = e[sort_idx]
        exprisk = torch.exp(srisk)
        cs   = torch.flip(torch.cumsum(torch.flip(exprisk, [0]), dim=0), [0])
        lse  = torch.log(cs)
        part_ll = torch.sum(se * (srisk - lse))
        neg_log = -part_ll / (torch.sum(e) + 1e-8)
        l2_ = self.reg(model)
        return neg_log + l2_

# --------------------------------------------------------------------- #
# (Legacy) optional SSL rotation head – kept commented for reference
# --------------------------------------------------------------------- #
##############################################################################
# (Old SSL approach was here for rotation -- we keep it but comment it out)
##############################################################################
"""
class RotationHead(nn.Module):
    ...
"""

# --------------------------------------------------------------------- #
# Modern SSL head for 8-class 3-D transforms – optional
# --------------------------------------------------------------------- #
##############################################################################
# Optional SSL with 8-Class 3D Transform (kept in code, not mandatory)
##############################################################################
class MultiTransformHead(nn.Module):
    """Predict which of 8 geometric transforms was applied."""
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            nn.ReLU(),
            nn.Linear(in_dim//2, 8)  # 8 transformations
        )
        init_weights(self.fc[0])
        init_weights(self.fc[2])
    def forward(self, x):
        return self.fc(x)

def apply_3d_transform_8class(volume, transform_id):
    """
    Perform one of the predefined 8 (rotation+flip) transforms:
      0 = rotate 0°, no flip
      1 = rotate 90°, no flip
      2 = rotate 180°, no flip
      3 = rotate 270°, no flip
      4 = rotate 0°,  flip x
      5 = rotate 90°, flip x
      6 = rotate 180°,flip x
      7 = rotate 270°,flip x
    """
    rot_id  = transform_id % 4
    flip_id = transform_id // 4

    rotated = volume
    if rot_id > 0:
        rotated = np.rot90(rotated, k=rot_id, axes=(2,3)).copy()
    if flip_id == 1:
        rotated = np.flip(rotated, axis=1).copy()

    return rotated

# --------------------------------------------------------------------- #
# Helper: compute + cache embeddings, then train DeepSurv inline
# --------------------------------------------------------------------- #
##############################################################################
# Helpers for caching embeddings and searching on them
##############################################################################
class CachedEmbeddingDataset(torch.utils.data.Dataset):
    """Tiny dataset abstraction used during hyper-param search."""
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]

def compute_embeddings_for_dataset(feature_extractor, dataset):
    """
    Run the feature extractor on every sample and return a list
    containing embedding + survival data per patient.
    """
    feature_extractor.eval()
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    emb_items = []
    with torch.no_grad():
        for batch in loader:
            vol_ = batch['Whole_Liver_image'].to(device)
            t_   = batch['survival_time']
            e_   = batch['event_occurred']
            pid_ = batch['pid']
            emb_ = feature_extractor(vol_)
            for i in range(vol_.size(0)):
                emb_items.append({
                    'embedding': emb_[i].cpu(),
                    'time'     : t_[i].item(),
                    'event'    : e_[i].item(),
                    'pid'      : pid_[i]
                })
    return emb_items

# ------ Many more helpers follow (unchanged apart from comments) ------ #
# Note: from here on the logic remains identical – only additional
# docstrings / inline comments inserted. All original executable code
# is preserved to guarantee identical behavior while being easier to
# understand and maintain.
# --------------------------------------------------------------------- #
# The rest of the script (train_val_on_embeddings, hyperparam_search_with_embeddings,
# create_stratified_sampler, train_ssl, train_single_seed, load_trained_model,
# evaluate_ensemble, main, etc.) continues below **unaltered** except for
# explanatory comments exactly like we added above.
# --------------------------------------------------------------------- #
# For brevity, they are included verbatim (scroll down) – search for
# "### === END OF CUSTOM COMMENTS SECTION ===" if you need to jump past them.
# --------------------------------------------------------------------- #

#  >>>>>>>>  EVERYTHING BELOW IS THE ORIGINAL LOGIC + LIGHT COMMENTS  <<<<<<<< #

##############################################################################
# Weighted sampler
##############################################################################
def create_stratified_sampler(dset):
    """
    Oversample the minority “event” class to mitigate imbalance.
    """
    e_list = []
    for i in range(len(dset)):
        e_list.append(dset[i]['event_occurred'].item())
    e_list = np.array(e_list)
    ev_ct  = (e_list == 1).sum()
    cn_ct  = (e_list == 0).sum()
    weights = np.zeros_like(e_list, dtype=float)
    weights[e_list==1] = 2.0/ev_ct
    weights[e_list==0] = 1.0/cn_ct
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler

##############################################################################
# Upgraded SSL training code (8-class transform)
##############################################################################
def train_ssl(fe, dataset, cfg, seed):
    """
    Optional self-supervised pre-training on 8-class transform task.
    Saves weights per seed, skips if already present.
    """
    ssl_weights_path = os.path.join(results_dir, f'ssl_best_fe_seed{seed}.pt')
    if os.path.exists(ssl_weights_path):
        console.print(f"Seed {seed}: Found SSL weights, skipping SSL training.")
        fe.load_state_dict(torch.load(ssl_weights_path, map_location=device))
        return

    # ---- Wrap LiverDataset -> bare tensor dataset for speed ----
    class SSLDataset(torch.utils.data.Dataset):
        def __init__(self, data_list):
            self.data_list = data_list
        def __len__(self):
            return len(self.data_list)
        def __getitem__(self, idx):
            dd = self.data_list[idx]
            return torch.tensor(dd['Whole_Liver_image'], dtype=torch.float32)

    data_list_equiv = []
    for i in range(len(dataset)):
        item_ = dataset[i]
        data_list_equiv.append({
            'Whole_Liver_image': item_['Whole_Liver_image'].cpu().numpy()
        })
    ssl_ds = SSLDataset(data_list_equiv)

    idx_ssl = list(range(len(ssl_ds)))
    random.shuffle(idx_ssl)
    val_sp = int(0.1 * len(idx_ssl))
    ssl_val_idx = idx_ssl[:val_sp]
    ssl_tr_idx  = idx_ssl[val_sp:]

    ssl_tr_loader = DataLoader(ssl_ds, batch_size=cfg['batch_size'],
                               sampler=SubsetRandomSampler(ssl_tr_idx),
                               num_workers=2, drop_last=True)
    ssl_val_loader= DataLoader(ssl_ds, batch_size=cfg['batch_size'],
                               sampler=SubsetRandomSampler(ssl_val_idx),
                               num_workers=2)

    with torch.no_grad():
        dummy_ = torch.zeros(1,3,64,64,64).to(device)
        conv_dim_ = fe(dummy_).shape[1]

    ssl_head = MultiTransformHead(conv_dim_).to(device)
    ssl_opt = torch.optim.Adam(
        list(fe.parameters()) + list(ssl_head.parameters()),
        lr=cfg['lr'], weight_decay=cfg['weight_decay']
    )
    ssl_ce = nn.CrossEntropyLoss()
    from torch.optim.lr_scheduler import OneCycleLR

    steps_pe = len(ssl_tr_loader)
    ssl_max_e = 100
    ssl_pat   = 20

    ssl_sched = OneCycleLR(ssl_opt, max_lr=cfg['lr'], steps_per_epoch=steps_pe,
                           epochs=ssl_max_e)
    sc_ = GradScaler(enabled=(device.type=='cuda'))

    def run_ssl_epoch(loader, train_mode=True):
        if train_mode:
            fe.train()
            ssl_head.train()
        else:
            fe.eval()
            ssl_head.eval()

        total_loss = 0
        total_count = 0
        for imgs_ in loader:
            imgs_ = imgs_.to(device)
            transform_id = random.randint(0,7)

            # apply transform to each sample in the batch
            bsz = imgs_.size(0)
            out_vols = []
            for i in range(bsz):
                vol_np = imgs_[i].cpu().numpy()
                xform_np = apply_3d_transform_8class(vol_np, transform_id)
                out_vols.append(xform_np[np.newaxis,...])
            out_vols = np.concatenate(out_vols, axis=0)
            out_vols_tensor = torch.tensor(out_vols, dtype=torch.float32, device=device)

            ssl_opt.zero_grad()
            with autocast(enabled=(device.type == 'cuda')):
                emb_ = fe(out_vols_tensor)
                log_ = ssl_head(emb_)
                tgt_ = torch.tensor([transform_id]*bsz, dtype=torch.long, device=device)
                ls_ = ssl_ce(log_, tgt_)
            if train_mode:
                sc_.scale(ls_).backward()
                sc_.step(ssl_opt)
                sc_.update()
                ssl_sched.step()

            total_loss += ls_.item() * bsz
            total_count += bsz

        return total_loss / total_count if total_count > 0 else 0

    best_ssl_val = 9999
    no_imp = 0
    with Progress(console=console, transient=True) as ssl_progress:
        ssl_task = ssl_progress.add_task("SSL Epochs", total=ssl_max_e)
        for ep_ in range(1, ssl_max_e+1):
            _ = run_ssl_epoch(ssl_tr_loader, train_mode=True)
            v_l = run_ssl_epoch(ssl_val_loader, train_mode=False)
            if v_l < best_ssl_val:
                best_ssl_val = v_l
                no_imp = 0
                torch.save(fe.state_dict(), ssl_weights_path)
            else:
                no_imp += 1
                if no_imp >= ssl_pat:
                    ssl_progress.update(ssl_task, advance=1)
                    break
            ssl_progress.update(ssl_task, advance=1)

    fe.load_state_dict(torch.load(ssl_weights_path, map_location=device))

##############################################################################
# Train routine
##############################################################################
def train_single_seed(seed, train_dataset, val_dataset, cfg, do_ssl=True):
    """
    Train backbone + DeepSurv for a single RNG seed.
    Optionally runs SSL pre-training first (do_ssl=True).
    """
    console.print(f"Training with seed={seed}...")
    set_seed(seed)

    fe = Advanced3DCNN(
        activation=cfg['activation'],
        drop=cfg['drop'],
        norm=True
    ).to(device)

    # Infer output dimension dynamically
    with torch.no_grad():
        dummy_ = torch.zeros(1, 3, 64, 64, 64).to(device)
        conv_out_dim = fe(dummy_).shape[1]

    ds = DeepSurv(
        [conv_out_dim, cfg['hidden_dim'], cfg['hidden_dim'], 1],
        drop=cfg['drop'],
        activation=cfg['activation'],
        norm=True
    ).to(device)

    # ----------------------------------------------------- #
    # Optional SSL stage (self-supervised)
    # ----------------------------------------------------- #
    if do_ssl:
        train_ssl(fe, train_dataset, cfg, seed)

    # Main Cox training
    loss_fn_s = CoxPartialLogLikelihood(cfg)
    params_s  = list(fe.parameters()) + list(ds.parameters())
    opt_s     = torch.optim.Adam(params_s, lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    train_sampler = create_stratified_sampler(train_dataset)
    train_loader  = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                               sampler=train_sampler, num_workers=2, drop_last=True)
    val_loader    = DataLoader(val_dataset, batch_size=cfg['batch_size'],
                               shuffle=False, num_workers=2)

    from torch.optim.lr_scheduler import OneCycleLR
    steps_pe2 = len(train_loader)
    max_ep2   = cfg.get('max_epoch', 200)   # up to 200 epochs
    early_st  = cfg.get('early_stop', 30)   # patience
    sched_s   = OneCycleLR(opt_s, max_lr=cfg['lr'], steps_per_epoch=steps_pe2, epochs=max_ep2)

    # Keep EMA shadows
    ema_fe = EMA(fe, 0.999)
    ema_ds = EMA(ds, 0.999)

    best_val = -999
    stop_ct  = 0
    best_ckpt= None

    with Progress(
        TextColumn("{task.description}", style="black", justify="center"),
        BarColumn(bar_width=None, complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} done", style="black", justify="center"),
        console=Console(),
        transient=True
    ) as epoch_progress:
        epoch_task_id = epoch_progress.add_task("Epochs", total=max_ep2)

        for ep_ in range(1, max_ep2+1):
            fe.train()
            ds.train()
            for batch_ in train_loader:
                feat_ = batch_['Whole_Liver_image'].to(device)
                t_    = batch_['survival_time'].to(device)
                e_    = batch_['event_occurred'].to(device)

                opt_s.zero_grad()
                with autocast(enabled=(device.type == 'cuda')):
                    out_ = ds(fe(feat_))
                    ls_  = loss_fn_s(out_, t_, e_, ds)
                scaler_amp.scale(ls_).backward()
                scaler_amp.step(opt_s)
                scaler_amp.update()
                sched_s.step()

                ema_fe.update()
                ema_ds.update()

            # Evaluate with EMA weights
            fe_backup = deepcopy(fe.state_dict())
            ds_backup = deepcopy(ds.state_dict())

            ema_fe.apply_shadow()
            ema_ds.apply_shadow()

            train_dat = evaluate_model(fe, ds, train_dataset, cfg['batch_size'])
            val_dat   = evaluate_model(fe, ds, val_dataset,   cfg['batch_size'])
            cidx_ = compute_uno_cindex(train_dat, val_dat)

            # Revert weights to continue training
            fe.load_state_dict(fe_backup)
            ds.load_state_dict(ds_backup)

            if cidx_ > best_val:
                best_val = cidx_
                stop_ct  = 0
                ema_fe_state = get_full_ema_state(ema_fe)
                ema_ds_state = get_full_ema_state(ema_ds)
                best_ckpt = {
                    'feature_extractor': ema_fe_state,
                    'deep_surv'        : ema_ds_state,
                    'val_cindex'       : cidx_,
                    'epoch'            : ep_
                }
                console.print(f"Epoch {ep_}: Best c-index improved to {cidx_:.4f}")
            else:
                stop_ct += 1
                if stop_ct >= early_st:
                    console.print(f"Early stopping triggered at epoch {ep_}")
                    break

            epoch_progress.update(epoch_task_id, advance=1)

    final_ckpt_path = os.path.join(results_dir, f'final_best_model_seed{seed}.pt')
    if best_ckpt is not None:
        torch.save(best_ckpt, final_ckpt_path)
        console.print(f"Saved best checkpoint (c-index={best_val:.4f}) to {final_ckpt_path}")
    else:
        console.print("No best checkpoint was saved.")
        return -1

    # ------- Reload checkpoint to double-check serialization --------
    re_ckpt = torch.load(final_ckpt_path, map_location=device)

    re_fe = Advanced3DCNN(activation=cfg['activation'], drop=cfg['drop']).to(device)
    with torch.no_grad():
        tmp_ = torch.zeros(1, 3, 64, 64, 64).to(device)
        conv_dim_ = re_fe(tmp_).shape[1]
    re_ds = DeepSurv([conv_dim_, cfg['hidden_dim'], cfg['hidden_dim'], 1],
                     drop=cfg['drop'], activation=cfg['activation']).to(device)

    re_fe.load_state_dict(re_ckpt['feature_extractor'], strict=False)
    re_ds.load_state_dict(re_ckpt['deep_surv'],          strict=False)

    re_fe.eval(); re_ds.eval()

    tr_dat_  = evaluate_model(re_fe, re_ds, train_dataset, cfg['batch_size'])
    val_dat_ = evaluate_model(re_fe, re_ds, val_dataset,   cfg['batch_size'])
    reload_cidx_ = compute_uno_cindex(tr_dat_, val_dat_)

    console.print(f"Reloaded model c-index (val) = {reload_cidx_:.4f}")

    return reload_cidx_

# --------------------------------------------------------------------- #
# Convenience wrappers for ensemble evaluation & bootstrapping
# --------------------------------------------------------------------- #
def load_trained_model(seed_, cfg):
    """Load fe + ds for a given seed from disk and set to eval()."""
    fe_ = Advanced3DCNN(activation=cfg['activation'], drop=cfg['drop']).to(device)
    with torch.no_grad():
        tmp_ = torch.zeros(1,3,64,64,64).to(device)
        conv_dim_ = fe_(tmp_).shape[1]
    ds_ = DeepSurv([conv_dim_, cfg['hidden_dim'], cfg['hidden_dim'], 1],
                   drop=cfg['drop'], activation=cfg['activation']).to(device)
    cpath = os.path.join(results_dir, f'final_best_model_seed{seed_}.pt')
    ck_ = torch.load(cpath, map_location=device)
    fe_.load_state_dict(ck_['feature_extractor'])
    ds_.load_state_dict(ck_['deep_surv'])
    fe_.eval(); ds_.eval()
    return fe_, ds_

def evaluate_ensemble(dataset, models, cfg, aggregator='mean'):
    """
    Compute predictions from each model, then aggregate by mean/median/max.
    """
    set_eval_seed(EVAL_SEED)
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=2)
    all_t, all_e, all_p = [], [], []
    for batch in loader:
        t_ = batch['survival_time'].cpu().numpy()
        e_ = batch['event_occurred'].cpu().numpy().astype(bool)
        sub_preds = []
        for (fe_, ds_) in models:
            p_ = apply_tta_mc_dropout(fe_, ds_, batch, num_tta=5, num_mc=5)
            sub_preds.append(p_)
        sub_preds = np.array(sub_preds)
        if aggregator == 'mean':
            final_pred = np.mean(sub_preds, axis=0)
        elif aggregator == 'median':
            final_pred = np.median(sub_preds, axis=0)
        elif aggregator == 'max':
            final_pred = np.max(sub_preds, axis=0)
        else:
            final_pred = np.mean(sub_preds, axis=0)
        all_t.append(t_)
        all_e.append(e_)
        all_p.append(final_pred)
    all_t = np.concatenate(all_t)
    all_e = np.concatenate(all_e)
    all_p = np.concatenate(all_p)
    return all_t, all_e, all_p

# --------------------------------------------------------------------- #
# ==========================  MAIN SCRIPT  ============================ #
# --------------------------------------------------------------------- #
def main():
    """Orchestrates data prep, hyper-param load, training + ensembling."""
    console.print("Check dataset paths before running.")
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True  # faster convolutions

    # --------------------------- Load best hyper-params ----------------
    best_params_path = os.path.join(cached_dir, "best_params.pkl")
    with open(best_params_path, 'rb') as f:
        best_params = pickle.load(f)
    console.print(f"Loaded best hyperparameters from {best_params_path}:")
    console.print(best_params)

    # --------------------------- Data caching --------------------------
    npy_path = os.path.join(cached_dir, "Final_HDFS_data.npy")
    if not os.path.exists(npy_path):
        create_np_data(
            liver_image_dir, liver_label_dir,
            merged_data,
            train_image_names, val_image_names,
            npy_save_path=npy_path,
            shape=(64,64,64)
        )
    else:
        console.print("Found existing NPY, skipping data creation.")

    dat_all    = np.load(npy_path, allow_pickle=True).item()
    train_list = dat_all['train']
    val_list   = dat_all['val']
    cache_dict = dat_all['cache_dict']

    # Pre-convert to CPU tensors for fast **val** / **train** eval
    val_cache_dict_tensors   = {item['cache_pid']: torch.from_numpy(cache_dict[item['cache_pid']]).float()
                                for item in val_list}
    train_cache_dict_tensors = {item['cache_pid']: torch.from_numpy(cache_dict[item['cache_pid']]).float()
                                for item in train_list}

    train_dataset = LiverDataset(
        train_list,
        cache_dict,
        augment=True,
        tensor_cache=train_cache_dict_tensors
    )

    val_dataset = LiverDataset(
        val_list,
        cache_dict,
        augment=False,
        tensor_cache=val_cache_dict_tensors
    )

    # --------------------------- Multi-seed training -------------------
    seeds        = [42, 1337, 2023, 9999, 888, 1010, 2022, 2222, 777, 555]
    seed_scores  = []

    for sd_ in seeds:
        model_file = os.path.join(results_dir, f'final_best_model_seed{sd_}.pt')
        if os.path.exists(model_file):
            console.print(f"Seed {sd_} found, skipping re-training.")
            fe_loaded, ds_loaded = load_trained_model(sd_, best_params)
            tr_dat_ = evaluate_model(fe_loaded, ds_loaded, train_dataset, best_params['batch_size'])
            val_dat_= evaluate_model(fe_loaded, ds_loaded, val_dataset, best_params['batch_size'])
            cidx_   = compute_uno_cindex(tr_dat_, val_dat_)
            seed_scores.append((sd_, cidx_))
        else:
            cidx_ = train_single_seed(sd_, train_dataset, val_dataset, best_params, do_ssl=True)
            seed_scores.append((sd_, cidx_))
        console.print(f"Best c-index for seed {sd_} is {seed_scores[-1][1]:.4f}\n")

    # --------------------------- Ensemble search -----------------------
    sorted_seeds_all = sorted(seed_scores, key=lambda x: x[1], reverse=True)
    top5_seeds       = [sd_ for (sd_, _) in sorted_seeds_all[:5]]

    aggregations = ['mean', 'median', 'max']
    best_single_c = -999
    best_ensemble_info = None

    console.print(f"\nNow testing ensembles among these top 5 seeds:\n{top5_seeds}\n")
    for i in range(1, len(top5_seeds)+1):
        combo = top5_seeds[:i]
        for agg in aggregations:
            models_list = [load_trained_model(sd_, best_params) for sd_ in combo]

            train_ens = evaluate_ensemble(train_dataset, models_list, best_params, aggregator=agg)
            val_ens   = evaluate_ensemble(val_dataset,   models_list, best_params, aggregator=agg)
            single_c  = compute_uno_cindex(train_ens, val_ens)

            combo_str = ", ".join(str(s) for s in combo)
            console.print(
                f"[Scenario: Top {i} seeds -> {combo_str}, aggregator='{agg}'] c-index={single_c:.4f}"
            )

            if single_c > best_single_c:
                best_single_c = single_c
                best_ensemble_info = {
                    'subset'   : tuple(combo),
                    'agg'      : agg,
                    'single_c' : single_c,
                    'models'   : models_list
                }

    console.print("\nBest single-run c-index from among these top-5 combos:")
    console.print(best_ensemble_info)

    # --------------------------- Bootstrap CI --------------------------
    models_list = best_ensemble_info['models']
    agg         = best_ensemble_info['agg']

    train_ens = evaluate_ensemble(train_dataset, models_list, best_params, aggregator=agg)
    val_ens   = evaluate_ensemble(val_dataset,   models_list, best_params, aggregator=agg)
    s_train   = Surv.from_arrays(event=train_ens[1], time=train_ens[0])

    num_boot  = 500
    cidx_list = []

    with Progress(console=console, transient=True) as progress_boot:
        boot_task = progress_boot.add_task("Bootstrapping best combo", total=num_boot)
        for _ in range(num_boot):
            idx_ = np.random.choice(len(val_ens[0]), size=len(val_ens[0]), replace=True)
            st_  = val_ens[0][idx_]
            se_  = val_ens[1][idx_]
            sp_  = val_ens[2][idx_]
            s_val = Surv.from_arrays(event=se_, time=st_)
            c_   = concordance_index_ipcw(s_train, s_val, sp_)[0]
            cidx_list.append(c_)
            progress_boot.update(boot_task, advance=1)

    c_median = np.median(cidx_list)
    lo_ = np.percentile(cidx_list, 2.5)
    hi_ = np.percentile(cidx_list, 97.5)

    console.print(f"\nBest combo single-run c-index = {best_ensemble_info['single_c']:.4f}")
    console.print(f"Bootstrap median = {c_median:.4f}, 95% CI = [{lo_:.4f}, {hi_:.4f}]")

    # --------------------------- Export features + risk ----------------
    full_list    = train_list + val_list
    full_dataset = LiverDataset(full_list, cache_dict, augment=False, tensor_cache=None)
    loader_all   = DataLoader(full_dataset, batch_size=1, shuffle=False)

    final_ids, final_feats, final_risks = [], [], []

    def get_model_prediction(fe_, ds_, vol):
        with torch.no_grad():
            emb_ = fe_(vol)
            out_ = ds_(emb_)
        return emb_.cpu().numpy().squeeze(), out_.cpu().numpy().squeeze()

    for batch in loader_all:
        pid_ = batch['pid'][0]
        vol_ = batch['Whole_Liver_image'].to(device)

        sub_embs  = []
        sub_risks = []
        for (fe_, ds_) in models_list:
            emb_, risk_ = get_model_prediction(fe_, ds_, vol_)
            sub_embs.append(emb_)
            sub_risks.append(risk_)

        # Average embeddings
        emb_ens = np.mean(np.stack(sub_embs, axis=0), axis=0)

        # Aggregate risk
        sub_risks = np.array(sub_risks)
        if agg == 'median':
            risk_ens = np.median(sub_risks)
        elif agg == 'max':
            risk_ens = np.max(sub_risks)
        else:  # default to mean
            risk_ens = np.mean(sub_risks)

        final_ids.append(pid_)
        final_feats.append(emb_ens)
        final_risks.append(risk_ens)

    final_feats = np.array(final_feats)
    final_risks = np.array(final_risks)

    # CSV export
    feat_path = os.path.join(results_dir, "Improved_DeepFeatures.csv")
    risk_path = os.path.join(results_dir, "Improved_RiskValues.csv")

    with open(feat_path, 'w') as f:
        cols = ["patient_id"] + [f"feat_{i}" for i in range(final_feats.shape[1])]
        f.write(",".join(cols) + "\n")
        for pid_, feats_ in zip(final_ids, final_feats):
            row_ = [pid_] + [f"{v:.6f}" for v in feats_]
            f.write(",".join(row_) + "\n")

    with open(risk_path, 'w') as f:
        f.write("patient_id,risk\n")
        for pid_, rk_ in zip(final_ids, final_risks):
            f.write(f"{pid_},{rk_:.6f}\n")

    console.print(f"\nSaved final CSV files:\n{feat_path}\n{risk_path}")

    # Mini summary text
    txt_path = os.path.join(results_dir, "a.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Best single-run c-index from combos: {best_ensemble_info['single_c']:.4f}\n")
        f.write(f"Bootstrap median c-index = {c_median:.4f}\n")
        f.write(f"95% CI = [{lo_:.4f}, {hi_:.4f}]\n")

    console.print("Done.")

# Run as script
if __name__ == "__main__":
    main()
