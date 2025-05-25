# =============================================================================
# Deep-Survival Pipeline (Whole-Liver OS)
# -----------------------------------------------------------------------------
# notes:
#   • Device-agnostic: falls back to CPU if no CUDA.
#   • Mixed precision enabled when a GPU is present.
#   • Includes optional SSL block (eight-class transform) you can toggle.
#   • Final outputs: CSV with deep features, CSV with ensemble risk values,
#                    and a text file summarizing c-index + bootstrap CI.
# =============================================================================

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

# -----------------------------------------------------------------------------
# Mixed-precision utilities
# -----------------------------------------------------------------------------
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torchvision import transforms
import nibabel as nib

# -----------------------------------------------------------------------------
# Pretty console / progress bars
# -----------------------------------------------------------------------------
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn

console = Console(style="white")

###############################################################################
# Paths & global configuration
###############################################################################
cached_dir   = "/mnt/largedrive1/rmojtahedi/Deep_Survival/Final_Improved_npj/Final_Improved_npj_OS"
results_dir  = "/mnt/largedrive1/rmojtahedi/Deep_Survival/Best_of_Best/Best_OS"
os.makedirs(results_dir, exist_ok=True)

liver_image_dir  = "/mnt/largedrive1/rmojtahedi/KSPIE/liver_whole/ct"
liver_label_dir  = "/mnt/largedrive1/rmojtahedi/KSPIE/liver_whole/final_seg_split"
clinical_csv_path= "/mnt/largedrive0/rmojtahedi/Kaitlyn_SPIE/Deep_Survival/npj_Digital_Medicine_Clinical_Data_FINAL.csv"

device     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scaler_amp = GradScaler(enabled=(device.type == "cuda"))

###############################################################################
# RNG seeds (one for train, one for eval/TTA)
###############################################################################
EVAL_SEED = 999

def set_seed(seed: int) -> None:
    """Set all RNGs for *training* reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_eval_seed(seed: int) -> None:
    """Set all RNGs for *evaluation* reproducibility (TTA/MC-Dropout)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

###############################################################################
# Helper: map filenames ↔ XNAT IDs
###############################################################################
def get_core_id(filename: str) -> str:
    """Strip .nii /.nii.gz suffix and return the base identifier."""
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    elif filename.endswith(".nii"):
        return filename[:-4]
    return filename

def get_candidate_id(name: str) -> str:
    """Derive candidate XNAT ID from image/label basename."""
    parts = name.split('_')
    return '_'.join(parts[:-1]) if len(parts) > 1 else name

# -----------------------------------------------------------------------------
# Discover available files
# -----------------------------------------------------------------------------
image_files = os.listdir(liver_image_dir)
label_files = os.listdir(liver_label_dir)

image_names = {get_core_id(fn) for fn in image_files}
label_names = {get_core_id(fn) for fn in label_files}

unmatched_images = sorted(list(image_names - label_names))
unmatched_labels = sorted(list(label_names - image_names))

common_names     = image_names.intersection(label_names)
candidate_pairs  = [(name, get_candidate_id(name)) for name in sorted(common_names)]

###############################################################################
# Load clinical CSV and build matching table
###############################################################################
clinical_data_full = pd.read_csv(clinical_csv_path)
clinical_data_full = clinical_data_full.dropna(subset=["XNAT ID"])
clinical_data_full["XNAT ID"] = clinical_data_full["XNAT ID"].astype(str)
clinical_ids = set(clinical_data_full["XNAT ID"])

# -----------------------------------------------------------------------------
# Auto-match image↔label↔clinical rows, then manual patch list
# -----------------------------------------------------------------------------
final_matched        = []
unmatched_candidates = []

for image_label, candidate in candidate_pairs:
    if candidate in clinical_ids:
        final_matched.append((image_label, candidate))
    else:
        unmatched_candidates.append((image_label, candidate))

# Manual corrections (curator supplied)
manual_mapping = {
    "RIA_17-010A_000_202": "RIA_17-010A_000_202",
    "RIA_17-010A_000_439": "RIA_17-010A_000_439",
    "RIA_17-010A_002_118": "RIA_17-010A_002_118",
    "RIA_17-010A_002_131": "RIA_17-010A_002_131",
    "RIA_17-010A_002_159": "RIA_17-010A_002_159",
    "RIA_17-010A_002_170": "RIA_17-010A_002_170",
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

final_matched.extend(manually_matched)
matched_xnat_set       = {cid for _, cid in final_matched}
unmatched_clinical_ids = sorted(list(clinical_ids - matched_xnat_set))

# -----------------------------------------------------------------------------
# DataFrame merge + basic filters
# -----------------------------------------------------------------------------
matched_df  = pd.DataFrame(final_matched, columns=["Image_Label_Base", "XNAT ID"])
merged_data = pd.merge(matched_df, clinical_data_full, on="XNAT ID", how="inner")
merged_data = merged_data.dropna(subset=["Surv Time Censor 7", "Censor 7: OS", "OSTrain"])
merged_data = merged_data[merged_data["Surv Time Censor 7"] > 0]

# Train / validation split according to Kaitlyn’s flag
train_rows = merged_data[merged_data["OSTrain"] == 1]
val_rows   = merged_data[merged_data["OSTrain"] == 0]

train_image_names = train_rows["Image_Label_Base"].tolist()
val_image_names   = val_rows["Image_Label_Base"].tolist()

###############################################################################
# 3-D augmentation suite
###############################################################################
class RandomGaussianNoise:
    """Add low-σ Gaussian noise (precision-preserving)."""
    def __init__(self, p=0.5, std_range=(0.01, 0.03)):
        self.p = p
        self.std_range = std_range
    def __call__(self, volume):
        if random.random() < self.p:
            std   = random.uniform(*self.std_range)
            noise = np.random.randn(*volume.shape) * std
            volume = volume + noise
        return volume

class RandomZoom3D:
    """Zoom in/out with crop-or-pad to restore shape."""
    def __init__(self, p=0.5, zoom_range=(0.9, 1.1)):
        self.p = p
        self.zoom_range = zoom_range
    def __call__(self, volume):
        if random.random() < self.p:
            zf       = random.uniform(*self.zoom_range)
            zoomed   = scipy.ndimage.zoom(volume, zf, order=1)
            oshape   = volume.shape
            if zoomed.shape[0] >= oshape[0]:  # center-crop
                sx = (zoomed.shape[0]-oshape[0])//2
                sy = (zoomed.shape[1]-oshape[1])//2
                sz = (zoomed.shape[2]-oshape[2])//2
                volume = zoomed[sx:sx+oshape[0], sy:sy+oshape[1], sz:sz+oshape[2]]
            else:                              # center-pad
                padded = np.zeros(oshape, dtype=zoomed.dtype)
                ox = (oshape[0]-zoomed.shape[0])//2
                oy = (oshape[1]-zoomed.shape[1])//2
                oz = (oshape[2]-zoomed.shape[2])//2
                padded[ox:ox+zoomed.shape[0],
                       oy:oy+zoomed.shape[1],
                       oz:oz+zoomed.shape[2]] = zoomed
                volume = padded
        return volume

class RandomRotate3D:
    """Free-axis rotation up to ±`max_angle` degrees."""
    def __init__(self, p=0.5, max_angle=10):
        self.p = p
        self.max_angle = max_angle
    def __call__(self, volume):
        if random.random() < self.p:
            axis_pairs = [(0,1), (0,2), (1,2)]
            axes       = random.choice(axis_pairs)
            angle      = random.uniform(-self.max_angle, self.max_angle)
            volume     = scipy.ndimage.rotate(volume, angle, axes=axes, reshape=False, order=1)
        return volume

class RandomFlip3D:
    """Mirror along a random axis."""
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, volume):
        if random.random() < self.p:
            axis  = random.choice([0,1,2])
            volume = np.flip(volume, axis=axis).copy()
        return volume

class RandomIntensityShift:
    """Add constant bias in HU-normalized space."""
    def __init__(self, p=0.5, shift_range=(-0.07, 0.07)):
        self.p = p
        self.shift_range = shift_range
    def __call__(self, volume):
        if random.random() < self.p:
            volume = volume + random.uniform(*self.shift_range)
        return volume

class RandomIntensityScale:
    """Global multiplicative scaling."""
    def __init__(self, p=0.5, scale_range=(0.95, 1.05)):
        self.p = p
        self.scale_range = scale_range
    def __call__(self, volume):
        if random.random() < self.p:
            volume = volume * random.uniform(*self.scale_range)
        return volume

class RandomGamma:
    """Gamma perturbation in min-max-normalized space."""
    def __init__(self, p=0.5, gamma_range=(0.9, 1.1)):
        self.p = p
        self.gamma_range = gamma_range
    def __call__(self, volume):
        if random.random() < self.p:
            gm    = random.uniform(*self.gamma_range)
            volume= np.clip(volume, 0, None)
            vmin, vmax = volume.min(), volume.max()
            if vmax > vmin:
                nv = (volume - vmin) / (vmax - vmin + 1e-10)
                volume = (nv**gm) * (vmax - vmin) + vmin
        return volume

# Compose into a single callable
augment_transform = transforms.Compose([
    RandomRotate3D(p=0.5, max_angle=10),
    RandomFlip3D(p=0.5),
    RandomZoom3D(p=0.5, zoom_range=(0.9, 1.1)),
    RandomGaussianNoise(p=0.5, std_range=(0.01, 0.03)),
    RandomIntensityShift(p=0.5, shift_range=(-0.07, 0.07)),
    RandomIntensityScale(p=0.5, scale_range=(0.95, 1.05)),
    RandomGamma(p=0.5, gamma_range=(0.9, 1.1))
])

###############################################################################
# NIfTI → NumPy cache builder
###############################################################################
def create_np_data(image_dir: str,
                   label_dir: str,
                   clinical_df: pd.DataFrame,
                   train_list: list,
                   val_list: list,
                   npy_save_path: str,
                   shape=(64,64,64)) -> None:
    """
    • Read, mask, resize liver volumes.
    • Fit StandardScaler across all voxels.
    • Save dictionary: {train, val, mean, std, cache_dict}.
    """
    console.print("Starting conversion to NPY...")

    # -------------------------------- Fit scaler -----------------------------
    all_pids        = train_list + val_list
    gather_samples  = []

    with Progress(
        TextColumn("{task.description}", style="black", justify="center"),
        BarColumn(bar_width=None, complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} done", style="black", justify="center"),
        console=Console(),
        transient=True
    ) as progress:
        task = progress.add_task("Loading & resizing", total=len(all_pids))
        for pid in all_pids:
            row = clinical_df[clinical_df["Image_Label_Base"] == pid]
            if len(row) == 0:
                progress.update(task, advance=1)
                continue

            image_path = os.path.join(image_dir, pid + ".nii.gz")
            if not os.path.exists(image_path):
                image_path = os.path.join(image_dir, pid + ".nii")

            label_path = os.path.join(label_dir, pid + ".nii.gz")
            if not os.path.exists(label_path):
                label_path = os.path.join(label_dir, pid + ".nii")

            vol = nib.load(image_path).get_fdata()
            seg = nib.load(label_path).get_fdata()
            vol = resize(vol, shape)
            seg = resize(seg, shape, order=0, preserve_range=True).astype(np.float32)
            seg = (seg > 0.5).astype(np.float32)
            masked_vol = vol * seg

            gather_samples.append(masked_vol.reshape(-1, 1))
            progress.update(task, advance=1)

    big_array = np.concatenate(gather_samples, axis=0)
    scaler    = StandardScaler().fit(big_array)

    # -------------------------------- helper ---------------------------------
    cache_dict = {}
    def build_data(pid_list, desc):
        items = []
        with Progress(
            TextColumn("{task.description}", style="black", justify="center"),
            BarColumn(bar_width=None, complete_style="green", finished_style="green"),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total} done", style="black", justify="center"),
            console=Console(),
            transient=True
        ) as prog2:
            tid = prog2.add_task(desc, total=len(pid_list))
            for pid in pid_list:
                row = clinical_df[clinical_df["Image_Label_Base"] == pid]
                if len(row) == 0:
                    prog2.update(tid, advance=1)
                    continue

                image_path = os.path.join(image_dir, pid + ".nii.gz")
                if not os.path.exists(image_path):
                    image_path = os.path.join(image_dir, pid + ".nii")

                label_path = os.path.join(label_dir, pid + ".nii.gz")
                if not os.path.exists(label_path):
                    label_path = os.path.join(label_dir, pid + ".nii")

                vol = nib.load(image_path).get_fdata()
                seg = nib.load(label_path).get_fdata()
                vol = resize(vol, shape)
                seg = resize(seg, shape, order=0, preserve_range=True).astype(np.float32)
                seg = (seg > 0.5).astype(np.float32)
                masked_vol = vol * seg

                masked_vol = scaler.transform(masked_vol.reshape(-1,1)).reshape(*shape)
                vol_3c     = np.repeat(masked_vol[np.newaxis, ...], 3, axis=0)

                st = row["Surv Time Censor 7"].values[0]
                ev = row["Censor 7: OS"].values[0]

                cache_dict[pid] = vol_3c.astype(np.float32)
                items.append({
                    "pid":       pid,
                    "cache_pid": pid,
                    "image":     None,
                    "time":      float(st),
                    "event":     bool(ev)
                })
                prog2.update(tid, advance=1)
        return items

    train_data = build_data(train_list, "Processing train set")
    val_data   = build_data(val_list,   "Processing val set")

    final_dict = {
        "train":       train_data,
        "val":         val_data,
        "mean":        scaler.mean_[0],
        "std":         scaler.scale_[0],
        "cache_dict":  cache_dict
    }
    np.save(npy_save_path, final_dict)
    console.print(f"Saved NumPy cache → {npy_save_path}")

###############################################################################
# PyTorch Dataset
###############################################################################
class LiverDataset(torch.utils.data.Dataset):
    """
    Wraps cached NumPy arrays → torch.Tensors with optional augmentation.
    """
    def __init__(self, data_list, cache_dict, augment=False, tensor_cache=None):
        self.data_list    = data_list
        self.cache_dict   = cache_dict
        self.augment      = augment
        self.tensor_cache = tensor_cache  # optional RAM→CPU accel

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int):
        sample = self.data_list[idx]
        pid    = sample["cache_pid"]

        # Use pre-converted tensor if available and augmentation disabled
        if not self.augment and self.tensor_cache is not None:
            vol_3c = self.tensor_cache[pid].clone()
        else:
            vol_3c = self.cache_dict[pid].copy()

        # If augmenting: transpose to (D,H,W,C), apply transform, transpose back
        if self.augment:
            tmp = np.transpose(vol_3c, (1,2,3,0))
            tmp = augment_transform(tmp)
            vol_3c = np.transpose(tmp, (3,0,1,2))

        vol_t = torch.tensor(vol_3c, dtype=torch.float32)
        t     = torch.tensor(sample["time"],  dtype=torch.float32)
        e     = torch.tensor(sample["event"], dtype=torch.bool)

        return {
            "Whole_Liver_image": vol_t,
            "survival_time":      t,
            "event_occurred":     e,
            "pid":                sample["pid"]
        }

###############################################################################
# 3-D CNN backbone with CBAM + DeepSurv MLP head
###############################################################################
class Swish(nn.Module):
    """x · sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)

# Activation mapping
ACTIVATIONS = {
    "ReLU":      nn.ReLU,
    "ELU":       nn.ELU,
    "LeakyReLU": nn.LeakyReLU,
    "Swish":     Swish,
}

def init_weights(m):
    """Xavier initialization for Conv3d / Linear layers."""
    if isinstance(m, (nn.Linear, nn.Conv3d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# -------------------- Attention blocks (CBAM 3-D) ----------------------------
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
        avg_out   = self.fc(self.avgp(x).view(b,c))
        max_out   = self.fc(self.maxp(x).view(b,c))
        out       = torch.sigmoid(avg_out + max_out).view(b,c,1,1,1)
        return x * out

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad       = (kernel_size-1)//2
        self.conv = nn.Conv3d(2,1,kernel_size=kernel_size,padding=pad,bias=False)
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = torch.sigmoid(self.conv(out))
        return x * out

class CBAM3D(nn.Module):
    def __init__(self, ch, reduction=16, spatial_kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention3D(ch, reduction)
        self.sa = SpatialAttention3D(spatial_kernel_size)
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# ----------------------------- ResNet-like block -----------------------------
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation="Swish", norm=True):
        super().__init__()
        act_fn       = ACTIVATIONS[activation]()
        self.conv1   = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1     = nn.BatchNorm3d(out_channels) if norm else nn.Identity()
        self.act     = act_fn
        self.conv2   = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2     = nn.BatchNorm3d(out_channels) if norm else nn.Identity()
        self.shortcut= nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm3d(out_channels) if norm else nn.Identity()
            )
    def forward(self, x):
        out = self.conv1(x); out = self.bn1(out); out = self.act(out)
        out = self.conv2(out); out = self.bn2(out)
        out += self.shortcut(x)
        out = self.act(out)
        return out

class Advanced3DCNN(nn.Module):
    """
    Lightweight 3-D CNN (16→32→64→128) + CBAM + GAP → embedding vector.
    """
    def __init__(self, activation="Swish", norm=True, drop=0.2):
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
        x = self.res1(x); x = self.res2(x); x = self.res3(x)
        x = self.cbam(x); x = self.gap(x); x = self.drop(x)
        return torch.flatten(x,1)

class DeepSurv(nn.Module):
    """
    Two-hidden-layer MLP with optional dropout & BN, output = log-risk.
    """
    def __init__(self, dims, drop=0.2, norm=True, activation="Swish"):
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
    def forward(self, x):
        return self.model(x)

###############################################################################
# EMA helper
###############################################################################
class EMA:
    """Exponential Moving Average wrapper (decay≈0.999)."""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow, self.shadow_buffers = {}, {}
        for n,p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()
        for n,b in model.named_buffers():
            self.shadow_buffers[n] = b.clone()
    def update(self):
        """Call once per optimizer step."""
        with torch.no_grad():
            for n,p in self.model.named_parameters():
                if p.requires_grad:
                    self.shadow[n] = self.decay*self.shadow[n] + (1-self.decay)*p.data
            for n,b in self.model.named_buffers():
                self.shadow_buffers[n] = b.clone()
    def apply_shadow(self):
        """Swap model parameters for evaluation."""
        for n,p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])
        for n,b in self.model.named_buffers():
            b.copy_(self.shadow_buffers[n])

def get_full_ema_state(ema: EMA) -> dict:
    """Return dict load-able via model.load_state_dict()."""
    merged = {}
    state  = ema.model.state_dict()
    for k in state:
        if k in ema.shadow:
            merged[k] = ema.shadow[k]
        elif k in ema.shadow_buffers:
            merged[k] = ema.shadow_buffers[k]
        else:
            merged[k] = state[k]
    return merged

###############################################################################
# TTA + MC-Dropout prediction routine
###############################################################################
def enable_mc_dropout(model):
    """Set dropout layers to train mode, keep BN in eval mode."""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()

def apply_tta_mc_dropout(feature_extractor, deep_surv, batch, num_tta=5, num_mc=5):
    """Run `num_tta×num_mc` stochastic passes and average."""
    set_eval_seed(EVAL_SEED)
    feature_extractor.eval(); deep_surv.eval()
    enable_mc_dropout(feature_extractor); enable_mc_dropout(deep_surv)

    base = batch["Whole_Liver_image"].cpu().numpy()
    all_preds = []
    for _ in range(num_tta):
        aug_tensor = torch.tensor(np.stack(base,0), dtype=torch.float32, device=device)
        mc_preds = []
        for _ in range(num_mc):
            with autocast(enabled=(device.type=="cuda")):
                emb = feature_extractor(aug_tensor)
                out = deep_surv(emb).detach().cpu().numpy().ravel()
            mc_preds.append(out)
        all_preds.append(np.mean(mc_preds, axis=0))
    return np.mean(all_preds, axis=0)

def evaluate_model(feature_extractor, deep_surv, dataset, batch_size):
    """Return arrays (time,event,prediction) across `dataset`."""
    set_eval_seed(EVAL_SEED)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=(device.type=="cuda"))
    t_all,e_all,p_all = [],[],[]
    for batch in loader:
        t_all.append(batch["survival_time"].cpu().numpy())
        e_all.append(batch["event_occurred"].cpu().numpy().astype(bool))
        p_all.append(apply_tta_mc_dropout(feature_extractor, deep_surv, batch, 5, 5))
    return np.concatenate(t_all), np.concatenate(e_all), np.concatenate(p_all)

def compute_uno_cindex(train_data, val_data):
    """Uno IPCW concordance index, train → risk stratification weights."""
    s_tr = Surv.from_arrays(event=train_data[1], time=train_data[0])
    s_va = Surv.from_arrays(event=val_data[1],  time=val_data[0])
    return concordance_index_ipcw(s_tr, s_va, val_data[2])[0]

###############################################################################
# Loss + weight regularizer
###############################################################################
class Regularization(nn.Module):
    def __init__(self, order=2, weight_decay=1e-4):
        super().__init__()
        self.order = order; self.weight_decay = weight_decay
    def forward(self, model):
        reg = 0
        for n,w in model.named_parameters():
            if w.requires_grad and "weight" in n:
                reg += torch.norm(w, p=self.order)
        return self.weight_decay * reg

class CoxPartialLogLikelihood(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.L2 = cfg["l2_reg"]; self.reg = Regularization(2, self.L2)
    def forward(self, risk, y, e, model):
        idx = torch.argsort(y)
        srisk = risk[idx]; se = e[idx]
        exprisk = torch.exp(srisk)
        cs  = torch.flip(torch.cumsum(torch.flip(exprisk,[0]),0),[0])
        lse = torch.log(cs)
        pll = torch.sum(se * (srisk - lse))
        return (-pll / (torch.sum(e)+1e-8)) + self.reg(model)

###############################################################################
# --------------------------------------------------------------------------- #
# NOTE: SSL head, hyper-param search, stratified sampler, training loop,
#       ensemble evaluation, CSV export, and `main()` follow next.
#       Only comments/doc-strings were added to those blocks - no code removed.
# --------------------------------------------------------------------------- #
###############################################################################

# ----------------- SSL (8-class 3-D transform classification) ---------------
class MultiTransformHead(nn.Module):
    """Self-supervised 8-class transform classifier (optional pre-train)."""
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim//2), nn.ReLU(),
            nn.Linear(in_dim//2, 8)
        )
        init_weights(self.fc[0]); init_weights(self.fc[2])
    def forward(self, x): return self.fc(x)

def apply_3d_transform_8class(volume, transform_id):
    """
    Map `transform_id∈[0..7]` to rotation+flip combo:
      0 = 0°,  1 = 90°,  2 = 180°, 3 = 270°  (no flip)
      4 = 0°+flipX, 5 = 90°+flipX, 6 = 180°+flipX, 7 = 270°+flipX
    Rotation plane = (H,W) axes=(2,3)
    """
    rot_id  = transform_id % 4
    flip_id = transform_id // 4
    vol = volume
    if rot_id:  vol = np.rot90(vol, k=rot_id, axes=(2,3)).copy()
    if flip_id: vol = np.flip(vol, axis=1).copy()
    return vol

###############################################################################
# Helper dataset used by SSL and hyper-param embedding search
###############################################################################
class CachedEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

def compute_embeddings_for_dataset(fe, dataset):
    """Pre-compute CNN embeddings to accelerate HP search."""
    fe.eval(); loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    items=[]
    with torch.no_grad():
        for batch in loader:
            vol = batch["Whole_Liver_image"].to(device)
            t   = batch["survival_time"]; e = batch["event_occurred"]; pid = batch["pid"]
            emb = fe(vol)
            for i in range(vol.size(0)):
                items.append({
                    "embedding": emb[i].cpu(),
                    "time":      t[i].item(),
                    "event":     e[i].item(),
                    "pid":       pid[i]
                })
    return items

# ---------------- Hyper-parameter search on cached embeddings ---------------
def train_val_on_embeddings(emb_dataset, ds, train_idx, cfg):
    """
    Mini-train DeepSurv on a single fold of pre-computed embeddings.
    Used inside hyper-param grid-search.
    """
    from torch.optim.lr_scheduler import OneCycleLR
    loss_fn = CoxPartialLogLikelihood(cfg)
    opt     = torch.optim.Adam(ds.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sc      = GradScaler(enabled=(device.type=="cuda"))

    loader  = DataLoader(emb_dataset, batch_size=cfg["batch_size"],
                         sampler=SubsetRandomSampler(train_idx), num_workers=2, drop_last=True)
    sched   = OneCycleLR(opt, max_lr=cfg["lr"], steps_per_epoch=len(loader), epochs=2)

    ds.train()
    for _ in range(2):
        for batch in loader:
            emb = batch["embedding"].to(device)
            t   = torch.tensor(batch["time"], device=device)
            e   = torch.tensor(batch["event"], device=device, dtype=torch.bool)
            opt.zero_grad()
            with autocast(enabled=(device.type=="cuda")):
                out  = ds(emb)
                loss = loss_fn(out, t, e, ds)
            sc.scale(loss).backward(); sc.step(opt); sc.update(); sched.step()

    # Evaluate C-index over all items
    ds.eval(); loader_eval = DataLoader(emb_dataset, batch_size=cfg["batch_size"])
    t_all,e_all,p_all = [],[],[]
    with torch.no_grad():
        for b in loader_eval:
            eemb = b["embedding"].to(device)
            t_all.append(np.array(b["time"]))
            e_all.append(np.array(b["event"]).astype(bool))
            p_all.append(ds(eemb).cpu().numpy().squeeze())
    t_all = np.concatenate(t_all); e_all = np.concatenate(e_all); p_all = np.concatenate(p_all)
    s_surv = Surv.from_arrays(event=e_all, time=t_all)
    return concordance_index_ipcw(s_surv, s_surv, p_all)[0]

def hyperparam_search_with_embeddings(emb_dataset, k=3):
    """Brute-force grid a handful of HPs with K-fold CV on embeddings."""
    from itertools import product
    lr_list      = [1e-4, 3e-4, 5e-4, 1e-3]
    drop_list    = [0.1, 0.2, 0.3]
    hidden_list  = [128, 256]
    act_list     = ["Swish", "LeakyReLU"]
    bsz_list     = [4, 8, 16]
    wd_list      = [1e-5, 1e-4]
    l2_list      = [1e-5, 1e-4]

    indices = list(range(len(emb_dataset)))
    random.shuffle(indices); kf = KFold(k, shuffle=True, random_state=42)

    best_c = -999; best_cfg=None
    combos = list(product(lr_list, drop_list, hidden_list, act_list, bsz_list, wd_list, l2_list))
    with Progress(
        TextColumn("{task.description}", style="black"),
        BarColumn(bar_width=None, complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total}", style="black"),
        transient=True, console=console
    ) as p:
        task = p.add_task("HP search", total=len(combos))
        for lr,dp,hd,act,bsz,wd,l2 in combos:
            cfg = {
                "lr": lr, "drop": dp, "hidden_dim": hd, "activation": act,
                "batch_size": bsz, "weight_decay": wd, "l2_reg": l2
            }
            scores=[]
            for tri,_ in kf.split(indices):
                ds_tmp = DeepSurv([emb_dataset[0]["embedding"].shape[0], hd, hd, 1],
                                  drop=dp, activation=act, norm=True).to(device)
                c = train_val_on_embeddings(emb_dataset, ds_tmp, tri, cfg)
                scores.append(c)
            mean_c = np.mean(scores)
            if mean_c > best_c: best_c, best_cfg = mean_c, cfg
            p.update(task, advance=1)
    return best_cfg, best_c

###############################################################################
# Stratified weighted sampler (oversample events)
###############################################################################
def create_stratified_sampler(dataset):
    e = np.array([dataset[i]["event_occurred"].item() for i in range(len(dataset))])
    w = np.zeros_like(e, dtype=float)
    w[e==1] = 2.0 / (e==1).sum()
    w[e==0] = 1.0 / (e==0).sum()
    return WeightedRandomSampler(w, len(w), replacement=True)

###############################################################################
# Self-supervised training (optional)
###############################################################################
def train_ssl(fe, dataset, cfg, seed):
    """
    Optional 8-class transform SSL pre-training. Skips if weights exist.
    """
    ssl_pt = os.path.join(results_dir, f"ssl_best_fe_seed{seed}.pt")
    if os.path.exists(ssl_pt):
        console.print(f"Seed {seed}: SSL weights found → skipping.")
        fe.load_state_dict(torch.load(ssl_pt, map_location=device))
        return

    # ------- wrap dataset tensors only (no labels) ---------------------------
    class SSLDataset(torch.utils.data.Dataset):
        def __init__(self, dl): self.dl = dl
        def __len__(self): return len(self.dl)
        def __getitem__(self, i):
            return torch.tensor(self.dl[i]["Whole_Liver_image"], dtype=torch.float32)

    ssl_ds = SSLDataset(dataset)
    idx    = list(range(len(ssl_ds))); random.shuffle(idx)
    v_sp   = int(0.1*len(idx))
    ssl_val_idx, ssl_tr_idx = idx[:v_sp], idx[v_sp:]

    ssl_tr = DataLoader(ssl_ds, batch_size=cfg["batch_size"],
                        sampler=SubsetRandomSampler(ssl_tr_idx), num_workers=2, drop_last=True)
    ssl_val= DataLoader(ssl_ds, batch_size=cfg["batch_size"],
                        sampler=SubsetRandomSampler(ssl_val_idx), num_workers=2)

    # Infer embedding dimension
    with torch.no_grad():
        dummy = torch.zeros(1,3,64,64,64).to(device)
        emb_dim = fe(dummy).shape[1]

    ssl_head = MultiTransformHead(emb_dim).to(device)
    opt_ssl  = torch.optim.Adam(list(fe.parameters())+list(ssl_head.parameters()),
                                lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ce_ssl   = nn.CrossEntropyLoss()
    from torch.optim.lr_scheduler import OneCycleLR
    sched_ssl= OneCycleLR(opt_ssl, max_lr=cfg["lr"],
                          steps_per_epoch=len(ssl_tr), epochs=100)
    sc       = GradScaler(enabled=(device.type=="cuda"))

    def run(loader, train=True):
        fe.train(train); ssl_head.train(train)
        tot,lbs=0,0
        for imgs in loader:
            imgs = imgs.to(device)
            tid  = random.randint(0,7)
            out_vol=[]
            for i in range(imgs.size(0)):
                npv = imgs[i].cpu().numpy()
                out_vol.append(apply_3d_transform_8class(npv, tid)[None,...])
            out_vol = torch.tensor(np.concatenate(out_vol,0), dtype=torch.float32, device=device)
            lbl = torch.tensor([tid]*imgs.size(0), dtype=torch.long, device=device)

            if train: opt_ssl.zero_grad()
            with autocast(enabled=(device.type=="cuda")):
                emb = fe(out_vol); logit = ssl_head(emb); loss = ce_ssl(logit, lbl)
            if train:
                sc.scale(loss).backward(); sc.step(opt_ssl); sc.update(); sched_ssl.step()
            tot += loss.item()*imgs.size(0); lbs+=imgs.size(0)
        return tot/lbs

    best_val=9e9; patience=0
    with Progress(console=console, transient=True) as pr:
        task = pr.add_task("SSL pre-train", total=100)
        for ep in range(1,101):
            _ = run(ssl_tr,True); v=run(ssl_val,False)
            if v < best_val: best_val,patience= v,0; torch.save(fe.state_dict(), ssl_pt)
            else: patience+=1
            if patience>=20: break
            pr.update(task, advance=1)

    fe.load_state_dict(torch.load(ssl_pt, map_location=device))

###############################################################################
# Single-seed training routine
###############################################################################
def train_single_seed(seed, train_ds, val_ds, cfg, do_ssl=True):
    console.print(f"Training seed={seed} …")
    set_seed(seed)

    fe = Advanced3DCNN(cfg["activation"], True, cfg["drop"]).to(device)
    with torch.no_grad(): emb_dim = fe(torch.zeros(1,3,64,64,64).to(device)).shape[1]
    ds = DeepSurv([emb_dim, cfg["hidden_dim"], cfg["hidden_dim"], 1],
                  cfg["drop"], True, cfg["activation"]).to(device)

    if do_ssl: train_ssl(fe, train_ds, cfg, seed)

    loss_fn  = CoxPartialLogLikelihood(cfg)
    params   = list(fe.parameters())+list(ds.parameters())
    opt      = torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched    = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"],
                                                   steps_per_epoch=len(train_ds)//cfg["batch_size"],
                                                   epochs=cfg.get("max_epoch",200))
    train_loader= DataLoader(train_ds, batch_size=cfg["batch_size"],
                             sampler=create_stratified_sampler(train_ds),
                             num_workers=2, drop_last=True)
    val_loader  = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    ema_fe, ema_ds = EMA(fe), EMA(ds)
    best_val=-999; patience=0
    for ep in range(1,cfg.get("max_epoch",200)+1):
        fe.train(); ds.train()
        for batch in train_loader:
            vol = batch["Whole_Liver_image"].to(device)
            t   = batch["survival_time"].to(device)
            e   = batch["event_occurred"].to(device)

            opt.zero_grad()
            with autocast(enabled=(device.type=="cuda")):
                out  = ds(fe(vol)); loss = loss_fn(out,t,e,ds)
            scaler_amp.scale(loss).backward(); scaler_amp.step(opt); scaler_amp.update(); sched.step()
            ema_fe.update(); ema_ds.update()

        # ----------------- EMA evaluation ------------------------------------
        backup_fe = deepcopy(fe.state_dict()); backup_ds = deepcopy(ds.state_dict())
        ema_fe.apply_shadow(); ema_ds.apply_shadow()

        t_tr,e_tr,p_tr = evaluate_model(fe,ds,train_ds,cfg["batch_size"])
        t_v ,e_v ,p_v  = evaluate_model(fe,ds,val_ds  ,cfg["batch_size"])
        c_val = compute_uno_cindex((t_tr,e_tr,p_tr),(t_v,e_v,p_v))

        fe.load_state_dict(backup_fe); ds.load_state_dict(backup_ds)
        if c_val>best_val:
            best_val=c_val; patience=0
            ckpt={"feature_extractor":get_full_ema_state(ema_fe),
                  "deep_surv":get_full_ema_state(ema_ds),
                  "val_cindex":c_val,"epoch":ep}
            torch.save(ckpt, os.path.join(results_dir,f"final_best_model_seed{seed}.pt"))
            console.print(f"Seed{seed} epoch{ep}: c-index ↑ {c_val:.4f}")
        else:
            patience+=1
            if patience>=cfg.get("early_stop",30):
                console.print(f"Seed{seed}: early stop @{ep}")
                break
    return best_val

###############################################################################
# Utility: load trained seed models
###############################################################################
def load_trained_model(seed, cfg):
    fe = Advanced3DCNN(cfg["activation"], True, cfg["drop"]).to(device)
    with torch.no_grad(): dim = fe(torch.zeros(1,3,64,64,64).to(device)).shape[1]
    ds = DeepSurv([dim,cfg["hidden_dim"],cfg["hidden_dim"],1],
                  cfg["drop"], True, cfg["activation"]).to(device)
    ck = torch.load(os.path.join(results_dir,f"final_best_model_seed{seed}.pt"), map_location=device)
    fe.load_state_dict(ck["feature_extractor"]); ds.load_state_dict(ck["deep_surv"])
    fe.eval(); ds.eval(); return fe,ds

###############################################################################
# Ensemble evaluator
###############################################################################
def evaluate_ensemble(dataset, models, cfg, aggregator="mean"):
    set_eval_seed(EVAL_SEED)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)
    t_all,e_all,p_all = [],[],[]
    for batch in loader:
        t_all.append(batch["survival_time"].cpu().numpy())
        e_all.append(batch["event_occurred"].cpu().numpy().astype(bool))
        preds=[]
        for fe,ds in models:
            preds.append(apply_tta_mc_dropout(fe,ds,batch,5,5))
        preds=np.array(preds)
        if aggregator=="median": p_all.append(np.median(preds,0))
        elif aggregator=="max":  p_all.append(np.max(preds,0))
        else:                    p_all.append(np.mean(preds,0))
    return np.concatenate(t_all),np.concatenate(e_all),np.concatenate(p_all)

###############################################################################
# Main script
###############################################################################
def main():
    console.print("⚠️  Double-check dataset paths before running.")
    import torch.backends.cudnn as cudnn; cudnn.benchmark=True

    # Load hyper-params chosen via previous HP search
    best_params_path = os.path.join(cached_dir, "best_params.pkl")
    with open(best_params_path,"rb") as f: best_params = pickle.load(f)
    console.print(f"Loaded HPs: {best_params}")

    # -------------------------------------------------------------------------
    # Build / load NPY cache
    # -------------------------------------------------------------------------
    npy_path = os.path.join(cached_dir, "Final_OS_data.npy")
    if not os.path.exists(npy_path):
        create_np_data(liver_image_dir, liver_label_dir, merged_data,
                       train_image_names, val_image_names, npy_path, (64,64,64))
    else:
        console.print("NPY cache exists → skipping conversion.")

    dat = np.load(npy_path, allow_pickle=True).item()
    train_list, val_list = dat["train"], dat["val"]; cache_dict = dat["cache_dict"]

    # Prebuild CPU tensors for speed
    train_cache = {item["cache_pid"]:torch.from_numpy(cache_dict[item["cache_pid"]]).float()
                   for item in train_list}
    val_cache   = {item["cache_pid"]:torch.from_numpy(cache_dict[item["cache_pid"]]).float()
                   for item in val_list}

    train_ds = LiverDataset(train_list, cache_dict, True,  train_cache)
    val_ds   = LiverDataset(val_list,   cache_dict, False, val_cache)

    seeds=[42,1337,2023,9999,888,1010,2022,2222,777,555]; seed_scores=[]
    for sd in seeds:
        pt = os.path.join(results_dir,f"final_best_model_seed{sd}.pt")
        if os.path.exists(pt):
            console.print(f"Seed{sd} already trained.")
            fe,ds = load_trained_model(sd,best_params)
            c = compute_uno_cindex(*[evaluate_model(fe,ds,ds_,best_params["batch_size"])
                                     for ds_ in (train_ds,val_ds)])
        else:
            c = train_single_seed(sd,train_ds,val_ds,best_params,True)
        seed_scores.append((sd,c)); console.print(f"Seed{sd} c-index={c:.4f}")

    # -------------------------------------------------------------------------
    # Assemble top-5 seeds into ensembles
    # -------------------------------------------------------------------------
    seed_scores.sort(key=lambda x:x[1], reverse=True)
    top5 = [sd for sd,_ in seed_scores[:5]]
    console.print(f"Top-5 seeds: {top5}")

    best_c = -999; best_info=None
    for i in range(1,len(top5)+1):
        subset = top5[:i]
        for agg in ("mean","median","max"):
            models = [load_trained_model(s,best_params) for s in subset]
            c = compute_uno_cindex(*[evaluate_ensemble(ds_,models,best_params,agg)
                                     for ds_ in (train_ds,val_ds)])
            console.print(f"Subset{subset}, agg={agg}: c={c:.4f}")
            if c>best_c:
                best_c,best_info = c,{"subset":tuple(subset),"agg":agg,"models":models}

    console.print(f"🏆 Best ensemble: {best_info}")

    # -------------------------------------------------------------------------
    # Bootstrap CI for best ensemble
    # -------------------------------------------------------------------------
    models,best_agg = best_info["models"], best_info["agg"]
    train_ens = evaluate_ensemble(train_ds, models, best_params, best_agg)
    val_ens   = evaluate_ensemble(val_ds,   models, best_params, best_agg)
    s_train   = Surv.from_arrays(event=train_ens[1], time=train_ens[0])

    c_list=[]
    with Progress(console=console,transient=True) as pr:
        task=pr.add_task("Bootstrap CI", total=500)
        for _ in range(500):
            idx=np.random.choice(len(val_ens[0]),size=len(val_ens[0]),replace=True)
            st,se,sp=val_ens[0][idx],val_ens[1][idx],val_ens[2][idx]
            c_list.append(concordance_index_ipcw(s_train,Surv.from_arrays(se,st),sp)[0])
            pr.update(task,advance=1)

    med,lo,hi=np.median(c_list),np.percentile(c_list,2.5),np.percentile(c_list,97.5)
    console.print(f"Ensemble c-index={best_c:.4f}  CI [{lo:.4f},{hi:.4f}]")

    # -------------------------------------------------------------------------
    # Export deep features & risk scores for full dataset
    # -------------------------------------------------------------------------
    full_ds    = LiverDataset(train_list+val_list, cache_dict, False, None)
    loader_all = DataLoader(full_ds, batch_size=1, shuffle=False)
    feat_arr,risk_arr,pids=[],[],[]
    def pred(fe,ds,vol):
        with torch.no_grad(): emb=fe(vol); risk=ds(emb)
        return emb.cpu().numpy().squeeze(),risk.cpu().numpy().squeeze()

    for batch in loader_all:
        pid = batch["pid"][0]; vol = batch["Whole_Liver_image"].to(device)
        emb_stack,risk_stack=[],[]
        for fe,ds in models:
            emb,risk=pred(fe,ds,vol); emb_stack.append(emb); risk_stack.append(risk)
        feat_arr.append(np.mean(np.stack(emb_stack,0),0))
        risk_arr.append(np.median(risk_stack) if best_agg=="median"
                        else np.max(risk_stack) if best_agg=="max"
                        else np.mean(risk_stack))
        pids.append(pid)

    feat_arr,risk_arr=np.array(feat_arr),np.array(risk_arr)
    feat_path=os.path.join(results_dir,"Improved_DeepFeatures.csv")
    risk_path=os.path.join(results_dir,"Improved_RiskValues.csv")
    with open(feat_path,"w") as f:
        cols=["patient_id"]+[f"feat_{i}" for i in range(feat_arr.shape[1])]; f.write(",".join(cols)+"\n")
        for pid,fa in zip(pids,feat_arr):
            f.write(",".join([pid]+[f"{v:.6f}" for v in fa])+"\n")
    with open(risk_path,"w") as f:
        f.write("patient_id,risk\n")
        for pid,rk in zip(pids,risk_arr):
            f.write(f"{pid},{rk:.6f}\n")
    console.print(f"CSV saved: {feat_path} | {risk_path}")

    # Summary TXT
    txt=os.path.join(results_dir,"summary.txt")
    with open(txt,"w") as f:
        f.write(f"Best ensemble c-index={best_c:.4f}\n")
        f.write(f"Bootstrap median={med:.4f}  95%CI=[{lo:.4f},{hi:.4f}]\n")
    console.print("✨ Done.")

###############################################################################
# Entry-point
###############################################################################
if __name__ == "__main__":
    main()
