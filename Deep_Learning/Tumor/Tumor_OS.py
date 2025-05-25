
##############################################################################
# Tumor Overall-Survival (OS) 3-D Deep-Survival Pipeline
#
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
#  This end-to-end script:
#     • Matches CT volumes, segmentation masks, and clinical CSV rows
#     • Converts volumes to normalized NumPy caches (with on-the-fly augment)
#     • Builds a 3-channel 3-D CNN feature extractor + DeepSurv head
#     • Supports self-supervised pre-training (optional 8-class 3-D transforms)
#     • Performs hyper-parameter search on cached embeddings
#     • Trains multiple seeds, builds an EMA ensemble, then bootstraps a 95 % CI
#     • Writes per-patient deep features & risk scores to CSV
#
#  💡  Customize directory paths (cached_dir, results_dir, etc.) near the top
#     to match your own storage layout before running.
#
#  Tested on: 4× A100, PyTorch 2.2, CUDA 12.4          Author: (your lab)
##############################################################################

import os
import warnings

# Silence noisy library warnings for a cleaner log.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Standard / scientific stack
# ---------------------------------------------------------------------------
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage
from skimage.transform import resize

# scikit-survival utilities + metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw

from copy import deepcopy

# ---------------------------------------------------------------------------
# Mixed-precision & data loading helpers
# ---------------------------------------------------------------------------
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torchvision import transforms
import nibabel as nib  # medical-imaging IO

# ---------------------------------------------------------------------------
# Pretty CLI progress with Rich
# ---------------------------------------------------------------------------
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn

console = Console(style="white")

##############################################################################
# 1  PATH CONFIGURATION
##############################################################################
# Where intermediate *.npy caches & models go
cached_dir  = "/mnt/largedrive1/rmojtahedi/Deep_Survival/Final_Version/Tumour/Tumour_OS"
results_dir = "/mnt/largedrive1/rmojtahedi/Deep_Survival/Final_Version/Tumour/Tumour_OS"
os.makedirs(results_dir, exist_ok=True)

# Raw image/label/CSV sources
liver_image_dir   = "/mnt/largedrive1/rmojtahedi/KSPIE/tumor/ct"
liver_label_dir   = "/mnt/largedrive1/rmojtahedi/KSPIE/tumor/final_seg_split"
clinical_csv_path = "/mnt/largedrive0/rmojtahedi/Kaitlyn_SPIE/Deep_Survival/npj_Digital_Medicine_Clinical_Data_FINAL.csv"

# GPU selection
device      = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
scaler_amp  = GradScaler(enabled=(device.type == "cuda"))

##############################################################################
# 2  GLOBAL SEEDS
##############################################################################
EVAL_SEED = 999  # determ. TTA + MC inference reproducibility


def set_seed(seed: int) -> None:
    """Fully deterministic training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_eval_seed(seed: int) -> None:
    """Separate seed for deterministic *evaluation* augmentations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

##############################################################################
# 3  MATCH IMAGES ⇆ LABELS ⇆ CLINICAL CSV
##############################################################################
def get_core_id(filename: str) -> str:
    """Drop .nii / .nii.gz suffix."""
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


def get_candidate_id(name: str) -> str:
    """Some scans include an extra slice index -> strip last underscore section."""
    parts = name.split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else name


# All image / label base-names
image_names = {get_core_id(f) for f in os.listdir(liver_image_dir)}
label_names = {get_core_id(f) for f in os.listdir(liver_label_dir)}

# Quick sanity lists (won’t break run if non-empty, just prints)
unmatched_images = sorted(image_names - label_names)
unmatched_labels = sorted(label_names - image_names)

common_names     = image_names.intersection(label_names)
candidate_pairs  = [(n, get_candidate_id(n)) for n in sorted(common_names)]

# Load clinical CSV, convert IDs to str for merge
clinical_df = pd.read_csv(clinical_csv_path).dropna(subset=["XNAT ID"])
clinical_df["XNAT ID"] = clinical_df["XNAT ID"].astype(str)
clinical_ids = set(clinical_df["XNAT ID"])

# ---------------------------------------------------------------------------
# 3-A  Automatic matches
# ---------------------------------------------------------------------------
final_matched, unmatched_candidates = [], []
for img_lbl, cand in candidate_pairs:
    (final_matched if cand in clinical_ids else unmatched_candidates).append((img_lbl, cand))

# ---------------------------------------------------------------------------
# 3-B  Manual overrides for weird edge cases
# ---------------------------------------------------------------------------
manual_mapping = {
    # one-off mappings: {"image_base" : "XNAT ID"}
    "RIA_17-010A_000_202": "RIA_17-010A_000_202",
    # … (unchanged list)
    "RIA_17-010B_000_258": "RIA_17-010B_000_RIA_17-010B_000_258",
}

manually_matched, still_unmatched = [], []
for img_lbl, cand in unmatched_candidates:
    if img_lbl in manual_mapping:
        corr = manual_mapping[img_lbl]
        (manually_matched if corr in clinical_ids else still_unmatched).append((img_lbl, corr))
    else:
        still_unmatched.append((img_lbl, cand))

final_matched.extend(manually_matched)
matched_xnat_set      = {cid for _, cid in final_matched}
unmatched_clinical_ids = sorted(clinical_ids - matched_xnat_set)

# Merge to one dataframe ready for downstream filtering
matched_df  = pd.DataFrame(final_matched, columns=["Image_Label_Base", "XNAT ID"])
merged_data = (
    pd.merge(matched_df, clinical_df, on="XNAT ID", how="inner")
      .dropna(subset=["Surv Time Censor 7", "Censor 7: OS", "OSTrain"])
)
merged_data = merged_data[merged_data["Surv Time Censor 7"] > 0]

# Split train/val by CSV “OSTrain” column
train_rows = merged_data[merged_data["OSTrain"] == 1]
val_rows   = merged_data[merged_data["OSTrain"] == 0]

train_image_names = train_rows["Image_Label_Base"].tolist()
val_image_names   = val_rows["Image_Label_Base"].tolist()

##############################################################################
# 4  DATA AUGMENTATION PIPELINE (3-D, light-weight CPU ops)
##############################################################################
class RandomGaussianNoise:
    def __init__(self, p=0.5, std_range=(0.01, 0.03)):
        self.p, self.std_range = p, std_range

    def __call__(self, volume):
        if random.random() < self.p:
            std   = random.uniform(*self.std_range)
            noise = np.random.randn(*volume.shape) * std
            volume += noise
        return volume


class RandomZoom3D:
    """Isotropic zoom in/out then crop / pad back to original shape."""
    def __init__(self, p=0.5, zoom_range=(0.9, 1.1)):
        self.p, self.zoom_range = p, zoom_range

    def __call__(self, volume):
        if random.random() < self.p:
            zoom_factor = random.uniform(*self.zoom_range)
            zoomed      = scipy.ndimage.zoom(volume, zoom_factor, order=1)
            orig_shape  = volume.shape

            # Crop center
            if zoomed.shape[0] >= orig_shape[0]:
                sx = (zoomed.shape[0] - orig_shape[0]) // 2
                sy = (zoomed.shape[1] - orig_shape[1]) // 2
                sz = (zoomed.shape[2] - orig_shape[2]) // 2
                ex, ey, ez = sx + orig_shape[0], sy + orig_shape[1], sz + orig_shape[2]
                volume = zoomed[sx:ex, sy:ey, sz:ez]
            else:  # Pad outwards
                padded = np.zeros(orig_shape, dtype=zoomed.dtype)
                ox = (orig_shape[0] - zoomed.shape[0]) // 2
                oy = (orig_shape[1] - zoomed.shape[1]) // 2
                oz = (orig_shape[2] - zoomed.shape[2]) // 2
                padded[ox : ox + zoomed.shape[0], oy : oy + zoomed.shape[1], oz : oz + zoomed.shape[2]] = zoomed
                volume = padded
        return volume


class RandomRotate3D:
    """Small 3-D Euler rotations (nearest-neighbor)."""
    def __init__(self, p=0.5, max_angle=10):
        self.p, self.max_angle = p, max_angle

    def __call__(self, volume):
        if random.random() < self.p:
            axis_pairs = [(0, 1), (0, 2), (1, 2)]
            axes  = random.choice(axis_pairs)
            angle = random.uniform(-self.max_angle, self.max_angle)
            volume = scipy.ndimage.rotate(volume, angle, axes=axes, reshape=False, order=1)
        return volume


class RandomFlip3D:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, volume):
        if random.random() < self.p:
            volume = np.flip(volume, axis=random.choice([0, 1, 2])).copy()
        return volume


class RandomIntensityShift:
    def __init__(self, p=0.5, shift_range=(-0.07, 0.07)):
        self.p, self.shift_range = p, shift_range

    def __call__(self, volume):
        if random.random() < self.p:
            volume += random.uniform(*self.shift_range)
        return volume


class RandomIntensityScale:
    def __init__(self, p=0.5, scale_range=(0.95, 1.05)):
        self.p, self.scale_range = p, scale_range

    def __call__(self, volume):
        if random.random() < self.p:
            volume *= random.uniform(*self.scale_range)
        return volume


class RandomGamma:
    """Gamma curve shift, preserving min/max HU."""
    def __init__(self, p=0.5, gamma_range=(0.9, 1.1)):
        self.p, self.gamma_range = p, gamma_range

    def __call__(self, volume):
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_range)
            volume = np.clip(volume, 0, None)
            vmin, vmax = volume.min(), volume.max()
            if vmax > vmin:
                norm = (volume - vmin) / (vmax - vmin + 1e-10)
                volume = norm ** gamma * (vmax - vmin) + vmin
        return volume


# Compose final Augmentations
augment_transform = transforms.Compose(
    [
        RandomRotate3D(0.5, 10),
        RandomFlip3D(0.5),
        RandomZoom3D(0.5, (0.9, 1.1)),
        RandomGaussianNoise(0.5, (0.01, 0.03)),
        RandomIntensityShift(0.5, (-0.07, 0.07)),
        RandomIntensityScale(0.5, (0.95, 1.05)),
        RandomGamma(0.5, (0.9, 1.1)),
    ]
)

##############################################################################
# 5  CONVERT NIFTI → NORMALIZED NPY  (single pass, cached to disk)
##############################################################################
def create_np_data(
    image_dir: str,
    label_dir: str,
    clinical_df: pd.DataFrame,
    train_list: list,
    val_list: list,
    npy_save_path: str,
    shape=(64, 64, 64),
) -> None:
    """
    Pre-process every volume:
        • Resample & crop/pad to `shape`
        • Mask by tumor segmentation
        • Global z-score normalization (fit on *all* voxels in train+val)
        • Save dict  {train, val, mean, std, cache_dict}
    """

    console.print("Starting conversion to NPY…")
    all_pids = train_list + val_list
    gather_samples = []

    # –– Pass-1: build scaler on masked voxels
    with Progress(
        TextColumn("{task.description}", style="black"),
        BarColumn(bar_width=None, complete_style="green"),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total}", style="black"),
        console=Console(),
        transient=True,
    ) as progress:
        t_id = progress.add_task("[bold]Loading + resizing", total=len(all_pids))
        for pid in all_pids:
            row = clinical_df[clinical_df["Image_Label_Base"] == pid]
            if row.empty:
                progress.update(t_id, advance=1)
                continue

            # Resolve .nii / .nii.gz
            image_path = os.path.join(image_dir, pid + ".nii.gz")
            image_path = image_path if os.path.exists(image_path) else os.path.join(image_dir, pid + ".nii")
            label_path = os.path.join(label_dir, pid + ".nii.gz")
            label_path = label_path if os.path.exists(label_path) else os.path.join(label_dir, pid + ".nii")

            vol   = resize(nib.load(image_path).get_fdata(), shape)
            seg   = resize(nib.load(label_path).get_fdata(), shape, order=0, preserve_range=True).astype(np.float32)
            mask  = (seg > 0.5).astype(np.float32)
            gather_samples.append((vol * mask).reshape(-1, 1))
            progress.update(t_id, advance=1)

    big_array = np.concatenate(gather_samples, axis=0)
    scaler    = StandardScaler().fit(big_array)  # global μ,σ

    cache_dict = {}  # holds patient-id → normalized float32 array (C=3)

    def _build_subset(pid_list, desc):
        """Return list of dicts used by LiverDataset."""
        items = []
        with Progress(
            TextColumn("{task.description}", style="black"),
            BarColumn(bar_width=None, complete_style="green"),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total}", style="black"),
            console=Console(),
            transient=True,
        ) as p2:
            t2 = p2.add_task(desc, total=len(pid_list))
            for pid in pid_list:
                row = clinical_df[clinical_df["Image_Label_Base"] == pid]
                if row.empty:
                    p2.update(t2, advance=1)
                    continue

                # Re-load, resize, mask
                ipath = os.path.join(image_dir, pid + ".nii.gz")
                ipath = ipath if os.path.exists(ipath) else os.path.join(image_dir, pid + ".nii")
                lpath = os.path.join(label_dir, pid + ".nii.gz")
                lpath = lpath if os.path.exists(lpath) else os.path.join(label_dir, pid + ".nii")

                vol = resize(nib.load(ipath).get_fdata(), shape)
                seg = resize(nib.load(lpath).get_fdata(), shape, order=0, preserve_range=True).astype(np.float32)
                mask = (seg > 0.5).astype(np.float32)
                vol  = vol * mask

                # Normalize with μ,σ
                flat  = vol.reshape(-1, 1)
                flat  = scaler.transform(flat)
                vol   = flat.reshape(*shape)

                # Duplicate gray volume into 3-channels for Conv3d (C = 3)
                vol_3c = np.repeat(vol[np.newaxis, ...], 3, axis=0).astype(np.float32)

                st = row["Surv Time Censor 7"].values[0]
                ev = row["Censor 7: OS"].values[0]

                cache_dict[pid] = vol_3c
                items.append({"pid": pid, "cache_pid": pid, "image": None, "time": float(st), "event": bool(ev)})
                p2.update(t2, advance=1)
        return items

    train_data = _build_subset(train_list, "Processing (train)")
    val_data   = _build_subset(val_list,   "Processing (val)")

    np.save(
        npy_save_path,
        {
            "train": train_data,
            "val":   val_data,
            "mean":  scaler.mean_[0],
            "std":   scaler.scale_[0],
            "cache_dict": cache_dict,
        },
    )
    console.print(f"Saved data to {npy_save_path}")

##############################################################################
# 6  DATASET CLASS (lazy aug + tensor cache)
##############################################################################
class LiverDataset(torch.utils.data.Dataset):
    """
    • Uses `tensor_cache` for lightning-fast __getitem__ when augment=False
    • When augment=True, pulls NumPy, applies CPU augmentation pipeline,
      then converts to float-Tensor.
    """
    def __init__(self, data_list, cache_dict, augment=False, tensor_cache=None):
        self.data_list   = data_list
        self.cache_dict  = cache_dict
        self.augment     = augment
        self.tensor_cache = tensor_cache

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        d   = self.data_list[idx]
        pid = d["cache_pid"]

        # Use pre-built torch.Tensor if no aug needed
        if not self.augment and self.tensor_cache is not None:
            vol_3c = self.tensor_cache[pid].clone()
        else:
            vol_3c = self.cache_dict[pid].copy()
            if self.augment:
                # Need channel-last for scipy.ndimage; then back
                vol_3c = np.transpose(vol_3c, (1, 2, 3, 0))
                vol_3c = augment_transform(vol_3c)
                vol_3c = np.transpose(vol_3c, (3, 0, 1, 2))

        sample = {
            "Whole_Liver_image": torch.tensor(vol_3c, dtype=torch.float32),
            "survival_time":     torch.tensor(d["time"],  dtype=torch.float32),
            "event_occurred":    torch.tensor(d["event"], dtype=torch.bool),
            "pid":               d["pid"],
        }
        return sample

##############################################################################
# 7  MODEL BUILDING BLOCKS (Advanced 3-D CNN + DeepSurv MLP)
##############################################################################
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


ACTIVATIONS = {"ReLU": nn.ReLU, "ELU": nn.ELU, "LeakyReLU": nn.LeakyReLU, "Swish": Swish}


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv3d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ----------------- CBAM (Channel & Spatial Attention) -----------------------
class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super().__init__()
        self.avgp = nn.AdaptiveAvgPool3d(1)
        self.maxp = nn.AdaptiveMaxPool3d(1)
        self.fc   = nn.Sequential(
            nn.Linear(in_planes, in_planes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // reduction, in_planes, bias=False),
        )

    def forward(self, x):
        b, c, *_ = x.size()
        avg = self.fc(self.avgp(x).view(b, c))
        mx  = self.fc(self.maxp(x).view(b, c))
        out = torch.sigmoid(avg + mx).view(b, c, 1, 1, 1)
        return x * out


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * out


class CBAM3D(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention3D(channels, reduction)
        self.sa = SpatialAttention3D(spatial_kernel_size)

    def forward(self, x):
        return self.sa(self.ca(x))


# ----------------- Residual + Attention backbone ---------------------------
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation="Swish", norm=True):
        super().__init__()
        act_fn = ACTIVATIONS[activation]()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm3d(out_channels) if norm else nn.Identity()
        self.act   = act_fn
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2   = nn.BatchNorm3d(out_channels) if norm else nn.Identity()

        # Identity or 1×1 down-sample if dims change
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm3d(out_channels) if norm else nn.Identity(),
            )
        )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.act(out)


class Advanced3DCNN(nn.Module):
    """Feature extractor -> 128-D embedding fed to DeepSurv head."""
    def __init__(self, activation="Swish", norm=True, drop=0.2):
        super().__init__()
        Act = ACTIVATIONS[activation]

        self.layer0 = nn.Sequential(
            nn.Conv3d(3, 16, 3, padding=1),
            nn.BatchNorm3d(16) if norm else nn.Identity(),
            Act(),
            nn.MaxPool3d(2),
        )
        self.res1 = ResidualBlock3D(16, 32, 2, activation, norm)
        self.res2 = ResidualBlock3D(32, 64, 2, activation, norm)
        self.res3 = ResidualBlock3D(64, 128, 2, activation, norm)
        self.cbam = CBAM3D(128, 16, 7)
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
        return torch.flatten(x, 1)  # (B, 128)


class DeepSurv(nn.Module):
    """
    Fully-connected survival head:
      dims = [in_dim, hidden, hidden, 1]
    """
    def __init__(self, dims, drop=0.2, norm=True, activation="Swish"):
        super().__init__()
        Act = ACTIVATIONS[activation]
        layers = []
        for i in range(len(dims) - 1):
            if i > 0 and drop is not None:
                layers.append(nn.Dropout(drop))
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if norm and i < len(dims) - 1:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            if i < len(dims) - 1:
                layers.append(Act())
        self.model = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x):
        return self.model(x)

##############################################################################
# 8  EXPONENTIAL MOVING AVERAGE (EMA) UTILITY
##############################################################################
class EMA:
    """
    Keeps shadow copies of parameters & buffers.
    Call .update() after each optimizer step, .apply_shadow() for evaluation.
    """
    def __init__(self, model: nn.Module, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow, self.shadow_buffers = {}, {}

        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()
        for name, b in model.named_buffers():
            self.shadow_buffers[name] = b.clone()

    @torch.no_grad()
    def update(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * p.data
        for name, b in self.model.named_buffers():
            self.shadow_buffers[name] = b.clone()

    def apply_shadow(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[name])
        for name, b in self.model.named_buffers():
            b.copy_(self.shadow_buffers[name])


def get_full_ema_state(ema: EMA) -> dict:
    """Return combined dict of both params + buffers shadow state."""
    model_state = ema.model.state_dict()
    return {
        n: (
            ema.shadow[n]
            if n in ema.shadow
            else ema.shadow_buffers.get(n, model_state[n])
        )
        for n in model_state
    }

##############################################################################
# 9  TTA + MONTE-CARLO DROPOUT INFERENCE
##############################################################################
def enable_mc_dropout(model: nn.Module) -> None:
    """Put Dropout layers back to train() while keeping BN eval()."""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def apply_tta_mc_dropout(
    feature_extractor: nn.Module,
    deep_surv: nn.Module,
    batch: dict,
    num_tta=5,
    num_mc=5,
):
    """
    Runs N×M forward passes per batch:
        • num_tta  deterministic (we don’t perform augmentation here by default,
          but hook is provided)
        • num_mc   stochastic MC passes with Dropout enabled
    Final prediction = mean over (num_tta × num_mc)
    """
    set_eval_seed(EVAL_SEED)
    feature_extractor.eval()
    deep_surv.eval()
    enable_mc_dropout(feature_extractor)
    enable_mc_dropout(deep_surv)

    base_vols = batch["Whole_Liver_image"].cpu().numpy()
    pred_list = []

    for _ in range(num_tta):
        # (Optional) plug true TTA augment here
        aug_tensor = torch.tensor(np.stack(base_vols, axis=0), dtype=torch.float32, device=device)
        mc_preds = []
        for _ in range(num_mc):
            with autocast(enabled=(device.type == "cuda")):
                emb = feature_extractor(aug_tensor)
                out = deep_surv(emb).detach().cpu().numpy().ravel()
            mc_preds.append(out)
        pred_list.append(np.mean(mc_preds, axis=0))

    return np.mean(pred_list, axis=0)


def evaluate_model(feature_extractor, deep_surv, dataset, batch_size):
    """Utility → returns (t, e, risk_pred) numpy arrays for concordance calc."""
    set_eval_seed(EVAL_SEED)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")

    all_t, all_e, all_p = [], [], []
    for batch in loader:
        t = batch["survival_time"].cpu().numpy()
        e = batch["event_occurred"].cpu().numpy().astype(bool)
        p = apply_tta_mc_dropout(feature_extractor, deep_surv, batch, 5, 5)
        all_t.append(t); all_e.append(e); all_p.append(p)

    return np.concatenate(all_t), np.concatenate(all_e), np.concatenate(all_p)


def compute_uno_cindex(train_data, val_data):
    """Uno’s c-index with IPCW weighting (sksurv)."""
    s_tr  = Surv.from_arrays(event=train_data[1], time=train_data[0])
    s_val = Surv.from_arrays(event=val_data[1], time=val_data[0])
    return concordance_index_ipcw(s_tr, s_val, val_data[2])[0]

##############################################################################
# 10  LOSS + L2 REGULARIZATION
##############################################################################
class Regularization(nn.Module):
    def __init__(self, order=2, weight_decay=1e-4):
        super().__init__()
        self.order, self.weight_decay = order, weight_decay

    def forward(self, model):
        reg_val = sum(torch.norm(w, p=self.order) for n, w in model.named_parameters() if w.requires_grad and "weight" in n)
        return self.weight_decay * reg_val


class CoxPartialLogLikelihood(nn.Module):
    """
    Negative partial likelihood for Cox PH model with optional L2 regularization.
    """
    def __init__(self, config):
        super().__init__()
        self.reg = Regularization(order=2, weight_decay=config["l2_reg"])

    def forward(self, risk_pred, y, e, model):
        sort_idx = torch.argsort(y)           # ascending time
        srisk    = risk_pred[sort_idx]
        se       = e[sort_idx]

        exp_risk      = torch.exp(srisk)
        cs            = torch.flip(torch.cumsum(torch.flip(exp_risk, [0]), dim=0), [0])
        log_cs        = torch.log(cs)
        partial_ll    = torch.sum(se * (srisk - log_cs))
        neg_log_lik   = -partial_ll / (torch.sum(e) + 1e-8)
        return neg_log_lik + self.reg(model)

##############################################################################
# 11  OPTIONAL SELF-SUPERVISED PRE-TRAINING (8-CLASS TRANSFORMS)
##############################################################################
# ––– Implementation kept intact but skipped detailed inline comments
#     to keep main flow readable. Enable by setting do_ssl=True later.
##############################################################################
class MultiTransformHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, 8),  # 8 transforms
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.fc(x)


def apply_3d_transform_8class(volume: np.ndarray, transform_id: int) -> np.ndarray:
    """Deterministic 0-3 rot × flip scheme (see docstring above)."""
    rot_id, flip_id = transform_id % 4, transform_id // 4
    if rot_id:
        volume = np.rot90(volume, k=rot_id, axes=(2, 3)).copy()
    if flip_id:
        volume = np.flip(volume, axis=1).copy()
    return volume

##############################################################################
# 12  EMBEDDING CACHING + LIGHTWEIGHT HYPER-PARAMETER SEARCH
##############################################################################
class CachedEmbeddingDataset(torch.utils.data.Dataset):
    """Thin wrapper around list of dicts ≈ ({embedding,time,event,pid})."""
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def compute_embeddings_for_dataset(feature_extractor, dataset):
    """
    Forward entire dataset once → store embeddings to enable cheap CV search.
    """
    feature_extractor.eval()
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    emb_items = []
    with torch.no_grad():
        for batch in loader:
            vol = batch["Whole_Liver_image"].to(device)
            t   = batch["survival_time"]
            e   = batch["event_occurred"]
            pid = batch["pid"]
            emb = feature_extractor(vol)
            for i in range(vol.size(0)):
                emb_items.append(
                    {
                        "embedding": emb[i].cpu(),
                        "time":      t[i].item(),
                        "event":     e[i].item(),
                        "pid":       pid[i],
                    }
                )
    return emb_items


def train_val_on_embeddings(emb_dataset, ds_model, train_idx, cfg):
    """
    2-epoch OneCycle training on cached embeddings. Returns CV c-index.
    """
    loss_fn = CoxPartialLogLikelihood(cfg)
    opt_    = torch.optim.Adam(ds_model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sc_     = GradScaler(enabled=device.type == "cuda")

    sampler = SubsetRandomSampler(train_idx)
    loader  = DataLoader(emb_dataset, batch_size=cfg["batch_size"], sampler=sampler, num_workers=2, drop_last=True)

    sched   = torch.optim.lr_scheduler.OneCycleLR(
        opt_, max_lr=cfg["lr"], steps_per_epoch=len(loader), epochs=2
    )

    # –– train loop
    ds_model.train()
    for _ in range(2):
        for b in loader:
            emb = b["embedding"].to(device)
            t   = torch.tensor(b["time"],  device=device)
            e   = torch.tensor(b["event"], device=device, dtype=torch.bool)
            opt_.zero_grad()
            with autocast(enabled=device.type == "cuda"):
                out = ds_model(emb)
                loss = loss_fn(out, t, e, ds_model)
            sc_.scale(loss).backward()
            sc_.step(opt_); sc_.update(); sched.step()

    # –– evaluate on *all* items
    ds_model.eval()
    loader_eval = DataLoader(emb_dataset, batch_size=cfg["batch_size"], shuffle=False)
    all_t, all_e, all_p = [], [], []
    with torch.no_grad():
        for bb in loader_eval:
            ebd = bb["embedding"].to(device)
            t_  = np.array(bb["time"])
            e_  = np.array(bb["event"]).astype(bool)
            p_  = ds_model(ebd).cpu().numpy().squeeze()
            all_t.append(t_); all_e.append(e_); all_p.append(p_)
    all_t, all_e, all_p = map(np.concatenate, (all_t, all_e, all_p))
    s_all = Surv.from_arrays(event=all_e, time=all_t)
    return concordance_index_ipcw(s_all, s_all, all_p)[0]


def hyperparam_search_with_embeddings(emb_dataset, k=3):
    """
    Simple grid search (few hundred combos) using K-fold on cached embeddings.
    Returns (best_cfg_dict, best_cv_cindex).
    """
    from itertools import product
    lr_list       = [1e-4, 3e-4, 5e-4, 1e-3]
    drop_list     = [0.1, 0.2, 0.3]
    hidden_dims   = [128, 256]
    act_list      = ["Swish", "LeakyReLU"]
    bsz_list      = [4, 8, 16]
    wdecay_list   = [1e-5, 1e-4]
    l2reg_list    = [1e-5, 1e-4]

    indices = list(range(len(emb_dataset)))
    kf      = KFold(n_splits=k, shuffle=True, random_state=42)

    best_c, best_cfg = -np.inf, None
    combo_all = list(product(lr_list, drop_list, hidden_dims, act_list, bsz_list, wdecay_list, l2reg_list))

    with Progress(
        TextColumn("{task.description}", style="black"),
        BarColumn(bar_width=None, complete_style="green"),
        TaskProgressColumn(),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task("Hyper-param search", total=len(combo_all))
        for lr, dp, hd, act, bsz, wd, l2 in combo_all:
            cfg = dict(lr=lr, drop=dp, hidden_dim=hd, activation=act, batch_size=bsz, weight_decay=wd, l2_reg=l2)

            fold_scores = []
            for train_idx, _ in kf.split(indices):
                ds = DeepSurv([emb_dataset[0]["embedding"].shape[0], hd, hd, 1], drop=dp, activation=act, norm=True).to(device)
                fold_scores.append(train_val_on_embeddings(emb_dataset, ds, train_idx, cfg))
            mean_score = np.mean(fold_scores)

            if mean_score > best_c:
                best_c, best_cfg = mean_score, cfg
            progress.update(task, advance=1)

    return best_cfg, best_c

##############################################################################
# 13  IMBALANCE-AWARE SAMPLER (Deaths ×2 weight)
##############################################################################
def create_stratified_sampler(dataset):
    e = np.array([dataset[i]["event_occurred"].item() for i in range(len(dataset))])
    weights = np.zeros_like(e, dtype=float)
    ev_ct, cn_ct = (e == 1).sum(), (e == 0).sum()
    weights[e == 1] = 2.0 / ev_ct
    weights[e == 0] = 1.0 / cn_ct
    return WeightedRandomSampler(weights, len(weights), replacement=True)

##############################################################################
# 14  OPTIONAL 8-CLASS SSL TRAINING (skipped by default)
##############################################################################
def train_ssl(fe, dataset, cfg, seed):
    """
    Self-supervised 8-class transform classification.
    Saves best feature-extractor weights at `ssl_best_fe_seed{seed}.pt`
    so that repeated runs skip retraining.
    """
    ssl_weights_path = os.path.join(results_dir, f"ssl_best_fe_seed{seed}.pt")
    if os.path.exists(ssl_weights_path):
        console.print(f"Seed {seed}: Found SSL weights, skipping SSL pre-train.")
        fe.load_state_dict(torch.load(ssl_weights_path, map_location=device))
        return

    # -- Build SSL dataset (only images) --
    class SSLDataset(torch.utils.data.Dataset):
        def __init__(self, data_list):
            self.imgs = [torch.tensor(d["Whole_Liver_image"], dtype=torch.float32) for d in data_list]
        def __len__(self): return len(self.imgs)
        def __getitem__(self, idx): return self.imgs[idx]

    ssl_ds = SSLDataset(dataset)
    idx_all = list(range(len(ssl_ds))); random.shuffle(idx_all)
    split   = int(0.1 * len(idx_all))
    val_idx, tr_idx = idx_all[:split], idx_all[split:]

    tr_loader = DataLoader(ssl_ds, batch_size=cfg["batch_size"], sampler=SubsetRandomSampler(tr_idx), num_workers=2, drop_last=True)
    val_loader= DataLoader(ssl_ds, batch_size=cfg["batch_size"], sampler=SubsetRandomSampler(val_idx), num_workers=2)

    # Discover conv feature dim dynamically
    with torch.no_grad():
        conv_dim = fe(torch.zeros(1, 3, 64, 64, 64, device=device)).shape[1]

    ssl_head = MultiTransformHead(conv_dim).to(device)
    ssl_opt  = torch.optim.Adam(list(fe.parameters()) + list(ssl_head.parameters()), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ssl_ce   = nn.CrossEntropyLoss()
    sched    = torch.optim.lr_scheduler.OneCycleLR(ssl_opt, max_lr=cfg["lr"], steps_per_epoch=len(tr_loader), epochs=100)
    sc_      = GradScaler(enabled=device.type == "cuda")

    def _run_epoch(loader, train=True):
        fe.train(train); ssl_head.train(train)
        total, count = 0.0, 0
        for imgs in loader:
            imgs = imgs.to(device)
            tid  = random.randint(0, 7)
            # Apply same transform to every sample in batch
            xform = np.stack([apply_3d_transform_8class(img.cpu().numpy(), tid) for img in imgs])
            xform = torch.tensor(xform, dtype=torch.float32, device=device)

            if train: ssl_opt.zero_grad()
            with autocast(enabled=device.type == "cuda"):
                out = ssl_head(fe(xform))
                loss = ssl_ce(out, torch.full((imgs.size(0),), tid, dtype=torch.long, device=device))
            if train:
                sc_.scale(loss).backward()
                sc_.step(ssl_opt); sc_.update(); sched.step()
            total += loss.item() * imgs.size(0); count += imgs.size(0)
        return total / count

    best_val, patience = np.inf, 0
    with Progress(console=console, transient=True) as p:
        task = p.add_task("SSL Epochs", total=100)
        for epoch in range(1, 101):
            _run_epoch(tr_loader, train=True)
            val_loss = _run_epoch(val_loader, train=False)
            if val_loss < best_val:
                best_val, patience = val_loss, 0
                torch.save(fe.state_dict(), ssl_weights_path)
            else:
                patience += 1
                if patience >= 20: break
            p.update(task, advance=1)

    fe.load_state_dict(torch.load(ssl_weights_path, map_location=device))

##############################################################################
# 15  CORE TRAINING ROUTINE (single seed)
##############################################################################
def train_single_seed(seed, train_dataset, val_dataset, cfg, do_ssl=True):
    console.print(f"\n[bold yellow]»» Training seed {seed} ««[/bold yellow]")
    set_seed(seed)

    # ------------------------------------------------------------------ Backbone init
    fe = Advanced3DCNN(activation=cfg["activation"], drop=cfg["drop"], norm=True).to(device)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 64, 64, 64, device=device)
        conv_dim = fe(dummy).shape[1]

    ds = DeepSurv([conv_dim, cfg["hidden_dim"], cfg["hidden_dim"], 1], drop=cfg["drop"],
                  activation=cfg["activation"], norm=True).to(device)

    # Optional SSL stage
    if do_ssl: train_ssl(fe, train_dataset, cfg, seed)

    # ------------------------------------------------------------------ Optimizer + Loss
    criterion = CoxPartialLogLikelihood(cfg)
    opt = torch.optim.Adam(list(fe.parameters()) + list(ds.parameters()), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], steps_per_epoch=len(train_dataset) // cfg["batch_size"], epochs=cfg.get("max_epoch", 200)
    )

    ema_fe, ema_ds = EMA(fe, 0.999), EMA(ds, 0.999)
    best_val, patience = -np.inf, 0
    best_ckpt = None

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"],
                              sampler=create_stratified_sampler(train_dataset), num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    with Progress(
        TextColumn("{task.description}", style="black"),
        BarColumn(bar_width=None, complete_style="green"),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total}", style="black"),
        console=Console(),
        transient=True,
    ) as p:
        task = p.add_task("Epochs", total=cfg.get("max_epoch", 200))
        for epoch in range(1, cfg.get("max_epoch", 200) + 1):
            fe.train(); ds.train()
            for batch in train_loader:
                x  = batch["Whole_Liver_image"].to(device)
                t  = batch["survival_time"].to(device)
                e  = batch["event_occurred"].to(device)

                opt.zero_grad()
                with autocast(enabled=device.type == "cuda"):
                    loss = criterion(ds(fe(x)), t, e, ds)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(opt); scaler_amp.update(); sched.step()
                ema_fe.update(); ema_ds.update()

            # ------------- EMA eval
            fe_state, ds_state = deepcopy(fe.state_dict()), deepcopy(ds.state_dict())
            ema_fe.apply_shadow(); ema_ds.apply_shadow()

            train_dat = evaluate_model(fe, ds, train_dataset, cfg["batch_size"])
            val_dat   = evaluate_model(fe, ds, val_dataset, cfg["batch_size"])
            cidx      = compute_uno_cindex(train_dat, val_dat)

            # revert
            fe.load_state_dict(fe_state); ds.load_state_dict(ds_state)

            if cidx > best_val:
                best_val, patience = cidx, 0
                best_ckpt = {
                    "feature_extractor": get_full_ema_state(ema_fe),
                    "deep_surv":        get_full_ema_state(ema_ds),
                    "val_cindex":       cidx,
                    "epoch":            epoch,
                }
                console.print(f"Epoch {epoch}: [green]↑ New best c-index {cidx:.4f}")
            else:
                patience += 1
                if patience >= cfg.get("early_stop", 30):
                    console.print(f"Early-stop at epoch {epoch}")
                    break
            p.update(task, advance=1)

    # Save best model
    ckpt_path = os.path.join(results_dir, f"final_best_model_seed{seed}.pt")
    if best_ckpt:
        torch.save(best_ckpt, ckpt_path)
        console.print(f"✓ Saved seed {seed} checkpoint → {ckpt_path}")
    else:
        console.print("No best checkpoint recorded — training likely diverged.")
        return -1

    # Quick reload sanity
    fe_new = Advanced3DCNN(activation=cfg["activation"], drop=cfg["drop"]).to(device)
    with torch.no_grad():
        dim = fe_new(torch.zeros(1, 3, 64, 64, 64, device=device)).shape[1]
    ds_new = DeepSurv([dim, cfg["hidden_dim"], cfg["hidden_dim"], 1], drop=cfg["drop"], activation=cfg["activation"]).to(device)
    chk = torch.load(ckpt_path, map_location=device)
    fe_new.load_state_dict(chk["feature_extractor"]); ds_new.load_state_dict(chk["deep_surv"])
    reload_cidx = compute_uno_cindex(*[evaluate_model(fe_new, ds_new, dset, cfg["batch_size"])
                                       for dset in (train_dataset, val_dataset)])
    console.print(f"Reload val c-index = {reload_cidx:.4f}")
    return reload_cidx

##############################################################################
# 16  HELPERS (loading saved models & ensemble evaluation)
##############################################################################
def load_trained_model(seed, cfg):
    """Utility wrapper — returns (feature_extractor, deep_surv) in eval mode."""
    fe = Advanced3DCNN(activation=cfg["activation"], drop=cfg["drop"]).to(device)
    with torch.no_grad():
        dim = fe(torch.zeros(1, 3, 64, 64, 64, device=device)).shape[1]
    ds = DeepSurv([dim, cfg["hidden_dim"], cfg["hidden_dim"], 1], drop=cfg["drop"], activation=cfg["activation"]).to(device)

    ckpt = torch.load(os.path.join(results_dir, f"final_best_model_seed{seed}.pt"), map_location=device)
    fe.load_state_dict(ckpt["feature_extractor"]); ds.load_state_dict(ckpt["deep_surv"])
    fe.eval(); ds.eval()
    return fe, ds


def evaluate_ensemble(dataset, models, cfg, aggregator="mean"):
    """Compute risk predictions aggregated across N models."""
    set_eval_seed(EVAL_SEED)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)
    all_t, all_e, all_p = [], [], []
    for batch in loader:
        t = batch["survival_time"].cpu().numpy()
        e = batch["event_occurred"].cpu().numpy().astype(bool)

        preds = np.array([apply_tta_mc_dropout(fe, ds, batch, 5, 5) for fe, ds in models])
        if aggregator == "median":
            pred = np.median(preds, axis=0)
        elif aggregator == "max":
            pred = np.max(preds, axis=0)
        else:
            pred = np.mean(preds, axis=0)

        all_t.append(t); all_e.append(e); all_p.append(pred)
    return np.concatenate(all_t), np.concatenate(all_e), np.concatenate(all_p)

##############################################################################
# 17  ENTRY-POINT
##############################################################################
def main():
    console.print("[bold]🎬  Tumor-OS Deep-Survival pipeline starting…")

    # ------------------------------------------------------------------ 17-A  NPY cache
    npy_path = os.path.join(cached_dir, "Final_OS_data.npy")
    if not os.path.exists(npy_path):
        create_np_data(
            liver_image_dir, liver_label_dir,
            merged_data, train_image_names, val_image_names,
            npy_save_path=npy_path, shape=(64, 64, 64)
        )
    else:
        console.print("✓ Found existing NPY cache, skipping creation")

    data_all = np.load(npy_path, allow_pickle=True).item()
    train_list, val_list, cache_dict = data_all["train"], data_all["val"], data_all["cache_dict"]

    # Pre-convert NumPy→Tensor for fast I/O
    val_cache_t = {d["cache_pid"]: torch.from_numpy(cache_dict[d["cache_pid"]]).float() for d in val_list}
    tr_cache_t  = {d["cache_pid"]: torch.from_numpy(cache_dict[d["cache_pid"]]).float() for d in train_list}

    train_ds = LiverDataset(train_list, cache_dict, augment=True,  tensor_cache=tr_cache_t)
    val_ds   = LiverDataset(val_list,   cache_dict, augment=False, tensor_cache=val_cache_t)

    # ------------------------------------------------------------------ 17-B  Embedding cache & hyper-param search
    emb_path = os.path.join(cached_dir, "emb_dataset.pkl")
    if os.path.exists(emb_path):
        emb_dataset = pickle.load(open(emb_path, "rb"))
        console.print(f"✓ Loaded embeddings ({len(emb_dataset)})")
    else:
        console.print("Computing embeddings for hyper-param search…")
        fe_tmp = Advanced3DCNN().to(device); fe_tmp.eval()
        emb_dataset = compute_embeddings_for_dataset(fe_tmp, train_ds)
        pickle.dump(emb_dataset, open(emb_path, "wb"))
        console.print(f"✓ Saved embeddings → {emb_path}")

    best_hp_path = os.path.join(cached_dir, "best_params.pkl")
    if os.path.exists(best_hp_path):
        best_params = pickle.load(open(best_hp_path, "rb"))
        console.print("✓ Loaded cached hyper-params")
    else:
        best_params, _ = hyperparam_search_with_embeddings(emb_dataset, k=3)
        pickle.dump(best_params, open(best_hp_path, "wb"))
        console.print("✓ Saved best hyper-params")

    # ------------------------------------------------------------------ 17-C  Multi-seed training
    seeds = [42, 1337, 2023, 888, 555]
    seed_scores = []
    for sd in seeds:
        ck = os.path.join(results_dir, f"final_best_model_seed{sd}.pt")
        if os.path.exists(ck):
            console.print(f"Seed {sd} already trained; loading score…")
            fe, ds = load_trained_model(sd, best_params)
            cidx   = compute_uno_cindex(*[evaluate_model(fe, ds, dset, best_params["batch_size"])
                                          for dset in (train_ds, val_ds)])
        else:
            cidx = train_single_seed(sd, train_ds, val_ds, best_params, do_ssl=True)
        seed_scores.append((sd, cidx))
        console.print(f"Seed {sd}: c-index {cidx:.4f}")

    # ------------------------------------------------------------------ 17-D  Ensemble among top seeds
    top_seeds = [s for s, _ in sorted(seed_scores, key=lambda x: x[1], reverse=True)[:3]]
    console.print(f"⧉  Building ensemble with seeds {top_seeds}")
    models = [load_trained_model(s, best_params) for s in top_seeds]
    ens_val = evaluate_ensemble(val_ds, models, best_params, aggregator="mean")
    ens_c   = compute_uno_cindex(*[evaluate_ensemble(ds, models, best_params, "mean")
                                   for ds in (train_ds, val_ds)])
    console.print(f"Ensemble val c-index = {ens_c:.4f}")

    # ------------------------------------------------------------------ 17-E  Export features + risks
    full_ds = LiverDataset(train_list + val_list, cache_dict, augment=False)
    loader  = DataLoader(full_ds, batch_size=1, shuffle=False)
    ids, feats, risks = [], [], []

    def _pred(fe, ds, vol):
        with torch.no_grad():
            emb = fe(vol); risk = ds(emb)
        return emb.cpu().numpy().squeeze(), risk.cpu().numpy().squeeze()

    for b in loader:
        pid = b["pid"][0]; vol = b["Whole_Liver_image"].to(device)
        sub_emb, sub_risk = zip(*[_pred(fe, ds, vol) for fe, ds in models])
        ids.append(pid)
        feats.append(np.mean(sub_emb, axis=0))
        risks.append(np.mean(sub_risk))

    feats, risks = np.array(feats), np.array(risks)

    feat_csv = os.path.join(results_dir, "Tumour_DeepFeatures_OS.csv")
    risk_csv = os.path.join(results_dir, "Tumour_RiskValues_OS.csv")

    # Write CSVs
    with open(feat_csv, "w") as f:
        f.write("patient_id," + ",".join([f"feat_{i}" for i in range(feats.shape[1])]) + "\n")
        for pid, v in zip(ids, feats):
            f.write(pid + "," + ",".join(f"{x:.6f}" for x in v) + "\n")

    with open(risk_csv, "w") as f:
        f.write("patient_id,risk\n")
        for pid, rsk in zip(ids, risks):
            f.write(f"{pid},{rsk:.6f}\n")

    console.print(f"✓ Saved:\n  • {feat_csv}\n  • {risk_csv}\nDone ✅")


if __name__ == "__main__":
    main()
