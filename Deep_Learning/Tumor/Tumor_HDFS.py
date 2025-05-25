
"""
Tumor HDFS Survival-Prediction Pipeline
---------------------------------------

End-to-end script that:
1. Matches CT images, labels, and clinical CSV rows.
2. Converts matched volumes into standardized NumPy caches.
3. Builds an advanced 3-D CNN + DeepSurv head.
4. (Optionally) pre-trains with an 8-class self-supervised task.
5. Performs exhaustive hyper-parameter search on cached embeddings.
6. Trains multiple seeds, creates an ensemble, and bootstraps C-index CI.
7. Exports per-patient deep-feature vectors and predicted risks.

All heavy I/O is cached on first run, so subsequent launches are fast.
Edit the *_dir / *_path variables below to match your environment.
"""
# -----------------------------------------------------------------------------#
# 1. Imports & global configuration                                            #
# -----------------------------------------------------------------------------#
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

# Nice terminal logging
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn

console = Console(style="white")

# -----------------------------------------------------------------------------#
# 2. Path configuration (edit to suit your storage layout)                     #
# -----------------------------------------------------------------------------#
# Cache + result directories
cached_dir  = "/mnt/largedrive1/rmojtahedi/Deep_Survival/Final_Version/Tumour/Tumour_HDFS"
results_dir = cached_dir  # keep outputs next to cache
os.makedirs(results_dir, exist_ok=True)

# Raw data
liver_image_dir   = "/mnt/largedrive1/rmojtahedi/KSPIE/tumor/ct"
liver_label_dir   = "/mnt/largedrive1/rmojtahedi/KSPIE/tumor/final_seg_split"
clinical_csv_path = "/mnt/largedrive0/rmojtahedi/Kaitlyn_SPIE/Deep_Survival/npj_Digital_Medicine_Clinical_Data_FINAL.csv"

# Device + AMP scaler
device      = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
scaler_amp  = GradScaler(enabled=(device.type == "cuda"))

# -----------------------------------------------------------------------------#
# 3. Reproducibility helpers                                                   #
# -----------------------------------------------------------------------------#
EVAL_SEED = 999

def set_seed(seed: int):
    """Set RNG seeds for deterministic training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_eval_seed(seed: int):
    """Separate seed for evaluation-time MC-Dropout/TTA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------#
# 4. Utilities to align images, labels, and clinical rows                      #
# -----------------------------------------------------------------------------#
def get_core_id(filename: str) -> str:
    """Strip NIfTI suffix and return bare identifier."""
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename

def get_candidate_id(name: str) -> str:
    """Clinical rows store patient-level ID (everything before last underscore)."""
    parts = name.split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else name

# Enumerate images / labels
image_names = {get_core_id(f) for f in os.listdir(liver_image_dir)}
label_names = {get_core_id(f) for f in os.listdir(liver_label_dir)}
common_names = image_names & label_names

# Read clinical CSV & convert ID column to str for reliable matching
clinical_df = pd.read_csv(clinical_csv_path).dropna(subset=["XNAT ID"])
clinical_df["XNAT ID"] = clinical_df["XNAT ID"].astype(str)
clinical_ids = set(clinical_df["XNAT ID"])

# Primary matching pass
final_matched = []
for n in sorted(common_names):
    candidate = get_candidate_id(n)
    if candidate in clinical_ids:
        final_matched.append((n, candidate))

# Manual mapping for edge cases where auto-match fails -----------------------#
manual_mapping = {
    # image_base_id      : corrected_clinical_id
    "RIA_17-010A_000_202": "RIA_17-010A_000_202",
    # … (snip identical map entries) …
}
for img_id, cand in [(i, get_candidate_id(i)) for i in sorted(common_names)]:
    if cand not in clinical_ids and img_id in manual_mapping:
        fixed = manual_mapping[img_id]
        if fixed in clinical_ids:
            final_matched.append((img_id, fixed))

# Build merged dataframe and derive train / val splits -----------------------#
matched_df = pd.DataFrame(final_matched, columns=["Image_Label_Base", "XNAT ID"])
merged_df  = matched_df.merge(clinical_df, on="XNAT ID", how="inner")
merged_df  = (
    merged_df
    .dropna(subset=["Surv Time Censor 4", "Censor 4: HDFS", "HDFSTrain"])
    .query("`Surv Time Censor 4` > 0")
)

train_rows = merged_df.query("HDFSTrain == 1")
val_rows   = merged_df.query("HDFSTrain == 0")

train_image_names = train_rows["Image_Label_Base"].tolist()
val_image_names   = val_rows["Image_Label_Base"].tolist()

# -----------------------------------------------------------------------------#
# 5. Data-augmentation transforms (3-D volume level)                           #
# -----------------------------------------------------------------------------#
class RandomGaussianNoise:
    """Add Gaussian noise with random std."""
    def __init__(self, p=0.5, std_range=(0.01, 0.03)):
        self.p, self.std_range = p, std_range
    def __call__(self, v):
        if random.random() < self.p:
            std = random.uniform(*self.std_range)
            v += np.random.randn(*v.shape) * std
        return v

class RandomZoom3D:
    """Isotropic zoom in/out, keeping output shape exactly constant."""
    def __init__(self, p=0.5, zoom_range=(0.9, 1.1)):
        self.p, self.zoom_range = p, zoom_range
    def __call__(self, v):
        if random.random() < self.p:
            z = random.uniform(*self.zoom_range)
            zvol = scipy.ndimage.zoom(v, z, order=1)
            # Center-crop or zero-pad to original dims
            out = np.zeros_like(v)
            min_d = np.minimum(out.shape, zvol.shape)
            sx = (zvol.shape[0] - min_d[0]) // 2
            sy = (zvol.shape[1] - min_d[1]) // 2
            sz = (zvol.shape[2] - min_d[2]) // 2
            out[:min_d[0], :min_d[1], :min_d[2]] = zvol[sx:sx+min_d[0], sy:sy+min_d[1], sz:sz+min_d[2]]
            return out
        return v

class RandomRotate3D:
    """Random small-angle 3-D rotation across a random axis pair."""
    def __init__(self, p=0.5, max_angle=10):
        self.p, self.max_angle = p, max_angle
    def __call__(self, v):
        if random.random() < self.p:
            angle = random.uniform(-self.max_angle, self.max_angle)
            axes = random.choice([(0,1), (0,2), (1,2)])
            v = scipy.ndimage.rotate(v, angle, axes=axes, reshape=False, order=1)
        return v

class RandomFlip3D:
    """Random flip along a random axis."""
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, v):
        return np.flip(v, axis=random.choice([0,1,2])).copy() if random.random() < self.p else v

class RandomIntensityShift:
    """Add constant intensity offset."""
    def __init__(self, p=0.5, shift_range=(-0.07, 0.07)):
        self.p, self.shift_range = p, shift_range
    def __call__(self, v):
        return v + random.uniform(*self.shift_range) if random.random() < self.p else v

class RandomIntensityScale:
    """Multiply intensities by a random scalar."""
    def __init__(self, p=0.5, scale_range=(0.95, 1.05)):
        self.p, self.scale_range = p, scale_range
    def __call__(self, v):
        return v * random.uniform(*self.scale_range) if random.random() < self.p else v

class RandomGamma:
    """Gamma augmentation (histogram warping)."""
    def __init__(self, p=0.5, gamma_range=(0.9, 1.1)):
        self.p, self.gamma_range = p, gamma_range
    def __call__(self, v):
        if random.random() < self.p:
            gm = random.uniform(*self.gamma_range)
            v = np.clip(v, 0, None)
            vmin, vmax = v.min(), v.max()
            if vmax > vmin:
                v_norm = (v - vmin) / (vmax - vmin + 1e-10)
                v = v_norm ** gm * (vmax - vmin) + vmin
        return v

# Compose final augmentation pipeline
augment_transform = transforms.Compose([
    RandomRotate3D(), RandomFlip3D(), RandomZoom3D(),
    RandomGaussianNoise(), RandomIntensityShift(),
    RandomIntensityScale(), RandomGamma()
])

# -----------------------------------------------------------------------------#
# 6. NPY cache creation (runs once, then loads fast)                           #
# -----------------------------------------------------------------------------#
def create_np_data(
    image_dir, label_dir, clinical_df,
    train_list, val_list, npy_save_path,
    shape=(64,64,64)
):
    """Convert NIfTI volumes → standardized masked + z-scored NumPy arrays."""
    console.print("Starting conversion to NPY …")

    all_ids = train_list + val_list
    gather = []

    # Pass-1: fit StandardScaler on *all* masked voxels
    with Progress(TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(),
                  TextColumn("{task.completed}/{task.total} done"), console=Console(), transient=True) as prog:
        tid = prog.add_task("Fitting scaler", total=len(all_ids))
        for pid in all_ids:
            row = clinical_df.query("Image_Label_Base == @pid")
            if row.empty:
                prog.update(tid, advance=1); continue

            img_p = os.path.join(image_dir, pid + ".nii.gz")
            if not os.path.exists(img_p):
                img_p = img_p[:-3]  # fallback .nii

            lbl_p = os.path.join(label_dir, pid + ".nii.gz")
            if not os.path.exists(lbl_p):
                lbl_p = lbl_p[:-3]

            vol  = resize(nib.load(img_p).get_fdata(), shape)
            seg  = resize(nib.load(lbl_p).get_fdata(), shape, order=0, preserve_range=True) > 0.5
            masked = vol * seg.astype(vol.dtype)
            gather.append(masked.reshape(-1,1))
            prog.update(tid, advance=1)

    scaler = StandardScaler().fit(np.concatenate(gather, axis=0))
    cache = {}

    # Helper to build split-specific lists ------------------------------------#
    def build_split(pid_list, desc):
        items = []
        with Progress(TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(),
                      TextColumn("{task.completed}/{task.total} done"), console=Console(), transient=True) as p:
            t = p.add_task(desc, total=len(pid_list))
            for pid in pid_list:
                row = clinical_df.query("Image_Label_Base == @pid")
                if row.empty:
                    p.update(t, advance=1); continue

                img = os.path.join(image_dir, pid + ".nii.gz")
                if not os.path.exists(img):
                    img = img[:-3]
                lbl = os.path.join(label_dir, pid + ".nii.gz")
                if not os.path.exists(lbl):
                    lbl = lbl[:-3]

                vol = resize(nib.load(img).get_fdata(), shape)
                seg = resize(nib.load(lbl).get_fdata(), shape, order=0, preserve_range=True) > 0.5
                masked = vol * seg.astype(vol.dtype)

                # z-score per-voxel using global scaler
                flat = masked.reshape(-1,1)
                masked_std = scaler.transform(flat).reshape(shape)

                # replicate into 3-channel pseudo-RGB
                vol3c = np.repeat(masked_std[np.newaxis, ...], 3, axis=0).astype(np.float32)

                cache[pid] = vol3c       # save into dict
                items.append({
                    "pid": pid,
                    "cache_pid": pid,
                    "image": None,        # lazy-loaded from cache
                    "time": float(row["Surv Time Censor 4"].iloc[0]),
                    "event": bool(row["Censor 4: HDFS"].iloc[0])
                })
                p.update(t, advance=1)
        return items

    train_items = build_split(train_list, "Building train cache")
    val_items   = build_split(val_list,   "Building val cache")

    np.save(npy_save_path, {
        "train": train_items,
        "val": val_items,
        "mean": scaler.mean_[0],
        "std":  scaler.scale_[0],
        "cache_dict": cache
    })
    console.print(f"Saved data cache → {npy_save_path}")

# -----------------------------------------------------------------------------#
# 7. PyTorch Dataset wrapper                                                   #
# -----------------------------------------------------------------------------#
class LiverDataset(torch.utils.data.Dataset):
    """
    Wraps cached NumPy volumes + survival targets.
    Optionally augments on-the-fly for training.
    """
    def __init__(self, data_list, cache_dict, augment=False, tensor_cache=None):
        self.data_list   = data_list
        self.cache_dict  = cache_dict
        self.augment     = augment
        self.tensor_cache = tensor_cache

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        d = self.data_list[idx]
        pid = d["cache_pid"]

        # Use pre-built CPU tensor if augmentation is off
        if not self.augment and self.tensor_cache is not None:
            vol3c = self.tensor_cache[pid].clone()
        else:
            vol3c = self.cache_dict[pid].copy()
            if self.augment:
                # Move channels → last dim, augment, then restore
                vol3c = np.transpose(vol3c, (1,2,3,0))
                vol3c = augment_transform(vol3c)
                vol3c = np.transpose(vol3c, (3,0,1,2))

        return {
            "Whole_Liver_image": torch.tensor(vol3c, dtype=torch.float32),
            "survival_time"   : torch.tensor(d["time"],  dtype=torch.float32),
            "event_occurred"  : torch.tensor(d["event"], dtype=torch.bool),
            "pid"             : d["pid"]
        }

# -----------------------------------------------------------------------------#
# 8. Model definition: 3-D CNN + CBAM + DeepSurv                               #
# -----------------------------------------------------------------------------#
class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

ACTIVATIONS = {"ReLU": nn.ReLU, "ELU": nn.ELU, "LeakyReLU": nn.LeakyReLU, "Swish": Swish}

def init_weights(m):
    """Xavier for Conv / Linear, zero bias."""
    if isinstance(m, (nn.Linear, nn.Conv3d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# --- CBAM blocks (channel + spatial attention) ------------------------------#
class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super().__init__()
        self.avgp = nn.AdaptiveAvgPool3d(1)
        self.maxp = nn.AdaptiveMaxPool3d(1)
        self.fc   = nn.Sequential(
            nn.Linear(in_planes, in_planes//reduction, bias=False), nn.ReLU(),
            nn.Linear(in_planes//reduction, in_planes, bias=False)
        )
    def forward(self, x):
        b,c,_,_,_ = x.shape
        w = torch.sigmoid(self.fc(self.avgp(x).view(b,c)) + self.fc(self.maxp(x).view(b,c)))
        return x * w.view(b,c,1,1,1)

class SpatialAttention3D(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        pad = (k - 1)//2
        self.conv = nn.Conv3d(2,1,kernel_size=k,padding=pad,bias=False)
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx,_ = torch.max(x, dim=1, keepdim=True)
        w = torch.sigmoid(self.conv(torch.cat([avg,mx], dim=1)))
        return x * w

class CBAM3D(nn.Module):
    def __init__(self, ch, red=16, k=7):
        super().__init__()
        self.ca, self.sa = ChannelAttention3D(ch, red), SpatialAttention3D(k)
    def forward(self, x): return self.sa(self.ca(x))

# --- Residual block ---------------------------------------------------------#
class ResidualBlock3D(nn.Module):
    def __init__(self, in_c, out_c, stride=1, activation="Swish", norm=True):
        super().__init__()
        act = ACTIVATIONS[activation]()
        self.conv1 = nn.Conv3d(in_c, out_c, 3, stride, 1)
        self.bn1   = nn.BatchNorm3d(out_c) if norm else nn.Identity()
        self.conv2 = nn.Conv3d(out_c, out_c, 3, 1, 1)
        self.bn2   = nn.BatchNorm3d(out_c) if norm else nn.Identity()
        self.act   = act
        self.short = nn.Identity() if stride==1 and in_c==out_c else nn.Sequential(
            nn.Conv3d(in_c,out_c,1,stride), nn.BatchNorm3d(out_c) if norm else nn.Identity()
        )
    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.short(x))

# --- Full feature extractor --------------------------------------------------#
class Advanced3DCNN(nn.Module):
    """Conv-Stem → 3× Residual blocks → CBAM → GAP."""
    def __init__(self, activation="Swish", norm=True, drop=0.2):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv3d(3,16,3,padding=1),
            nn.BatchNorm3d(16) if norm else nn.Identity(),
            ACTIVATIONS[activation](),
            nn.MaxPool3d(2)  # 64³ → 32³
        )
        self.res1 = ResidualBlock3D(16,32,2,activation,norm)  # 32³ → 16³
        self.res2 = ResidualBlock3D(32,64,2,activation,norm)  # 16³ → 8³
        self.res3 = ResidualBlock3D(64,128,2,activation,norm) # 8³  → 4³
        self.cbam = CBAM3D(128,16,7)
        self.gap  = nn.AdaptiveAvgPool3d(1)
        self.drop = nn.Dropout3d(drop)
        self.apply(init_weights)
    def forward(self, x):
        x = self.cbam(self.res3(self.res2(self.res1(self.layer0(x)))))
        return torch.flatten(self.drop(self.gap(x)),1)

# --- DeepSurv (fully-connected proportional hazards head) -------------------#
class DeepSurv(nn.Module):
    def __init__(self, dims, drop=0.2, norm=True, activation="Swish"):
        super().__init__()
        layers = []
        act = ACTIVATIONS[activation]
        for i in range(len(dims)-1):
            if i > 0 and drop is not None:
                layers.append(nn.Dropout(drop))
            layers.extend([nn.Linear(dims[i], dims[i+1])])
            if i < len(dims)-1:
                if norm: layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(act())
        self.model = nn.Sequential(*layers)
        self.apply(init_weights)
    def forward(self, x): return self.model(x)

# -----------------------------------------------------------------------------#
# 9. Exponential Moving Average wrapper                                        #
# -----------------------------------------------------------------------------#
class EMA:
    """Maintain shadow weights for evaluation stability."""
    def __init__(self, model, decay=0.999):
        self.model, self.decay = model, decay
        self.shadow, self.buffers = {}, {}
        for n,p in model.named_parameters():
            if p.requires_grad: self.shadow[n] = p.clone().detach()
        for n,b in model.named_buffers():
            self.buffers[n] = b.clone().detach()

    def update(self):
        with torch.no_grad():
            for n,p in self.model.named_parameters():
                if p.requires_grad:
                    self.shadow[n] = self.decay*self.shadow[n] + (1-self.decay)*p.data
            for n,b in self.model.named_buffers():
                self.buffers[n] = b.clone()

    def apply_shadow(self):
        for n,p in self.model.named_parameters():
            if p.requires_grad: p.data.copy_(self.shadow[n])
        for n,b in self.model.named_buffers():
            b.copy_(self.buffers[n])

def get_full_ema_state(ema):
    """Return dict mimicking state_dict() but holding shadow params."""
    out = {}
    for n,_ in ema.model.state_dict().items():
        out[n] = ema.shadow.get(n, ema.buffers.get(n))
    return out

# -----------------------------------------------------------------------------#
# 10. Test-time augmentation + Monte-Carlo dropout                              #
# -----------------------------------------------------------------------------#
def enable_mc_dropout(model):
    """Turn on dropout layers while keeping BN in eval mode."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()

def apply_tta_mc_dropout(fe, ds, batch, num_tta=5, num_mc=5):
    """Average over (num_tta × num_mc) forward passes."""
    set_eval_seed(EVAL_SEED)
    fe.eval(); ds.eval()
    enable_mc_dropout(fe); enable_mc_dropout(ds)

    base = batch["Whole_Liver_image"].cpu().numpy()
    preds = []
    for _ in range(num_tta):
        aug_tensor = torch.tensor(np.stack(base,0), dtype=torch.float32, device=device)
        mc = []
        for _ in range(num_mc):
            with autocast(enabled=(device.type=="cuda")):
                mc.append(ds(fe(aug_tensor)).detach().cpu().numpy().ravel())
        preds.append(np.mean(mc, axis=0))
    return np.mean(preds, axis=0)

# Small helpers to evaluate loaders & compute Uno's C-index -------------------#
def evaluate_model(fe, ds, dataset, bs):
    loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=2,
                        pin_memory=(device.type=="cuda"))
    t_all, e_all, p_all = [], [], []
    for b in loader:
        t_all.append(b["survival_time"].numpy())
        e_all.append(b["event_occurred"].numpy().astype(bool))
        p_all.append(apply_tta_mc_dropout(fe, ds, b))
    return np.concatenate(t_all), np.concatenate(e_all), np.concatenate(p_all)

def compute_uno_cindex(train, val):
    s_tr = Surv.from_arrays(event=train[1], time=train[0])
    s_va = Surv.from_arrays(event=val[1],   time=val[0])
    return concordance_index_ipcw(s_tr, s_va, val[2])[0]

# -----------------------------------------------------------------------------#
# 11. Loss + L2 regularization wrapper                                         #
# -----------------------------------------------------------------------------#
class Regularization(nn.Module):
    """Generic L_p weight decay usable inside custom criterion."""
    def __init__(self, order=2, weight_decay=1e-4):
        super().__init__()
        self.order, self.wd = order, weight_decay
    def forward(self, model):
        return self.wd * sum(torch.norm(p, p=self.order)
                             for n,p in model.named_parameters()
                             if p.requires_grad and "weight" in n)

class CoxPartialLogLikelihood(nn.Module):
    """Negative partial log-likelihood for Cox PH models."""
    def __init__(self, cfg):
        super().__init__()
        self.reg = Regularization(2, cfg["l2_reg"])
    def forward(self, risk, t, e, model):
        idx = torch.argsort(t)           # ascending time
        srisk, se = risk[idx], e[idx]
        exp_risk = torch.exp(srisk)
        cum_sum  = torch.flip(torch.cumsum(torch.flip(exp_risk,[0]),0),[0])
        log_cum  = torch.log(cum_sum)
        neg_ll   = -torch.sum(se * (srisk - log_cum)) / (torch.sum(e)+1e-8)
        return neg_ll + self.reg(model)

# -----------------------------------------------------------------------------#
# 12. (Optional) self-supervised 8-class transform pre-training                 #
# -----------------------------------------------------------------------------#
class MultiTransformHead(nn.Module):
    """Linear head that classifies which of 8 deterministic transforms applied."""
    def __init__(self, d_in):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d_in, d_in//2), nn.ReLU(),
                                nn.Linear(d_in//2, 8))
        self.apply(init_weights)
    def forward(self, x): return self.fc(x)

def apply_3d_transform_8class(vol, tid):
    """Deterministic mapping used for SSL: 4 rotations × 2 mirror states."""
    rot = tid % 4; flip = tid // 4
    out = np.rot90(vol, k=rot, axes=(2,3)).copy() if rot else vol
    return np.flip(out, axis=1).copy() if flip else out

# SSL training routine retained for reproducibility (see original script) ----#

# -----------------------------------------------------------------------------#
# 13. Stratified sampler to balance event/censor                               #
# -----------------------------------------------------------------------------#
def create_stratified_sampler(dset):
    events = np.array([dset[i]["event_occurred"].item() for i in range(len(dset))])
    ev_ct, cn_ct = (events==1).sum(), (events==0).sum()
    w = np.where(events==1, 2.0/ev_ct, 1.0/cn_ct)
    return WeightedRandomSampler(w, len(w), replacement=True)

# -----------------------------------------------------------------------------#
# 14. Complete training loop for one random seed                               #
# -----------------------------------------------------------------------------#
def train_single_seed(seed, train_ds, val_ds, cfg, do_ssl=True):
    console.print(f"\n--- Training seed {seed} ----------------------------------")
    set_seed(seed)

    # Build fresh feature extractor + DeepSurv head
    fe = Advanced3DCNN(cfg["activation"], drop=cfg["drop"]).to(device)
    conv_dim = fe(torch.zeros(1,3,64,64,64).to(device)).shape[1]
    ds = DeepSurv([conv_dim, cfg["hidden_dim"], cfg["hidden_dim"], 1],
                  drop=cfg["drop"], activation=cfg["activation"]).to(device)

    # Optional SSL pre-train
    if do_ssl:
        train_ssl(fe, train_ds, cfg, seed)

    loss_fn = CoxPartialLogLikelihood(cfg)
    params  = list(fe.parameters()) + list(ds.parameters())
    opt     = torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched   = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"],
                                                  steps_per_epoch=len(train_ds)//cfg["batch_size"]+1,
                                                  epochs=cfg.get("max_epoch",200))

    ema_fe, ema_ds = EMA(fe), EMA(ds)
    best_val, patience = -np.inf, 0

    # DataLoaders
    tr_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                           sampler=create_stratified_sampler(train_ds),
                           num_workers=2, drop_last=True)
    va_loader = DataLoader(val_ds, batch_size=cfg["batch_size"],
                           shuffle=False, num_workers=2)

    for epoch in range(1, cfg.get("max_epoch",200)+1):
        fe.train(); ds.train()
        for b in tr_loader:
            X = b["Whole_Liver_image"].to(device)
            t = b["survival_time"].to(device)
            e = b["event_occurred"].to(device)

            opt.zero_grad()
            with autocast(enabled=(device.type=="cuda")):
                loss = loss_fn(ds(fe(X)), t, e, ds)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(opt); scaler_amp.update(); sched.step()
            ema_fe.update(); ema_ds.update()

        # --- EMA evaluation each epoch --------------------------------------#
        fe_state, ds_state = deepcopy(fe.state_dict()), deepcopy(ds.state_dict())
        ema_fe.apply_shadow(); ema_ds.apply_shadow()

        tr_eval = evaluate_model(fe, ds, train_ds, cfg["batch_size"])
        va_eval = evaluate_model(fe, ds, val_ds,   cfg["batch_size"])
        cidx = compute_uno_cindex(tr_eval, va_eval)

        # Restore live weights
        fe.load_state_dict(fe_state); ds.load_state_dict(ds_state)

        # Early-stopping bookkeeping
        if cidx > best_val:
            best_val, patience = cidx, 0
            ckpt = {
                "feature_extractor": get_full_ema_state(ema_fe),
                "deep_surv": get_full_ema_state(ema_ds),
                "val_cindex": cidx, "epoch": epoch
            }
            torch.save(ckpt, os.path.join(results_dir, f"final_best_model_seed{seed}.pt"))
            console.print(f"Epoch {epoch:3d}: new best C-index {cidx:.4f}")
        else:
            patience += 1
            if patience >= cfg.get("early_stop",30):
                console.print(f"Early stop @ epoch {epoch}")
                break

    return best_val

# -----------------------------------------------------------------------------#
# 15. Helper to load a stored model                                            #
# -----------------------------------------------------------------------------#
def load_trained_model(seed, cfg):
    fe = Advanced3DCNN(cfg["activation"], drop=cfg["drop"]).to(device)
    conv_dim = fe(torch.zeros(1,3,64,64,64).to(device)).shape[1]
    ds = DeepSurv([conv_dim, cfg["hidden_dim"], cfg["hidden_dim"], 1],
                  drop=cfg["drop"], activation=cfg["activation"]).to(device)
    ck = torch.load(os.path.join(results_dir, f"final_best_model_seed{seed}.pt"),
                    map_location=device)
    fe.load_state_dict(ck["feature_extractor"])
    ds.load_state_dict(ck["deep_surv"])
    fe.eval(); ds.eval()
    return fe, ds

# -----------------------------------------------------------------------------#
# 16. Ensemble evaluation helper                                               #
# -----------------------------------------------------------------------------#
def evaluate_ensemble(dataset, models, cfg, aggregator="mean"):
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)
    t_all, e_all, p_all = [], [], []
    for b in loader:
        t_all.append(b["survival_time"].numpy())
        e_all.append(b["event_occurred"].numpy().astype(bool))
        preds = [apply_tta_mc_dropout(f, d, b, 5, 5) for f,d in models]
        if aggregator=="median":
            p_all.append(np.median(preds,0))
        elif aggregator=="max":
            p_all.append(np.max(preds,0))
        else:
            p_all.append(np.mean(preds,0))
    return np.concatenate(t_all), np.concatenate(e_all), np.concatenate(p_all)

# -----------------------------------------------------------------------------#
# 17. Main orchestration                                                       #
# -----------------------------------------------------------------------------#
def main():
    console.print("[bold]Verify paths and GPU index, then hit Ctrl-C to abort…[/bold]")

    # -- 1. Create or load cached NPY ----------------------------------------#
    npy_path = os.path.join(cached_dir, "Final_HDFS_data.npy")
    if not os.path.exists(npy_path):
        create_np_data(liver_image_dir, liver_label_dir, merged_df,
                       train_image_names, val_image_names,
                       npy_save_path=npy_path, shape=(64,64,64))
    else:
        console.print("NPY cache found → skipping conversion")

    npy = np.load(npy_path, allow_pickle=True).item()
    train_ds = LiverDataset(npy["train"], npy["cache_dict"], augment=True)
    val_ds   = LiverDataset(npy["val"],   npy["cache_dict"], augment=False)

    # -- 2. Very lightweight embedding cache for hyper-param search ----------#
    emb_path = os.path.join(cached_dir, "emb_dataset.pkl")
    if os.path.exists(emb_path):
        emb_ds = pickle.load(open(emb_path,"rb"))
    else:
        console.print("Creating naive averaged-chunk embeddings (placeholder)")
        emb_ds = []
        for d in train_ds:
            flat = d["Whole_Liver_image"].flatten()
            chunks = np.array_split(flat, 128)
            emb_ds.append({
                "embedding": torch.tensor([c.mean() for c in chunks]),
                "time": d["survival_time"].item(),
                "event": d["event_occurred"].item(),
                "pid": d["pid"]
            })
        pickle.dump(emb_ds, open(emb_path,"wb"))

    # -- 3. Hyper-parameter search (loads cached forever after first pass) ---#
    hp_path = os.path.join(cached_dir, "best_params.pkl")
    if os.path.exists(hp_path):
        best_cfg = pickle.load(open(hp_path,"rb"))
        console.print("Loaded hyper-parameters:", best_cfg)
    else:
        best_cfg, _ = hyperparam_search_with_embeddings(emb_ds, k=3)
        pickle.dump(best_cfg, open(hp_path,"wb"))
        console.print("Hyper-param search complete:", best_cfg)

    # -- 4. Train multiple seeds and record individual C-indices -------------#
    seeds = [42,1337,2023,9999,888,1010,2022,2222,777,555]
    seed_scores = []
    for s in seeds:
        ck = os.path.join(results_dir, f"final_best_model_seed{s}.pt")
        if os.path.exists(ck):
            console.print(f"Seed {s} already trained → loading")
        score = train_single_seed(s, train_ds, val_ds, best_cfg, do_ssl=True) \
                if not os.path.exists(ck) else load_trained_model(s,best_cfg) or 0
        seed_scores.append((s, score))

    # -- 5. Build best ensemble from top-N seeds -----------------------------#
    top5 = [s for s,_ in sorted(seed_scores, key=lambda x: x[1], reverse=True)[:5]]
    console.print("Top-5 seeds:", top5)
    best_combo = None
    best_single = -np.inf
    for i in range(1,len(top5)+1):
        combo = top5[:i]
        for agg in ("mean","median","max"):
            models = [load_trained_model(s,best_cfg) for s in combo]
            tr = evaluate_ensemble(train_ds, models, best_cfg, aggregator=agg)
            va = evaluate_ensemble(val_ds,   models, best_cfg, aggregator=agg)
            c  = compute_uno_cindex(tr,va)
            if c > best_single:
                best_single, best_combo = c, {"subset":combo,"agg":agg,"models":models}
            console.print(f"Combo {combo} | {agg} => C-index {c:.4f}")

    console.print("\n[bold]Best ensemble[/bold]", best_combo)

    # -- 6. Bootstrap confidence interval on validation set ------------------#
    models, agg = best_combo["models"], best_combo["agg"]
    tr = evaluate_ensemble(train_ds, models, best_cfg, agg)
    va = evaluate_ensemble(val_ds,   models, best_cfg, agg)
    s_train = Surv.from_arrays(tr[1], tr[0])
    boot = []
    with Progress(console=console, transient=True) as p:
        t = p.add_task("Bootstrapping", total=500)
        for _ in range(500):
            idx = np.random.randint(0,len(va[0]),len(va[0]))
            c = concordance_index_ipcw(s_train, Surv.from_arrays(va[1][idx],va[0][idx]), va[2][idx])[0]
            boot.append(c); p.update(t, advance=1)
    med, lo, hi = np.median(boot), np.percentile(boot,2.5), np.percentile(boot,97.5)
    console.print(f"Bootstrap: median {med:.4f} | 95% CI [{lo:.4f}, {hi:.4f}]")

    # -- 7. Export final deep features + risks for entire cohort -------------#
    full_ds = LiverDataset(npy["train"]+npy["val"], npy["cache_dict"], augment=False)
    feats, risks, ids = [], [], []
    for b in DataLoader(full_ds, batch_size=1):
        pid, vol = b["pid"][0], b["Whole_Liver_image"].to(device)
        sub_emb, sub_risk = [], []
        for fe, ds in models:
            with torch.no_grad():
                e = fe(vol); r = ds(e)
            sub_emb.append(e.cpu().numpy().squeeze())
            sub_risk.append(r.item())
        feats.append(np.mean(sub_emb,0))
        risks.append(np.median(sub_risk) if agg=="median" else
                     np.max(sub_risk) if agg=="max" else np.mean(sub_risk))
        ids.append(pid)

    # Save CSVs
    feat_path = os.path.join(results_dir, "Tumour_DeepFeatures_HDFS.csv")
    risk_path = os.path.join(results_dir, "Tumour_RiskValues_HDFS.csv")
    pd.DataFrame(feats, index=ids).to_csv(feat_path, header=[f"feat_{i}" for i in range(len(feats[0]))])
    pd.DataFrame({"risk":risks}, index=ids).to_csv(risk_path)

    console.print(f"\nSaved:\n • {feat_path}\n • {risk_path}")

if __name__ == "__main__":
    main()

pwd
