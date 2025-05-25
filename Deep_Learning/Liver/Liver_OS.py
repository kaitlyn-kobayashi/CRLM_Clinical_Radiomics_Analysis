
##############################################################################
# Deep-Survival Liver-OS Pipeline
##############################################################################

# ============================ Imports & Warnings =========================== #
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)     # <<< tidy console >>>
warnings.filterwarnings("ignore", category=FutureWarning)   # <<< tidy console >>>

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

# ----- Mixed precision helpers ------------------------------------------------
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torchvision import transforms
import nibabel as nib

# ----- Rich for pretty terminal progress bars ---------------------------------
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn

console = Console(style="white")

##############################################################################
#                               Path Settings                                #
##############################################################################
cached_dir  = "/mnt/largedrive1/rmojtahedi/Deep_Survival/Final_Version/Liver/Liver_OS"
results_dir = "/mnt/largedrive1/rmojtahedi/Deep_Survival/Final_Version/Liver/Liver_OS"
os.makedirs(results_dir, exist_ok=True)

liver_image_dir   = "/mnt/largedrive1/rmojtahedi/KSPIE/liver/ct"
liver_label_dir   = "/mnt/largedrive1/rmojtahedi/KSPIE/liver/final_seg_split"
clinical_csv_path = "/mnt/largedrive0/rmojtahedi/Kaitlyn_SPIE/Deep_Survival/npj_Digital_Medicine_Clinical_Data_FINAL.csv"

device     = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
scaler_amp = GradScaler(enabled=(device.type == 'cuda'))

##############################################################################
#                             Reproducibility                                #
##############################################################################
EVAL_SEED = 999

def set_seed(seed):
    """Use for training loops (affects CUDA & CPU)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def set_eval_seed(seed):
    """Use inside deterministic eval (TTA / MC dropout)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

##############################################################################
#                    Match Image / Label ↔ Clinical CSV                      #
##############################################################################
def get_core_id(filename):
    if filename.endswith(".nii.gz"): return filename[:-7]
    if filename.endswith(".nii"):    return filename[:-4]
    return filename

def get_candidate_id(name):  # XNAT IDs don’t include final slice token
    parts = name.split('_')
    return '_'.join(parts[:-1]) if len(parts) > 1 else name

# ----- Build sets -------------------------------------------------------------
image_files = os.listdir(liver_image_dir)
label_files = os.listdir(liver_label_dir)

image_names = {get_core_id(fn) for fn in image_files}
label_names = {get_core_id(fn) for fn in label_files}

unmatched_images = sorted(list(image_names - label_names))
unmatched_labels = sorted(list(label_names - image_names))

common_names    = image_names.intersection(label_names)
candidate_pairs = [(n, get_candidate_id(n)) for n in sorted(common_names)]

# ----- Clinical CSV filtering -------------------------------------------------
clinical_data_full          = pd.read_csv(clinical_csv_path)
clinical_data_full          = clinical_data_full.dropna(subset=["XNAT ID"])
clinical_data_full["XNAT ID"]= clinical_data_full["XNAT ID"].astype(str)
clinical_ids                = set(clinical_data_full["XNAT ID"])

# ----- Automatic & manual matching -------------------------------------------
final_matched, unmatched_candidates = [], []
for img_lbl, cand in candidate_pairs:
    (final_matched if cand in clinical_ids else unmatched_candidates).append((img_lbl, cand))

manual_mapping = {  # study-specific weird IDs
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

manually_matched, still_unmatched = [], []
for img_lbl, cand in unmatched_candidates:
    if img_lbl in manual_mapping:
        corr = manual_mapping[img_lbl]
        (manually_matched if corr in clinical_ids else still_unmatched).append((img_lbl, corr))
    else:
        still_unmatched.append((img_lbl, cand))

final_matched.extend(manually_matched)
matched_xnat_set      = {cid for _, cid in final_matched}
unmatched_clinical_ids= sorted(list(clinical_ids - matched_xnat_set))

matched_df  = pd.DataFrame(final_matched, columns=["Image_Label_Base", "XNAT ID"])
merged_data = pd.merge(matched_df, clinical_data_full, on="XNAT ID", how="inner")
merged_data = merged_data.dropna(subset=["Surv Time Censor 7", "Censor 7: OS", "OSTrain"])
merged_data = merged_data[merged_data["Surv Time Censor 7"] > 0]

train_rows = merged_data[merged_data["OSTrain"] == 1]
val_rows   = merged_data[merged_data["OSTrain"] == 0]

train_image_names = train_rows["Image_Label_Base"].tolist()
val_image_names   = val_rows["Image_Label_Base"].tolist()

##############################################################################
#                             Data Augmentation                              #
##############################################################################
class RandomGaussianNoise:         # remaining transform classes unchanged
    def __init__(self, p=0.5, std_range=(0.01, 0.03)): self.p=p; self.std_range=std_range
    def __call__(self, volume):
        if random.random()<self.p:
            volume += np.random.randn(*volume.shape)*random.uniform(*self.std_range)
        return volume
class RandomZoom3D:
    def __init__(self,p=0.5,zoom_range=(0.9,1.1)): self.p=p; self.zoom_range=zoom_range
    def __call__(self,volume):
        if random.random()<self.p:
            zf=random.uniform(*self.zoom_range)
            zoomed=scipy.ndimage.zoom(volume,zf,order=1); osz=volume.shape
            if zoomed.shape[0]>=osz[0]:
                sx,sy,sz=[(zoomed.shape[i]-osz[i])//2 for i in range(3)]
                ex,ey,ez=[sx+osz[0],sy+osz[1],sz+osz[2]]
                volume=zoomed[sx:ex,sy:ey,sz:ez]
            else:
                pad=np.zeros(osz,zoomed.dtype)
                ox,oy,oz=[(osz[i]-zoomed.shape[i])//2 for i in range(3)]
                pad[ox:ox+zoomed.shape[0],oy:oy+zoomed.shape[1],oz:oz+zoomed.shape[2]]=zoomed
                volume=pad
        return volume
class RandomRotate3D:
    def __init__(self,p=0.5,max_angle=10): self.p=p; self.max_angle=max_angle
    def __call__(self,volume):
        if random.random()<self.p:
            axes=random.choice([(0,1),(0,2),(1,2)])
            angle=random.uniform(-self.max_angle,self.max_angle)
            volume=scipy.ndimage.rotate(volume,angle,axes=axes,reshape=False,order=1)
        return volume
class RandomFlip3D:
    def __init__(self,p=0.5): self.p=p
    def __call__(self,volume):
        if random.random()<self.p:
            volume=np.flip(volume,axis=random.choice([0,1,2])).copy()
        return volume
class RandomIntensityShift:
    def __init__(self,p=0.5,shift_range=(-0.07,0.07)): self.p=p; self.shift_range=shift_range
    def __call__(self,volume):
        if random.random()<self.p: volume+=random.uniform(*self.shift_range)
        return volume
class RandomIntensityScale:
    def __init__(self,p=0.5,scale_range=(0.95,1.05)): self.p=p; self.scale_range=scale_range
    def __call__(self,volume):
        if random.random()<self.p: volume*=random.uniform(*self.scale_range)
        return volume
class RandomGamma:
    def __init__(self,p=0.5,gamma_range=(0.9,1.1)): self.p=p; self.gamma_range=gamma_range
    def __call__(self,volume):
        if random.random()<self.p:
            gm=random.uniform(*self.gamma_range); volume=np.clip(volume,0,None)
            vmin,vmax=volume.min(),volume.max()
            if vmax>vmin:
                norm=(volume-vmin)/(vmax-vmin+1e-10); volume=norm**gm*(vmax-vmin)+vmin
        return volume

augment_transform=transforms.Compose([
    RandomRotate3D(0.5,10), RandomFlip3D(0.5), RandomZoom3D(0.5,(0.9,1.1)),
    RandomGaussianNoise(0.5,(0.01,0.03)), RandomIntensityShift(0.5,(-0.07,0.07)),
    RandomIntensityScale(0.5,(0.95,1.05)), RandomGamma(0.5,(0.9,1.1))
])

##############################################################################
#                NIfTI → standardized NumPy cache (.npy)                     #
##############################################################################
def create_np_data(image_dir,label_dir,clinical_df,train_list,val_list,npy_save_path,shape=(64,64,64)):
    console.print("Starting conversion to NPY...")
    all_pids=train_list+val_list; gather_samples=[]
    with Progress(TextColumn("{task.description}",style="black"),BarColumn(None,"green","green"),
                  TaskProgressColumn(),TextColumn("{task.completed}/{task.total}"),console=Console(),transient=True) as progress:
        tid=progress.add_task("Loading and resizing",total=len(all_pids))
        for pid in all_pids:
            if clinical_df[clinical_df["Image_Label_Base"]==pid].empty:
                progress.update(tid,advance=1); continue
            ip=os.path.join(image_dir,pid+".nii.gz"); lp=os.path.join(label_dir,pid+".nii.gz")
            if not os.path.exists(ip): ip=os.path.join(image_dir,pid+".nii")
            if not os.path.exists(lp): lp=os.path.join(label_dir,pid+".nii")
            vol=nib.load(ip).get_fdata(); seg=nib.load(lp).get_fdata()
            vol,seg=resize(vol,shape),resize(seg,shape,order=0,preserve_range=True).astype(np.float32)
            seg=(seg>0.5).astype(np.float32); gather_samples.append((vol*seg).reshape(-1,1))
            progress.update(tid,advance=1)

    scaler=StandardScaler().fit(np.concatenate(gather_samples,axis=0))
    cache_dict={}
    def build_data(pid_list,desc):
        items=[]
        with Progress(TextColumn("{task.description}",style="black"),BarColumn(None,"green","green"),
                      TaskProgressColumn(),TextColumn("{task.completed}/{task.total}"),console=Console(),transient=True) as p2:
            tid2=p2.add_task(desc,total=len(pid_list))
            for pid_ in pid_list:
                row_=clinical_df[clinical_df["Image_Label_Base"]==pid_]
                ip_=os.path.join(image_dir,pid_+".nii.gz"); lp_=os.path.join(label_dir,pid_+".nii.gz")
                if not os.path.exists(ip_): ip_=os.path.join(image_dir,pid_+".nii")
                if not os.path.exists(lp_): lp_=os.path.join(label_dir,pid_+".nii")
                vol_=resize(nib.load(ip_).get_fdata(),shape)
                seg_=resize(nib.load(lp_).get_fdata(),shape,order=0,preserve_range=True).astype(np.float32)
                seg_=(seg_>0.5).astype(np.float32); masked=vol_*seg_
                masked=scaler.transform(masked.reshape(-1,1)).reshape(*shape)
                vol3c=np.repeat(masked[np.newaxis,...],3,axis=0)
                cache_dict[pid_]=vol3c.astype(np.float32)
                items.append({'pid':pid_,'cache_pid':pid_,'image':None,
                              'time':float(row_["Surv Time Censor 7"].values[0]),
                              'event':bool(row_["Censor 7: OS"].values[0])})
                p2.update(tid2,advance=1)
        return items
    train_data=build_data(train_list,"Processing and scaling (train)")
    val_data  =build_data(val_list,  "Processing and scaling (val)")
    np.save(npy_save_path,{'train':train_data,'val':val_data,'mean':scaler.mean_[0],
                           'std':scaler.scale_[0],'cache_dict':cache_dict})
    console.print(f"Saved data to {npy_save_path}")

##############################################################################
#                                Dataset                                     #
##############################################################################
class LiverDataset(torch.utils.data.Dataset):
    def __init__(self,data_list,cache_dict,augment=False,tensor_cache=None):
        self.data_list=data_list; self.cache_dict=cache_dict
        self.augment=augment; self.tensor_cache=tensor_cache
    def __len__(self): return len(self.data_list)
    def __getitem__(self,idx):
        d=self.data_list[idx]; pid=d['cache_pid']
        vol=self.tensor_cache[pid].clone() if (not self.augment and self.tensor_cache) else self.cache_dict[pid].copy()
        if self.augment:
            t=np.transpose(vol,(1,2,3,0)); t=augment_transform(t); vol=np.transpose(t,(3,0,1,2))
        return {'Whole_Liver_image':torch.tensor(vol,dtype=torch.float32),
                'survival_time'    :torch.tensor(d['time'],dtype=torch.float32),
                'event_occurred'   :torch.tensor(d['event'],dtype=torch.bool),
                'pid'              :d['pid']}

##############################################################################
#                       3-D CNN Feature Extractor                            #
##############################################################################
class Swish(nn.Module):                  # custom activation
    def forward(self,x): return x*torch.sigmoid(x)
ACTIVATIONS={'ReLU':nn.ReLU,'ELU':nn.ELU,'LeakyReLU':nn.LeakyReLU,'Swish':Swish}
def init_weights(m):
    if isinstance(m,(nn.Linear,nn.Conv3d)):
        nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias) if m.bias is not None else None
class ChannelAttention3D(nn.Module):
    def __init__(self,in_planes,reduction=16):
        super().__init__()
        self.avgp,self.maxp=nn.AdaptiveAvgPool3d(1),nn.AdaptiveMaxPool3d(1)
        self.fc=nn.Sequential(nn.Linear(in_planes,in_planes//reduction,bias=False),nn.ReLU(inplace=True),
                              nn.Linear(in_planes//reduction,in_planes,bias=False))
    def forward(self,x):
        b,c= x.size(0),x.size(1)
        out=torch.sigmoid(self.fc(self.avgp(x).view(b,c))+self.fc(self.maxp(x).view(b,c))).view(b,c,1,1,1)
        return x*out
class SpatialAttention3D(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__(); pad=(kernel_size-1)//2
        self.conv=nn.Conv3d(2,1,kernel_size,padding=pad,bias=False)
    def forward(self,x):
        out=torch.cat([torch.mean(x,dim=1,keepdim=True),torch.max(x,dim=1,keepdim=True)[0]],dim=1)
        return x*torch.sigmoid(self.conv(out))
class CBAM3D(nn.Module):
    def __init__(self,c,reduction=16,spatial_kernel_size=7): super().__init__(); self.ca,self.sa=ChannelAttention3D(c,reduction),SpatialAttention3D(spatial_kernel_size)
    def forward(self,x): return self.sa(self.ca(x))
class ResidualBlock3D(nn.Module):
    def __init__(self,ic,oc,stride=1,activation='Swish',norm=True):
        super().__init__(); act=ACTIVATIONS[activation]()
        self.conv1,self.bn1=nn.Conv3d(ic,oc,3,stride,padding=1),nn.BatchNorm3d(oc) if norm else nn.Identity()
        self.act=act
        self.conv2,self.bn2=nn.Conv3d(oc,oc,3,padding=1),nn.BatchNorm3d(oc) if norm else nn.Identity()
        self.shortcut=nn.Identity() if stride==1 and ic==oc else nn.Sequential(nn.Conv3d(ic,oc,1,stride),
                                                                               nn.BatchNorm3d(oc) if norm else nn.Identity())
    def forward(self,x):
        out=self.act(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out=self.act(out+self.shortcut(x))
        return out
class Advanced3DCNN(nn.Module):
    def __init__(self,activation='Swish',norm=True,drop=0.2):
        super().__init__()
        self.layer0=nn.Sequential(nn.Conv3d(3,16,3,padding=1),
                                  nn.BatchNorm3d(16) if norm else nn.Identity(),
                                  ACTIVATIONS[activation](),nn.MaxPool3d(2))
        self.res1,self.res2,self.res3=ResidualBlock3D(16,32,2,activation,norm),ResidualBlock3D(32,64,2,activation,norm),ResidualBlock3D(64,128,2,activation,norm)
        self.cbam, self.gap, self.drop = CBAM3D(128,16,7), nn.AdaptiveAvgPool3d(1), nn.Dropout3d(drop)
        self.apply(init_weights)
    def forward(self,x):
        x=self.layer0(x); x=self.res1(x); x=self.res2(x); x=self.res3(x)
        x=self.cbam(x);  x=self.gap(x);  x=self.drop(x)
        return torch.flatten(x,1)
class DeepSurv(nn.Module):
    def __init__(self,dims,drop=0.2,norm=True,activation='Swish'):
        super().__init__(); act=ACTIVATIONS[activation]
        layers=[]
        for i in range(len(dims)-1):
            if i>0 and drop: layers.append(nn.Dropout(drop))
            layers.append(nn.Linear(dims[i],dims[i+1]))
            if norm and i<len(dims)-1: layers.append(nn.BatchNorm1d(dims[i+1]))
            if i<len(dims)-1: layers.append(act())
        self.model=nn.Sequential(*layers); self.apply(init_weights)
    def forward(self,X): return self.model(X)

##############################################################################
#                   EMA / TTA / MC-Dropout utilities                         #
##############################################################################
class EMA:
    def __init__(self,model,decay=0.999):
        self.model,self.decay=model,decay
        self.shadow={n:p.data.clone() for n,p in model.named_parameters() if p.requires_grad}
        self.shadow_buffers={n:buf.clone() for n,buf in model.named_buffers()}
    def update(self):
        with torch.no_grad():
            for n,p in self.model.named_parameters():
                if p.requires_grad: self.shadow[n]=self.decay*self.shadow[n]+(1-self.decay)*p.data
            for n,b in self.model.named_buffers(): self.shadow_buffers[n]=b.clone()
    def apply_shadow(self):
        for n,p in self.model.named_parameters():
            if p.requires_grad: p.data.copy_(self.shadow[n])
        for n,b in self.model.named_buffers(): b.copy_(self.shadow_buffers[n])
def get_full_ema_state(ema):
    st=ema.model.state_dict()
    return {n:(ema.shadow[n] if n in ema.shadow else (ema.shadow_buffers[n] if n in ema.shadow_buffers else v))
            for n,v in st.items()}
def enable_mc_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'): m.train()
        elif isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d)): m.eval()
def apply_tta_mc_dropout(fe,ds,batch,num_tta=5,num_mc=5):
    set_eval_seed(EVAL_SEED); fe.eval(); ds.eval(); enable_mc_dropout(fe); enable_mc_dropout(ds)
    base=batch['Whole_Liver_image'].cpu().numpy(); preds=[]
    for _ in range(num_tta):
        aug=torch.tensor(np.stack(base),dtype=torch.float32,device=device); mc=[]
        for _ in range(num_mc):
            with autocast(enabled=(device.type=='cuda')):
                mc.append(ds(fe(aug)).detach().cpu().numpy().ravel())
        preds.append(np.mean(mc,axis=0))
    return np.mean(preds,axis=0)
def evaluate_model(fe,ds,dset,bsz):
    set_eval_seed(EVAL_SEED)
    loader=DataLoader(dset,batch_size=bsz,shuffle=False,num_workers=2,pin_memory=(device.type=='cuda'))
    t,e,p=[],[],[]
    for batch in loader:
        t.append(batch['survival_time'].cpu().numpy())
        e.append(batch['event_occurred'].cpu().numpy().astype(bool))
        p.append(apply_tta_mc_dropout(fe,ds,batch,5,5))
    return np.concatenate(t),np.concatenate(e),np.concatenate(p)
def compute_uno_cindex(train,val):
    return concordance_index_ipcw(Surv.from_arrays(event=train[1],time=train[0]),
                                  Surv.from_arrays(event=val[1],time=val[0]),val[2])[0]

##############################################################################
#                        Loss & Regularization                               #
##############################################################################
class Regularization(nn.Module):
    def __init__(self,order=2,weight_decay=1e-4): super().__init__(); self.order,self.weight_decay=order,weight_decay
    def forward(self,model): return self.weight_decay*sum(torch.norm(w,p=self.order) for n,w in model.named_parameters() if w.requires_grad and 'weight' in n)
class CoxPartialLogLikelihood(nn.Module):
    def __init__(self,cfg): super().__init__(); self.L2_reg=cfg['l2_reg']; self.reg=Regularization(2,self.L2_reg)
    def forward(self,risk,y,e,model):
        idx=torch.argsort(y); srisk, se=risk[idx], e[idx]; expr=torch.exp(srisk)
        cs=torch.flip(torch.cumsum(torch.flip(expr,[0]),0),[0]); lse=torch.log(cs)
        return -torch.sum(se*(srisk-lse))/(torch.sum(e)+1e-8)+self.reg(model)

##############################################################################
#                  (Optional) 8-class SSL transform head                     #
##############################################################################
class MultiTransformHead(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        self.fc=nn.Sequential(nn.Linear(in_dim,in_dim//2),nn.ReLU(),nn.Linear(in_dim//2,8))
        init_weights(self.fc[0]); init_weights(self.fc[2])
    def forward(self,x): return self.fc(x)
def apply_3d_transform_8class(volume,tid):
    rot,flip=tid%4,tid//4
    if rot>0: volume=np.rot90(volume,k=rot,axes=(2,3)).copy()
    if flip==1: volume=np.flip(volume,axis=1).copy()
    return volume

##############################################################################
#             Embeddings helpers → Hyper-parameter search                    #
##############################################################################
class CachedEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self,items): self.items=items
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]

def compute_embeddings_for_dataset(fe,dset):
    fe.eval(); loader=DataLoader(dset,batch_size=4,shuffle=False,num_workers=2)
    items=[]
    with torch.no_grad():
        for b in loader:
            emb=fe(b['Whole_Liver_image'].to(device))
            for i in range(emb.size(0)):
                items.append({'embedding':emb[i].cpu(),'time':b['survival_time'][i].item(),
                              'event':b['event_occurred'][i].item(),'pid':b['pid'][i]})
    return items

def train_val_on_embeddings(emb_ds,ds,train_idx,cfg):
    from torch.optim.lr_scheduler import OneCycleLR
    loss_fn=CoxPartialLogLikelihood(cfg)
    opt=torch.optim.Adam(ds.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    sc=GradScaler(enabled=(device.type=='cuda'))
    loader=DataLoader(emb_ds,batch_size=cfg['batch_size'],sampler=SubsetRandomSampler(train_idx),num_workers=2,drop_last=True)
    sched=OneCycleLR(opt,max_lr=cfg['lr'],steps_per_epoch=len(loader),epochs=2)
    ds.train()
    for _ in range(2):
        for b in loader:
            opt.zero_grad()
            with autocast(enabled=(device.type=='cuda')):
                out=ds(b['embedding'].to(device))
                loss=loss_fn(out,torch.tensor(b['time'],device=device),
                             torch.tensor(b['event'],device=device,dtype=torch.bool),ds)
            sc.scale(loss).backward(); sc.step(opt); sc.update(); sched.step()
    # full-set evaluation
    ds.eval(); loader_eval=DataLoader(emb_ds,batch_size=cfg['batch_size'],shuffle=False)
    t,e,p=[],[],[]
    with torch.no_grad():
        for b in loader_eval:
            o=ds(b['embedding'].to(device)).cpu().numpy().squeeze()
            t.append(np.array(b['time'])); e.append(np.array(b['event']).astype(bool)); p.append(o)
    t,e,p=np.concatenate(t),np.concatenate(e),np.concatenate(p)
    return concordance_index_ipcw(Surv.from_arrays(event=e,time=t),Surv.from_arrays(event=e,time=t),p)[0]

def hyperparam_search_with_embeddings(emb_ds,k=3):
    from itertools import product
    lr_list=[1e-4,3e-4,5e-4,1e-3]; drop_list=[0.1,0.2,0.3]; hid_list=[128,256]
    act_list=['Swish','LeakyReLU']; bsz_list=[4,8,16]; wd_list=[1e-5,1e-4]; l2_list=[1e-5,1e-4]
    idx=list(range(len(emb_ds))); random.shuffle(idx); kf=KFold(k,True,42)
    best_c=-999; best_cfg=None; all_comb=list(product(lr_list,drop_list,hid_list,act_list,bsz_list,wd_list,l2_list))
    with Progress(TextColumn("{task.description}",style="black"),BarColumn(None,"green","green"),
                  TaskProgressColumn(),TextColumn("{task.completed}/{task.total}"),transient=True,console=console) as p:
        tid=p.add_task("Hyperparameter Search",total=len(all_comb))
        for lr,dp,hd,act,bsz,wd,l2 in all_comb:
            cfg={'lr':lr,'drop':dp,'hidden_dim':hd,'activation':act,'batch_size':bsz,'weight_decay':wd,'l2_reg':l2}
            scores=[]
            for tri,_ in kf.split(idx):
                ds=DeepSurv([emb_ds[0]['embedding'].shape[0],hd,hd,1],drop=dp,activation=act,norm=True).to(device)
                scores.append(train_val_on_embeddings(emb_ds,ds,tri,cfg))
            m=np.mean(scores)
            if m>best_c: best_c,best_cfg=m,cfg
            p.update(tid,advance=1)
    return best_cfg,best_c

##############################################################################
#               Stratified WeightedRandomSampler for class-imbalance         #
##############################################################################
def create_stratified_sampler(dset):
    e=np.array([dset[i]['event_occurred'].item() for i in range(len(dset))])
    w=np.zeros_like(e,dtype=float); w[e==1]=2.0/(e==1).sum(); w[e==0]=1.0/(e==0).sum()
    return WeightedRandomSampler(w,len(w),replacement=True)

##############################################################################
#                       (Optional) Self-Supervised pre-train                 #
##############################################################################
def train_ssl(fe,dataset,cfg,seed):
    ssl_path=os.path.join(results_dir,f'ssl_best_fe_seed{seed}.pt')
    if os.path.exists(ssl_path):
        console.print(f"Seed {seed}: Reusing SSL weights."); fe.load_state_dict(torch.load(ssl_path,map_location=device)); return
    class SSLDS(torch.utils.data.Dataset):
        def __init__(self,dl): self.dl=dl
        def __len__(self): return len(self.dl)
        def __getitem__(self,i): return torch.tensor(self.dl[i]['Whole_Liver_image'],dtype=torch.float32)
    ssl_ds=SSLDS([dataset[i] for i in range(len(dataset))])
    idx=list(range(len(ssl_ds))); random.shuffle(idx); val_split=int(0.1*len(idx))
    train_idx,val_idx=idx[val_split:],idx[:val_split]
    tr_loader=DataLoader(ssl_ds,batch_size=cfg['batch_size'],sampler=SubsetRandomSampler(train_idx),num_workers=2,drop_last=True)
    val_loader=DataLoader(ssl_ds,batch_size=cfg['batch_size'],sampler=SubsetRandomSampler(val_idx),num_workers=2)
    conv_dim=fe(torch.zeros(1,3,64,64,64).to(device)).shape[1]
    head=MultiTransformHead(conv_dim).to(device); opt=torch.optim.Adam(list(fe.parameters())+list(head.parameters()),
                                                                       lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    ce=nn.CrossEntropyLoss(); from torch.optim.lr_scheduler import OneCycleLR
    sched=OneCycleLR(opt,max_lr=cfg['lr'],steps_per_epoch=len(tr_loader),epochs=100); sc=GradScaler(enabled=(device.type=='cuda'))
    def run(loader,train=True):
        fe.train() if train else fe.eval(); head.train() if train else head.eval()
        loss_tot=cnt=0
        for imgs in loader:
            imgs=imgs.to(device); tid=random.randint(0,7); bsz=imgs.size(0)
            xfrm=torch.tensor(np.concatenate([apply_3d_transform_8class(imgs[i].cpu().numpy(),tid)[None,...] for i in range(bsz)]),
                              dtype=torch.float32,device=device)
            if train: opt.zero_grad()
            with autocast(enabled=(device.type=='cuda')):
                out=head(fe(xfrm)); tgt=torch.full((bsz,),tid,dtype=torch.long,device=device); loss=ce(out,tgt)
            if train:
                sc.scale(loss).backward(); sc.step(opt); sc.update(); sched.step()
            loss_tot+=loss.item()*bsz; cnt+=bsz
        return loss_tot/cnt
    best=9e9; patience=0
    with Progress(console=console,transient=True) as prog:
        task=prog.add_task("SSL Epochs",total=100)
        for ep in range(1,101):
            run(tr_loader,True); val=run(val_loader,False)
            if val<best: best,val = val,val; patience=0; torch.save(fe.state_dict(),ssl_path)
            else: patience+=1;
            if patience>=20: break
            prog.update(task,advance=1)
    fe.load_state_dict(torch.load(ssl_path,map_location=device))

##############################################################################
#                       Supervised training per seed                         #
##############################################################################
def train_single_seed(seed,train_ds,val_ds,cfg,do_ssl=True):
    console.print(f"Training with seed={seed}..."); set_seed(seed)
    fe=Advanced3DCNN(cfg['activation'],True,cfg['drop']).to(device)
    conv_dim=fe(torch.zeros(1,3,64,64,64).to(device)).shape[1]
    ds=DeepSurv([conv_dim,cfg['hidden_dim'],cfg['hidden_dim'],1],cfg['drop'],True,cfg['activation']).to(device)
    if do_ssl: train_ssl(fe,train_ds,cfg,seed)
    loss_fn=CoxPartialLogLikelihood(cfg)
    opt=torch.optim.Adam(list(fe.parameters())+list(ds.parameters()),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    train_loader=DataLoader(train_ds,batch_size=cfg['batch_size'],sampler=create_stratified_sampler(train_ds),num_workers=2,drop_last=True)
    val_loader=DataLoader(val_ds,batch_size=cfg['batch_size'],shuffle=False,num_workers=2)
    from torch.optim.lr_scheduler import OneCycleLR
    sched=OneCycleLR(opt,max_lr=cfg['lr'],steps_per_epoch=len(train_loader),epochs=cfg.get('max_epoch',200))
    ema_fe,ema_ds=EMA(fe),EMA(ds); best_val=-999; patience=0; best_ckpt=None
    with Progress(TextColumn("{task.description}",style="black"),BarColumn(None,"green","green"),TaskProgressColumn(),
                  TextColumn("{task.completed}/{task.total}"),console=Console(),transient=True) as prog:
        task=prog.add_task("Epochs",total=cfg.get('max_epoch',200))
        for ep in range(1,cfg.get('max_epoch',200)+1):
            fe.train(); ds.train()
            for b in train_loader:
                x=b['Whole_Liver_image'].to(device); t=b['survival_time'].to(device); e=b['event_occurred'].to(device)
                opt.zero_grad()
                with autocast(enabled=(device.type=='cuda')):
                    loss=loss_fn(ds(fe(x)),t,e,ds)
                scaler_amp.scale(loss).backward(); scaler_amp.step(opt); scaler_amp.update(); sched.step()
                ema_fe.update(); ema_ds.update()
            # ----- EMA eval -----
            fe_b,ds_b=deepcopy(fe.state_dict()),deepcopy(ds.state_dict())
            ema_fe.apply_shadow(); ema_ds.apply_shadow()
            val_c=compute_uno_cindex(evaluate_model(fe,ds,train_ds,cfg['batch_size']),
                                     evaluate_model(fe,ds,val_ds,cfg['batch_size']))
            fe.load_state_dict(fe_b); ds.load_state_dict(ds_b)
            if val_c>best_val:
                best_val=val_c; patience=0
                best_ckpt={'feature_extractor':get_full_ema_state(ema_fe),'deep_surv':get_full_ema_state(ema_ds),
                           'val_cindex':val_c,'epoch':ep}
                console.print(f"Epoch {ep}: new best c-index = {val_c:.4f}")
            else:
                patience+=1
                if patience>=cfg.get('early_stop',30): console.print("Early stop."); break
            prog.update(task,advance=1)
    path=os.path.join(results_dir,f'final_best_model_seed{seed}.pt')
    if best_ckpt: torch.save(best_ckpt,path); console.print(f"Saved checkpoint to {path}")
    # reload verification
    fe_r=Advanced3DCNN(cfg['activation'],True,cfg['drop']).to(device); conv_dim=fe_r(torch.zeros(1,3,64,64,64).to(device)).shape[1]
    ds_r=DeepSurv([conv_dim,cfg['hidden_dim'],cfg['hidden_dim'],1],cfg['drop'],True,cfg['activation']).to(device)
    ck=torch.load(path,map_location=device); fe_r.load_state_dict(ck['feature_extractor'],False); ds_r.load_state_dict(ck['deep_surv'],False)
    r_c=compute_uno_cindex(evaluate_model(fe_r,ds_r,train_ds,cfg['batch_size']),
                           evaluate_model(fe_r,ds_r,val_ds,cfg['batch_size']))
    console.print(f"Reloaded model c-index (val) = {r_c:.4f}")
    return r_c

##############################################################################
#                  Utility – load trained model by seed                      #
##############################################################################
def load_trained_model(seed,cfg):
    fe=Advanced3DCNN(cfg['activation'],True,cfg['drop']).to(device)
    conv_dim=fe(torch.zeros(1,3,64,64,64).to(device)).shape[1]
    ds=DeepSurv([conv_dim,cfg['hidden_dim'],cfg['hidden_dim'],1],cfg['drop'],True,cfg['activation']).to(device)
    ck=torch.load(os.path.join(results_dir,f'final_best_model_seed{seed}.pt'),map_location=device)
    fe.load_state_dict(ck['feature_extractor']); ds.load_state_dict(ck['deep_surv'])
    fe.eval(); ds.eval(); return fe,ds

##############################################################################
#                  Ensemble evaluation with aggregator                       #
##############################################################################
def evaluate_ensemble(dset,models,cfg,aggregator='mean'):
    set_eval_seed(EVAL_SEED)
    loader=DataLoader(dset,batch_size=cfg['batch_size'],shuffle=False,num_workers=2)
    t,e,p=[],[],[]
    for b in loader:
        t.append(b['survival_time'].cpu().numpy()); e.append(b['event_occurred'].cpu().numpy().astype(bool))
        preds=np.array([apply_tta_mc_dropout(f,d,b,5,5) for f,d in models])
        if   aggregator=='median': p.append(np.median(preds,axis=0))
        elif aggregator=='max'   : p.append(np.max(preds,axis=0))
        else                     : p.append(np.mean(preds,axis=0))
    return np.concatenate(t),np.concatenate(e),np.concatenate(p)

##############################################################################
#                                   Main                                     #
##############################################################################
def main():
    console.print("Check dataset paths before running."); import torch.backends.cudnn as cudnn; cudnn.benchmark=True
    npy_path=os.path.join(cached_dir,"Final_OS_data.npy")
    if not os.path.exists(npy_path):
        create_np_data(liver_image_dir,liver_label_dir,merged_data,train_image_names,val_image_names,npy_path,(64,64,64))
    else: console.print("Existing NPY found.")
    dat=np.load(npy_path,allow_pickle=True).item(); train_list, val_list, cache_dict=dat['train'],dat['val'],dat['cache_dict']
    # tensor caches for fast eval
    val_cache_t={i['cache_pid']:torch.from_numpy(cache_dict[i['cache_pid']]).float() for i in val_list}
    train_cache_t={i['cache_pid']:torch.from_numpy(cache_dict[i['cache_pid']]).float() for i in train_list}
    train_ds=LiverDataset(train_list,cache_dict,True,train_cache_t); val_ds=LiverDataset(val_list,cache_dict,False,val_cache_t)
    # ----- Embedding cache for H-param search ---------------------------------
    emb_path=os.path.join(cached_dir,"emb_dataset.pkl")
    if os.path.exists(emb_path):
        emb_ds=pickle.load(open(emb_path,'rb')); console.print("Loaded cached embeddings.")
    else:
        console.print("Building embeddings for hyper-param search (quick approx).")
        emb_ds=[]
        for d in train_ds:
            vol=d['Whole_Liver_image'].flatten(); n_chunks,min_dim=128,max(1,vol.shape[0]//128)
            emb=torch.stack([vol[i*min_dim:(i+1)*min_dim].mean() for i in range(n_chunks)])
            emb_ds.append({'embedding':emb,'time':d['survival_time'].item(),'event':d['event_occurred'].item(),'pid':d['pid']})
        pickle.dump(emb_ds,open(emb_path,'wb')); console.print("Embeddings saved.")
    # ----- Hyper-param search cache -------------------------------------------
    h_path=os.path.join(cached_dir,"best_params.pkl")
    if os.path.exists(h_path):
        best_params=pickle.load(open(h_path,'rb')); console.print(f"Loaded best params: {best_params}")
    else:
        best_params,best_c=hyperparam_search_with_embeddings(CachedEmbeddingDataset(emb_ds),3)
        pickle.dump(best_params,open(h_path,'wb')); console.print(f"Best params saved: {best_params}")
    # ----- Train multiple seeds -----------------------------------------------
    seeds=[42,1337,2023,9999,888,1010,2022,2222,777,555]; scores=[]
    for s in seeds:
        path=os.path.join(results_dir,f'final_best_model_seed{s}.pt')
        c=compute_uno_cindex(evaluate_model(*load_trained_model(s,best_params),train_ds,best_params['batch_size']),
                             evaluate_model(*load_trained_model(s,best_params),val_ds,best_params['batch_size'])) if os.path.exists(path) \
          else train_single_seed(s,train_ds,val_ds,best_params,True)
        scores.append((s,c)); console.print(f"Seed {s}: c-index={c:.4f}")
    top5=[s for s,_ in sorted(scores,key=lambda x:x[1],reverse=True)[:5]]
    console.print(f"Top-5 seeds: {top5}")
    aggregations=['mean','median','max']; best_single=-999; best_info=None
    for i in range(1,len(top5)+1):
        combo=top5[:i]
        for agg in aggregations:
            mdl=[load_trained_model(s,best_params) for s in combo]
            c=compute_uno_cindex(evaluate_ensemble(train_ds,mdl,best_params,agg),
                                 evaluate_ensemble(val_ds,mdl,best_params,agg))
            console.print(f"[combo {combo}, agg={agg}] c-index={c:.4f}")
            if c>best_single: best_single, best_info=c,{'subset':tuple(combo),'agg':agg,'models':mdl,'single_c':c}
    console.print(f"Best ensemble: {best_info}")
    # ----- Bootstrap CI --------------------------------------------------------
    mdl,agg=best_info['models'],best_info['agg']
    t_tr,e_tr,p_tr=evaluate_ensemble(train_ds,mdl,best_params,agg)
    s_tr=Surv.from_arrays(event=e_tr,time=t_tr)
    t_val,e_val,p_val=evaluate_ensemble(val_ds,mdl,best_params,agg)
    boot=[concordance_index_ipcw(s_tr,Surv.from_arrays(event=e_val[idx],time=t_val[idx]),p_val[idx])[0]
          for idx in (np.random.choice(len(t_val),len(t_val),True) for _ in range(500))]
    med,lo,hi=np.median(boot),np.percentile(boot,2.5),np.percentile(boot,97.5)
    console.print(f"Bootstrap median={med:.4f}, 95% CI=[{lo:.4f},{hi:.4f}]")
    # ----- Generate final CSVs -------------------------------------------------
    full_ds=LiverDataset(train_list+val_list,cache_dict,False,None)
    feats,risks,ids=[],[],[]
    def pred(fe,ds,x): with torch.no_grad(): e=fe(x); r=ds(e); return e.cpu().numpy().squeeze(),r.cpu().numpy().squeeze()
    for b in DataLoader(full_ds,batch_size=1,shuffle=False):
        sub_e,sub_r=[],[]
        for fe,ds in mdl:
            e,r=pred(fe,ds,b['Whole_Liver_image'].to(device)); sub_e.append(e); sub_r.append(r)
        feats.append(np.mean(sub_e,0)); risks.append({'mean':np.mean,'median':np.median,'max':np.max}[agg](np.array(sub_r)))
        ids.append(b['pid'][0])
    np.savetxt(os.path.join(results_dir,"Liver_DeepFeatures_OS.csv"),
               np.column_stack([ids,np.stack(feats)]),fmt='%s',delimiter=',',
               header="patient_id,"+','.join(f"feat_{i}" for i in range(feats[0].shape[0])),comments='')
    np.savetxt(os.path.join(results_dir,"Liver_RiskValues_OS.csv"),
               np.column_stack([ids,risks]),fmt='%s',delimiter=',',header="patient_id,risk",comments='')
    with open(os.path.join(results_dir,"a.txt"),'w') as f:
        f.write(f"Best ensemble c-index={best_single:.4f}\nBootstrap median={med:.4f}, 95% CI=[{lo:.4f},{hi:.4f}]\n")
    console.print("Done.")

if __name__ == "__main__":
    main()
