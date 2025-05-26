# CRLM Clinical-Radiomics Analysis – Deep Learning

Automated 3-D deep-learning pipeline for **liver-only colorectal cancer liver metastasis (CRLM)** survival prediction.

![GPU](https://img.shields.io/badge/GPU-RTX%20A6000-77B900?logo=nvidia&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.4-1482C5?logo=nvidia)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-E34F26?logo=pytorch&logoColor=white)

The repository provides six runnable scripts—one per region-of-interest (ROI) × endpoint pair:

| ROI (3-D mask)   | Endpoint | Script               |
|------------------|---------|----------------------|
| **Whole Liver**  | Overall Survival (OS) | `Whole_Liver_OS.py` |
|                  | Hepatic DFS (HDFS)   | `Whole_Liver_HDFS.py` |
| **Liver Parenchyma** | OS    | `Liver_OS.py` |
|                      | HDFS  | `Liver_HDFS.py` |
| **Tumor Volume** | OS    | `Tumor_OS.py` |
|                  | HDFS  | `Tumor_HDFS.py` |

---

## Technical Specs ⚙️

* 🖥️ Single **NVIDIA RTX A6000** (driver `550.90.07`, CUDA `12.4`)  
* 🔥 **PyTorch 2.1** with Automatic Mixed Precision (AMP)

All experiments were executed on this hardware configuration.

---

## Installation
```bash
git clone https://github.com/<user>/CRLM_Clinical_Radiomics_Analysis.git
cd CRLM_Clinical_Radiomics_Analysis/Deep_Learning

# Conda example
conda create -n crlm_dl python=3.10
conda activate crlm_dl
pip install -r requirements.txt
```
## Software Requirements
- `torch==2.1`
- `torchvision==0.16`
- `scikit-learn>=1.4`
- `scikit-survival>=0.22`
- `numpy>=1.26`
- `pandas>=2.2`
- `nibabel>=5.2`
- `scipy>=1.13`
- `scikit-image>=0.23`
- `rich>=13.7`

