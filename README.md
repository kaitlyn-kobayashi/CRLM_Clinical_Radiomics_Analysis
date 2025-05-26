# CRLM Clinical-Radiomics Analysis – Deep Learning

Automated 3-D deep-learning pipeline for **liver-only colorectal cancer liver metastasis (CRLM)** survival prediction.

![GPU](https://img.shields.io/badge/GPU-RTX%20A6000-77B900?logo=nvidia&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.4-1482C5?logo=nvidia)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-E34F26?logo=pytorch&logoColor=white)

The repository provides six runnable scripts—one per region-of-interest (ROI) × end-point pair:

| ROI (3-D mask) | Endpoint | Script |
|----------------|----------|--------|
| **Whole Liver** | Overall Survival | `Whole_Liver_OS.py` |
|                | Hepatic DFS      | `Whole_Liver_HDFS.py` |
| **Liver Parenchyma** | OS | `Liver_OS.py` |
|                    | HDFS | `Liver_HDFS.py` |
| **Tumor Volume** | OS | `Tumor_OS.py` |
|                 | HDFS | `Tumor_HDFS.py` |

---

## Technical Specs ⚙️

* 🖥️ Single **NVIDIA RTX A6000** GPU (driver `550.90.07`, CUDA `12.4`)  
* 🔥 **PyTorch 2.1** with Automatic Mixed Precision (AMP)

All experiments were executed on this hardware configuration.

---
