# Feature Extraction

Features were extracted on a SLURM cluster using `chi-fexp`, which is a simple wrapper of `pyradiomics`, and is included here as a git submodule.

The `pyradiomics` config file was [here](chi-fexp/chi/fexp/configs/r01_crlm_liver_filters.yaml).

## Preprocessed Files

The deep learning methods relied on the preprocessed images, which mimiced the preprocessing used by `pyradiomics` for feature extraction. Preprocessed images can be output by `chi-fexp` by adding the arguments

```
python -m chi.fexp ALL OTHER ARGS --dump_preprocessed --dump_dir <your-output-dir-here>
```