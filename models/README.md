# Bottle autoencoder research checkpoints

These are Ali's PyTorch checkpoints for the MVTec AD Bottle experiments in this repository. They are research artifacts, not the 15-category Keras CAE models used by the separate [industrial-component-anomaly-detection](https://github.com/foersben/industrial-component-anomaly-detection/) Streamlit app. The raw dataset is not included.

| Checkpoints | Experiment | Notebook |
| --- | --- | --- |
| `ae01_clean_reconstruction.pt` through `ae04_clean_lower_lr.pt` | Clean reconstruction and continued/lower-rate training | [`03_modelling_autoencoder.ipynb`](../notebooks/03_modelling_autoencoder.ipynb) |
| `ae05_masked_restoration.pt` | Masked-region restoration | [`03_modelling_autoencoder.ipynb`](../notebooks/03_modelling_autoencoder.ipynb) |
| `ae06_conv_autoencoder16.pt` | Stronger compression with a 16×16 spatial bottleneck | [`03_modelling_autoencoder.ipynb`](../notebooks/03_modelling_autoencoder.ipynb) |
| `ae07_conv_autoencoder16.pt` | Longer AE-06 training | [`03_modelling_autoencoder.ipynb`](../notebooks/03_modelling_autoencoder.ipynb) |
| `ae08_compressed_masked_restoration.pt` | Compressed masked restoration | [`03_modelling_autoencoder.ipynb`](../notebooks/03_modelling_autoencoder.ipynb) |
| `ae09_conv_autoencoder_narrow.pt` | Narrower compressed bottleneck | [`03_modelling_autoencoder.ipynb`](../notebooks/03_modelling_autoencoder.ipynb) |
| `ae10_conv_autoencoder16_pure_ssim.pt` | AE-06 architecture trained with SSIM loss | [`04_modelling_ssim_scoring.ipynb`](../notebooks/04_modelling_ssim_scoring.ipynb) |
| `ae06_team_protocol_85_15.pt` | AE-06 under the shared 85/15 comparison protocol | [`05_team_autoencoder_comparison.ipynb`](../notebooks/05_team_autoencoder_comparison.ipynb) |

The checkpoint dictionaries contain model weights and experiment metadata. In `ae06_team_protocol_85_15.pt`, saved train/validation paths are relative to the repository (`data/mvtec/...`), so the checkpoint does not publish a personal filesystem path. Notebook 05 uses the same relative-path representation when validating this checkpoint.

PyTorch checkpoint files use pickle-based serialization. Only load files you trust, and prefer `torch.load(..., weights_only=True)` where supported. Review [MVTec AD's dataset terms](https://www.mvtec.com/company/research/datasets/mvtec-ad) before redistributing or using derived artifacts.
