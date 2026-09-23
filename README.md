# Anomaly Detection Research

Notebook-based experiments in visual anomaly detection for industrial components using [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad). The work includes data exploration, frozen-feature baselines, and PyTorch autoencoder experiments on the Bottle category.

Install the Linux/CUDA environment with `pixi install`, obtain MVTec AD separately, and place the dataset under `data/mvtec/`. The notebooks in [`notebooks/`](notebooks/) contain the methods and evaluation details. The 11 selected, self-authored Bottle checkpoints are in [`models/`](models/README.md), with a file-by-file index and source notebook links.

This is a research repository, not the [industrial-component-anomaly-detection](https://github.com/foersben/industrial-component-anomaly-detection/) Streamlit application or its model registry.
