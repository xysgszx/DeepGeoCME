# Deep Learning-Enabled Prediction of Geoeffective CMEs Using SOHO and SDO Observations

## Authors

Zhaoxin Yan, Yasser Abduallah, Jason T. L. Wang

## Overview

A deep learning fusion model for predicting CME geoeffectiveness from SOHO and SDO observations.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xysgszx/DeepGeoCME/blob/main/reproduce_fig7.ipynb)

Open the notebook and select **Runtime → Run all** to generate the confusion matrix in Fig. 7.

## Files

- `reproduce_fig7.ipynb`: Prediction notebook.
- `reproduce_fig7.py`: Testing and plotting code.
- `model.py`: Model architecture.
- `data.py`, `dnet_data.py`, `metrics.py`: Data loading and evaluation.
- `train_dnet.py`: Training code.
- `test.csv`: Test-event list.
- `Prediction_workflow.pdf`: Prediction workflow.

## Model and Data

Download the [pretrained weights](https://github.com/xysgszx/DeepGeoCME/releases/download/v1.0/dnet_best.pt) and [testing data](https://github.com/xysgszx/DeepGeoCME/releases/download/v1.0/test_tensors.zip) into `assets/`. The notebook downloads these files automatically.

The complete package and prediction workflow are also available on [Zenodo](https://doi.org/10.5281/zenodo.22942116).

## Installation and Testing

Python 3.10 or later:

```bash
pip install -r requirements.txt
python reproduce_fig7.py
```

The program displays event dates and predictions and saves `fig7_confusion_matrix.png` and `predictions.csv`.

## Training

```bash
python train_dnet.py --train-csv /path/to/train.csv --test-csv /path/to/test.csv --cache-dir /path/to/tensors --save checkpoints/run.pt
```

## Reference

[Deep Learning-Enabled Prediction of Geoeffective CMEs Using SOHO and SDO Observations](https://arxiv.org/abs/2605.24748).

Copyright (c) 2026 Zhaoxin Yan. See `COPYRIGHT.txt`.
