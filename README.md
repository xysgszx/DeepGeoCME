# Deep Learning-Enabled Prediction of Geoeffective CMEs Using SOHO and SDO Observations

## Authors

Zhaoxin Yan, Yasser Abduallah, Jason T. L. Wang

## Overview

A deep learning fusion model for predicting CME geoeffectiveness from SOHO and SDO observations. This repository provides training and testing code, with pretrained weights and prediction examples available on [Zenodo](https://zenodo.org/records/22942117).

## File Structure

- `train_dnet.py`: Training script.
- `demo/test.py`: Testing script.
- `demo/model.py`: Model architecture.
- `demo/dnet_best.pt`: Pretrained weights in the Zenodo package.
- `demo/test.csv`, `demo/test_tensors/`: Test labels and inputs.
- `Prediction_workflow.pdf`: Prediction workflow.

## Installation

Use Python 3.10 or later.

```bash
pip install -r requirements.txt
```

## Training

```bash
python train_dnet.py --train-csv /path/to/train.csv --test-csv /path/to/test.csv --cache-dir /path/to/tensors --save checkpoints/run.pt
```

## Testing

Extract `dnet_paper_examples.zip` from Zenodo, open its `dnet` directory, and run:

```bash
python demo/test.py --num-workers 0
```

## Reference

[Deep Learning-Enabled Prediction of Geoeffective CMEs Using SOHO and SDO Observations](https://arxiv.org/abs/2605.24748).

Copyright (c) 2026 Zhaoxin Yan. See `COPYRIGHT.txt`.
