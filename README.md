# COMP5329 Assignment 2: SAM OOD Generalisation Project

## Research Goal

This project studies whether SAM improves out-of-distribution generalisation only by finding flatter minima, or whether it also produces more stable feature representations.

## Main Experimental Design

- Model: ResNet-18
- Training dataset: CIFAR-10
- OOD dataset: CIFAR-10-C
- Optimisers:
  - SGD
  - AdamW
  - SAM with rho = 0.01
  - SAM with rho = 0.05
  - SAM with rho = 0.10

## Metrics

- Clean CIFAR-10 accuracy
- Mean CIFAR-10-C accuracy
- Robustness gap
- Expected Calibration Error
- Sharpness proxy
- Feature stability cosine similarity
- Effective rank
- Class separation ratio
- Layer-wise representation stability
- Linear CKA between clean and corrupted features
- Correlation between OOD accuracy and feature stability

## File Structure

```text
config.py                  Global paths and experiment configs
utils.py                   Seed and helper functions
data.py                    CIFAR-10 and CIFAR-10-C dataloaders
models.py                  ResNet-18 for CIFAR-10
sam.py                     SAM optimiser
optimizers.py              Optimiser selector
train_utils.py             Training and evaluation functions
metrics.py                 ECE, sharpness, feature stability, effective rank
layer_analysis.py          Layer-wise feature stability and CKA
download_cifar10c.py       Download CIFAR-10-C
main_train.py              Train SGD, AdamW, and SAM models
main_eval_ood.py           Evaluate CIFAR-10-C robustness
main_analysis.py           Main mechanism analysis
main_layer_analysis.py     Deeper layer-wise representation analysis
main_correlation.py        Correlation analysis
plots.py                   Generate figures
run_colab.py               Colab running guide
```

## How to Run in Colab

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/COMP5329_Ass2_SAM
```

Then run:

```bash
!python download_cifar10c.py
!python main_train.py
!python main_eval_ood.py
!python main_analysis.py
!python main_layer_analysis.py
!python main_correlation.py
!python plots.py
```

## Notes

- Start with 10 epochs to debug.
- For the formal experiment, change `main_train.py` from `epochs=10` to `epochs=30` or `epochs=50`.
- If time allows, add more seeds such as 42, 43, and 44 in `config.py`.
