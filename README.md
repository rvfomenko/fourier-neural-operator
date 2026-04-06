# Fourier Neural Operator: A reproduction
This repository contains a limited reproduction of Li et al. (2020), *Fourier Neural Operator for Parametric Partial Differential Equations*, together with a few extensions for the 2-dimensional Darcy Flow scenario.

Main experiments present in this repo:
- baseline FNO2d for Darcy flow
- baseline FNO3d for Navier-Stokes
- shared-weight FNO2d
- uncertainty quantifiers variants for FNO2d (NLL, $\beta$-NLL)

## Reference
- Paper: [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
- Source code used as reference: [li-Pingan/fourier-neural-operator](https://github.com/li-Pingan/fourier-neural-operator)
- Location of the original dataset from the source repository: [PDE datasets](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-)

## Requirements
This code was written as simple standalone scripts and notebooks. A minimal environment needs the following packages: `torch`, `numpy`, `scipy`, `h5py`, `matplotlib`, `jupyter`.

## Data
Place the datasets in the root `data/` folder, or ensure paths point to the right location within the training scripts.

Files used by this repo:
- `piececonst_r241_N1024_smooth1.mat`
- `piececonst_r241_N1024_smooth2.mat`
- `piececonst_r421_N1024_smooth1.mat`
- `piececonst_r421_N1024_smooth2.mat`
- `ns_V1e-3_N5000_T50.mat`

The 2D Darcy scripts use:
- `smooth1` for training
- `smooth2` for testing

The 3D script uses:
- `ns_V1e-3_N5000_T50.mat`

and splits them manually in the training script. Note that the input and output window in the 3-dimensional case alter the shape of the training data and model output.

The [original repo](https://github.com/li-Pingan/fourier-neural-operator) contains scripts to also generate more data, but as they are computationally expensive, we opted to use the pre-generated ones.

## Repository contents
- `model_2d.py`: baseline 2D FNO
- `model_2d_shared.py`: shared-weight 2D FNO
- `model_2d_uq.py`: 2D UQ model
- `model_3d.py`: 3D FNO
- `train_utils.py`: shared data loading, normalization, and metrics
- `train_2d.py`: baseline Darcy training
- `train_2d_shared.py`: shared-weight Darcy training
- `train_2d_uq.py`: Gaussian-NLL UQ training
- `train_2d_uq_2.py`: Beta-NLL UQ training
- `train_3d.py`: 3D Navier-Stokes training

## Training procedure
Run the training script corresponding to your model (and mode) in the repo root. For example:

```powershell
python train_2d.py
```

For the 2D scripts, the Darcy resolution is selected inside the file with:

```python
mode = 0  # 0: 241x241 case; 1: 421x421 case
```

## Outputs
Trained weights and loss histories are saved in `outputs/`. Typically during training weights are saved at the end of training and every 100 epochs to allow for model analysis as the model is running, or to manually stop training without losing learned weights. Loss histories contain usually the relative $L^2$ loss for comparison purposes, but note that the UQ variants do not optimize on this loss function. One can also save the NLL losses, as they are being tracked in the training scripts, though mind the keys as notebooks assume the plain `'train'` and `'test'` keys for the $L^2$ loss.

## Visualization
- `visualize_loss.ipynb` plots the training and test loss curves for the trained models.
- `visualize_predictions_2d.ipynb` shows Darcy-flow permeability fields, predicted solutions, and absolute error maps.
- `visualize_predictions_2d_uq.ipynb` visualizes also uncertainty maps and calibration diagnostics for the FNO2dUQ models.
- `visualize_predictions_3d.ipynb` shows predicted Navier-Stokes vorticity snapshots.
- `visualize_weights.ipynb` inspects learned spectral weights in the baseline FNO2d.


## Some notes
- The 2D Darcy scripts apply dataset normalization through `train_utils.py`.
- The 3D script does not normalize the Navier-Stokes data.
