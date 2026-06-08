# Color MIBNETV2

Train a `MobileNetV2` classifier to predict **car color from CCTV images** using PyTorch and NVIDIA GPU acceleration.

This project currently focuses on **15 color classes**:

`beige`, `black`, `blue`, `brown`, `gold`, `green`, `grey`, `orange`, `pink`, `purple`, `red`, `silver`, `tan`, `white`, `yellow`

## Project Contents

- `train_mobilenetv2.py` - training script for MobileNetV2
- `requirements.txt` - Python dependencies
- `dataset/` - local dataset folder, intentionally excluded from Git tracking
- `runs/` - local training outputs, intentionally excluded from Git tracking
- `weights/` - curated model files intended for public release

## Features

- Pretrained `MobileNetV2` backbone from `torchvision`
- Automatic CUDA detection
- Mixed precision training with AMP on GPU
- Class-weighted loss for imbalanced classes
- Validation-based checkpoint saving
- Early stopping
- `train / val / test` evaluation
- Output files: `best.pt`, `last.pt`, `history.csv`, `summary.json`, `class_to_idx.json`

## Requirements

- Python 3.10+
- NVIDIA GPU recommended
- Tested locally with:
  - `torch 2.9.1+cu128`
  - `NVIDIA GeForce RTX 3070`

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Dataset Structure

The script expects an ImageFolder-style directory layout:

```text
dataset/
  train/
    beige/
    black/
    ...
  val/
    beige/
    black/
    ...
  test/
    beige/
    black/
    ...
```

### Dataset Summary

| Split | Samples | Classes |
| --- | ---: | ---: |
| train | 7,267 | 15 |
| val | 1,550 | 15 |
| test | 1,556 | 15 |

Per-class counts:

| Class | Train | Val | Test |
| --- | ---: | ---: | ---: |
| beige | 421 | 90 | 90 |
| black | 406 | 86 | 87 |
| blue | 742 | 159 | 159 |
| brown | 565 | 121 | 121 |
| gold | 210 | 45 | 45 |
| green | 563 | 120 | 121 |
| grey | 428 | 91 | 92 |
| orange | 534 | 114 | 114 |
| pink | 483 | 103 | 103 |
| purple | 536 | 114 | 115 |
| red | 637 | 136 | 136 |
| silver | 362 | 77 | 77 |
| tan | 400 | 85 | 86 |
| white | 403 | 86 | 86 |
| yellow | 577 | 123 | 124 |

## Training

Default training is already configured for this project:

- dataset path: `dataset`
- epochs: `25`
- default batch size: `32`
- default output dir: `runs/mobilenetv2_car_color`

Run training:

```powershell
python train_mobilenetv2.py --data-dir dataset --epochs 25 --batch-size 32 --num-workers 4 --output-dir runs/mobilenetv2_car_color
```

If your GPU memory has headroom, you can try:

```powershell
python train_mobilenetv2.py --data-dir dataset --epochs 25 --batch-size 64 --num-workers 4 --output-dir runs/mobilenetv2_car_color
```

## Main Arguments

```text
--data-dir            Dataset root directory
--epochs              Number of training epochs
--batch-size          Batch size
--img-size            Input image size
--lr                  Learning rate
--weight-decay        AdamW weight decay
--num-workers         DataLoader workers
--seed                Random seed
--output-dir          Directory for checkpoints and logs
--patience            Early stopping patience
--label-smoothing     Label smoothing for cross entropy
--no-amp              Disable mixed precision
```

## Output Files

After training, the script saves:

- `best.pt` - checkpoint with the best validation accuracy
- `last.pt` - latest checkpoint from the most recent epoch
- `history.csv` - epoch-by-epoch metrics
- `summary.json` - final summary metrics
- `class_to_idx.json` - class label mapping

## Publishing a Trained Model

If you want the public GitHub repo to include a finished model, do **not** upload the whole `runs/` folder.

Recommended approach:

1. Train the model normally
2. Take the final `best.pt` from the target run
3. Copy only the public artifacts into `weights/`

Example:

```powershell
New-Item -ItemType Directory -Force weights
Copy-Item runs\mobilenetv2_car_color\best.pt weights\mobilenetv2_car_color_best.pt
Copy-Item runs\mobilenetv2_car_color\last.pt weights\mobilenetv2_car_color_last.pt
Copy-Item runs\mobilenetv2_car_color\class_to_idx.json weights\class_to_idx.json
Copy-Item runs\mobilenetv2_car_color\summary.json weights\summary.json
```

This keeps the repository smaller and avoids publishing temporary checkpoints or logs that are not needed by end users.

### Which Checkpoint Should You Use?

- Use `best.pt` for inference, evaluation, export, or deployment
- Use `last.pt` if you want to resume training

## Smoke Test Result

A local 1-epoch smoke test on the current dataset completed successfully on `CUDA` with an `RTX 3070` and produced:

- validation accuracy: `0.7800`
- test accuracy: `0.7796`

These values are only a sanity check, not the final 25-epoch benchmark.

## Public Repository Notes

This repository is prepared for a **public GitHub repo** with the following assumptions:

- `dataset/` is **not included** in Git because it may be large and may contain sensitive CCTV-derived imagery
- `runs/` is **not included** in Git because it contains temporary training artifacts
- `weights/` is the place for the final trained model files that you intentionally want to publish
- source code, documentation, and curated release artifacts should be published

Before pushing publicly, review:

1. Dataset ownership and permission to share
2. CCTV/privacy constraints
3. Whether to publish trained weights separately
4. Whether to add a project license

## Example GitHub Push

```powershell
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin https://github.com/ryusr/<repo-name>.git
git push -u origin main
```

## Notes

- Color-based classification can be sensitive to lighting, white balance, reflections, and CCTV compression
- The current augmentation intentionally avoids heavy color distortion because the label itself is the target color
