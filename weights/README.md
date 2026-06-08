# Weights Folder

Put the **public model files** that you want to upload to GitHub in this folder.

Recommended files:

- `mobilenetv2_car_color_best.pt`
- `mobilenetv2_car_color_last.pt`
- `class_to_idx.json`
- `summary.json`

Recommended source:

- copy `best.pt` from a finished training run
- copy `class_to_idx.json` from the same run
- copy `summary.json` from the same run

Example after training:

```powershell
New-Item -ItemType Directory -Force weights
Copy-Item runs\mobilenetv2_car_color\best.pt weights\mobilenetv2_car_color_best.pt
Copy-Item runs\mobilenetv2_car_color\last.pt weights\mobilenetv2_car_color_last.pt
Copy-Item runs\mobilenetv2_car_color\class_to_idx.json weights\class_to_idx.json
Copy-Item runs\mobilenetv2_car_color\summary.json weights\summary.json
```
