# TF Flowers: data preparation

Data preparation for EN3150 Assignment 03. Source:
[TensorFlow Datasets TF Flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers).
The dataset has 3,670 RGB images across five flower classes.

Prepared split counts (seed 42):

| Label | Class | Training | Validation | Test |
|---|---|---:|---:|---:|
| 0 | dandelion | 629 | 135 | 134 |
| 1 | daisy | 443 | 95 | 95 |
| 2 | tulips | 559 | 120 | 120 |
| 3 | sunflowers | 489 | 105 | 105 |
| 4 | roses | 449 | 96 | 96 |
| | **Total** | **2,569** | **551** | **550** |

## Run (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe prepare_data.py
```

If Windows reports a path-length error while installing TensorFlow, use a
shorter virtual environment path (the environment used for this preparation):

```powershell
python -m venv C:\tf-flowers-venv
C:\tf-flowers-venv\Scripts\python.exe -m pip install -r requirements.txt
C:\tf-flowers-venv\Scripts\python.exe prepare_data.py
```

The first run downloads the dataset (approximately 218 MiB). The raw TFDS
cache and prepared files stay in `data/`, which is excluded from Git.

Images are resized directly to 64 × 64 using bilinear interpolation with
antialiasing. Direct resizing can alter aspect ratios. A seeded (42), stratified
split preserves class proportions: each class assigns rounded 70% to training,
rounded 15% to validation, and the remainder to testing. Integer rounding means
the overall percentages may differ slightly from 70/15/15. Every image is used
once, with disjoint source indices checked before saving.

Outputs:

- `data/train.npz`, `validation.npz`, `test.npz`: RGB uint8 images, integer labels,
  and source indices in the pinned TFDS version's deterministic iteration order.
- `data/summary.json`: class mapping, preprocessing settings, and split counts.
- `data/split_manifest.csv`: split membership for every source image.
- `data/training_preview.png`: labeled examples after resizing.

## Use in training

```python
from prepare_data import load_datasets

train_ds, val_ds, test_ds = load_datasets()
# Images: float32 [0, 1], shape (batch, 64, 64, 3).
# Labels: integer class IDs; use sparse categorical cross-entropy.
# model.fit(train_ds, validation_data=val_ds, epochs=20)
```

Only training data is shuffled. No augmentation is applied during preparation;
add it to training only if needed. Validation and test images are deterministic.
Use the same saved splits for all four assignment models. For pretrained models,
call `load_datasets(normalize=False)` and apply the chosen architecture's own
preprocessing to the resulting float32 [0,255] images. Do not normalize twice.
Keep the test split reserved for final evaluation.
