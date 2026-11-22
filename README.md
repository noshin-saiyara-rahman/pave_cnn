# Basic CNN (from scratch) on the BD-PaveSurface V2 dataset

A small convolutional neural network, built from scratch in PyTorch, that classifies pavement surface images into three classes: **Crack**, **Good** and **Pothole**. Written for a coursework assignment; the goal was to implement the CNN and discuss its results.

**Dataset:** BD-PaveSurface V2 - https://zenodo.org/records/21855576
(not included in this repository; download and unzip it into a folder named `data`).

The processed set used here has 12,000 RGB images (3 classes, 4,000 per class). It uses the split provided by the dataset authors:

| Split | Images | Notes |
|---|---|---|
| Train | 9,300 | includes 3,000 dataset-provided augmented images |
| Validation | 1,350 | original images only |
| Test | 1,350 | original images only |

## Repository contents

| File / folder | Description |
|---|---|
| `inspect_dataset.py` | Looks inside the dataset folder: folder tree, image counts, sizes, file types. |
| `train_cnn.py` | Loads the data, trains the CNN and evaluates it on the test set. |
| `outputs/` | Results of the full run (see below). |
| `run3_console_log.txt` | Console output of the full run. |

### Contents of `outputs/`
- `results.json` - metrics for both models (best validation accuracy, test accuracy, macro F1, epochs, time) and the run setup
- `classification_report.txt` - precision / recall / F1 per class (with_aug model)
- `confusion_matrix.png` - confusion matrix (with_aug model)
- `learning_curves.png` - training and validation loss/accuracy per epoch
- `misclassified_examples.png` - test images the with_aug model got wrong
- `split_used.csv` - which image belongs to which split
- `cnn_with_aug.pt` - saved weights of the with_aug model

## Model

- 4 convolutional blocks, each with two 3x3 convolutions (Conv - BatchNorm - ReLU) followed by 2x2 max pooling, with 32, 64, 128 and 256 filters
- Global average pooling, then a classifier: Dropout (0.4) - Linear(256, 128) - ReLU - Dropout (0.4) - Linear(128, 3)
- **1,207,459** trainable parameters

## Training setup

| Setting | Value |
|---|---|
| Image size | 64 x 64 |
| Epochs | 5 |
| Optimizer | AdamW, learning rate 0.001, weight decay 1e-4 |
| Learning rate scheduler | ReduceLROnPlateau (factor 0.5, patience 2) |
| Regularization | Dropout (0.4), weight decay, BatchNorm |
| Loss | Cross-entropy |
| Early stopping | patience 6 (not triggered in 5 epochs) |
| Model selection | best validation-accuracy epoch |
| Seed | 42 |
| Hardware | CPU only |

Two models were trained and compared:
- **with_aug** - the training set plus extra random on-the-fly augmentation (horizontal flip, small rotation/shift/zoom, brightness and contrast jitter)
- **no_aug** - the same training set without the extra on-the-fly augmentation

Both use the same 9,300 training images and are evaluated on the same 1,350 test images.

## Results (test set)

| Model | Test accuracy | Macro F1 | Best validation accuracy |
|---|---|---|---|
| with_aug | 0.9630 | 0.9630 | 0.9637 |
| no_aug | 0.9933 | 0.9933 | 0.9867 |

Per-class results for the with_aug model:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Crack | 0.957 | 0.938 | 0.947 |
| Good | 0.939 | 0.962 | 0.951 |
| Pothole | 0.993 | 0.989 | 0.991 |

Main confusions (with_aug): 28 Crack predicted as Good, 14 Good as Crack, 3 Good as Pothole, 5 Pothole as Crack.

**Note:** the classification report, confusion matrix, misclassified examples and the saved `.pt` weights in `outputs/` belong to the **with_aug** model. Only its weights were saved. The no_aug model's metrics are in `results.json` and in the console log.

## How to reproduce

1. Create a Python environment (Python 3.11) with: `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `pillow`.
2. Download the dataset from Zenodo and unzip it into a folder named `data` next to the scripts.
3. Run from the project folder:

```
python inspect_dataset.py data
python train_cnn.py --data data --img-size 64 --epochs 5
```

A quick pipeline test (results not meaningful) can be run with:

```
python train_cnn.py --data data --quick
```

Training on a CPU takes roughly 20 minutes for both models at this setting.
