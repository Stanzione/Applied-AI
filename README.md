# Museum Image Classifier

Binary image classifier distinguishing **museum interiors** from **museum exteriors**, tackled through two complementary methodologies — classical **feature extraction + shallow learners** (two paradigms) and **end-to-end convolutional neural networks** — as part of an Applied AI course project.

Group Antigravity.
Link: https://github.com/Stanzione/Applied-AI/

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Preprocessing Pipelines](#preprocessing-pipelines)
  - [Pipeline A — ResNet18 (512-d)](#pipeline-a--resnet18-512-d)
  - [Pipeline B — HOG](#pipeline-b--hog-1-764-d-supervised-only-baseline)
  - [Pipeline Comparison — ResNet18 vs HOG](#pipeline-comparison--resnet18-vs-hog)
- [Paradigm 1 — Supervised: Gradient Boosting](#paradigm-1--supervised-gradient-boosting)
- [Paradigm 2 — Semi-supervised: Manual Self-Training](#paradigm-2--semi-supervised-manual-self-training)
- [Results Summary — Feature-Based Paradigms](#results-summary--feature-based-paradigms)
- [End-to-End CNN — Custom Network vs ResNet18 Backbone](#end-to-end-cnn--custom-network-vs-resnet18-backbone)
  - [Notebook Structure](#notebook-structure)
  - [Custom CNN Architecture](#custom-cnn-architecture)
  - [Controlled Comparison — Custom CNN vs ResNet18 Extractor](#controlled-comparison--custom-cnn-vs-resnet18-extractor)
  - [Sensitivity Check — Native 224 px](#sensitivity-check--native-224-px)
  - [Saving, Loading & Application Mode](#saving-loading--application-mode)
  - [CNN Results](#cnn-results)
- [Checkpoints](#checkpoints)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [References](#references)

---

## Overview

The project attacks the same binary classification task from two methodological angles.

**1. Feature extraction + shallow learners** — a frozen ResNet18 (or HOG) front-end turns each image into a fixed feature vector, which a classical model then classifies. Two paradigms live here:

| Paradigm | Notebook | Algorithm |
|---|---|---|
| Supervised | `museum_supervised_dt_boosting.ipynb` | `GradientBoostingClassifier` + Decision Tree baselines |
| Semi-supervised | `museum_semisupervised_dt.ipynb` | `ManualSelfTrainingDT` — custom explicit self-training loop |

Both paradigms share the same feature extraction front-end (ResNet18 embeddings) and are evaluated on the same held-out validation set.

**2. End-to-end convolutional neural networks** — the network learns the representation *and* the decision boundary jointly, trained directly on pixels in PyTorch:

| Approach | Notebook | Model |
|---|---|---|
| Deep learning | `museum_cnn_colab.ipynb` | Custom CNN (from scratch) vs ResNet18 backbone — controlled comparison |

The CNN notebook runs end-to-end on Google Colab with the dataset mounted from Drive and per-epoch checkpointing, and is described in its own [section below](#end-to-end-cnn--custom-network-vs-resnet18-backbone).

---

## Repository Structure

```
AppliedAI/
├── museum_supervised_dt_boosting.ipynb     ← supervised notebook (feature-based)
├── museum_semisupervised_dt.ipynb          ← semi-supervised notebook (feature-based)
├── museum_cnn_colab.ipynb                  ← end-to-end CNN notebook (Colab)
├── drive-download-...-910Z.zip             ← ResNet18 features + trained models
└── drive-download-...-329Z.zip             ← HOG features + DT baselines
```

---

## Dataset

**Source:** Places365-Standard (MIT) — `museum_indoor` and `museum_outdoor` categories.

```
training/
├── museum-indoor/    (5 000 images)
└── museum-outdoor/   (5 000 images)

museum_validation/
├── museum-indoor/    (100 images)
└── museum-outdoor/   (100 images)

test/                 (unlabeled — final prediction target)
```

Images span `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp` formats. `MuseumDataset` (a `torch.utils.data.Dataset` subclass) loads them with deterministic ordering via `samples.sort()` to guarantee reproducible (feature, label) alignment across Colab sessions.

> Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., & Torralba, A. (2018). *Places: A 10 million image database for scene recognition.* IEEE TPAMI.

---

## Preprocessing Pipelines

### Pipeline A — ResNet18 (512-d)

A pretrained ResNet18 backbone (ImageNet weights) with the final fully-connected layer replaced by `nn.Identity()` is run **frozen** over every image to produce 512-dimensional feature vectors.

```python
backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
backbone.fc = nn.Identity()   # strip classifier head
backbone.eval()               # frozen — no gradient updates
```

Input images are resized to 224 × 224 and normalized with ImageNet mean/std. Features are extracted in batches (`BATCH_SIZE=32`) and cached to `.npz` checkpoints on first run.

> He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR. [[arXiv]](https://arxiv.org/abs/1512.03385)

### Pipeline B — HOG (~1 764-d) *(supervised only, baseline)*

HOG features are computed on each image resized to 224 × 224 grayscale. The image is divided into a grid of 8×8 cells (each cell spanning 28×28 pixels), with each cell producing a 9-bin gradient orientation histogram. Adjacent cells are grouped into 2×2 overlapping blocks and L2-Hys normalized — yielding a descriptor of approximately 1 764 dimensions that captures the local edge and gradient structure of the scene.

```python
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(28, 28),   # 224 / 28 = 8 cells per side
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    transform_sqrt=True,
)
```

> Dalal, N., & Triggs, B. (2005). *Histograms of Oriented Gradients for Human Detection.* CVPR.

### Pipeline Comparison — ResNet18 vs HOG

To justify the choice of ResNet18 as the feature extractor, the **same simple model** (`DecisionTreeClassifier(max_depth=2, gini)` plus an unconstrained `DecisionTreeClassifier(max_depth=None)`) is trained on both feature spaces and compared side by side. Because the classifier and data are identical, any performance gap isolates the contribution of the feature representation itself.

| Aspect | Pipeline A — ResNet18 | Pipeline B — HOG |
|---|---|---|
| Feature dimensions | 512-d | ~1 764-d |
| Feature type | Deep semantic embeddings | Handcrafted edge/gradient descriptors |
| Requires GPU | Recommended (feature extraction) | No |
| Extraction time | ~2 min (10k images, CPU) | ~5 min (10k images, CPU) |
| Interpretability | Low — learned representations | High — each bin corresponds to a gradient orientation |
| Expected accuracy | Higher — ImageNet pretraining transfers well to scene classification | Lower — edge descriptors miss high-level semantics |

ResNet18 features benefit from transfer learning — the backbone was pretrained on 1.2M ImageNet images and has implicitly learned to distinguish indoor from outdoor scenes through intermediate visual concepts (furniture, architecture, sky, foliage). HOG, by contrast, is a purely local descriptor that captures only gradient orientations and does not encode object-level semantics, which limits its ceiling on scene-level classification.

The comparison is checkpointed via `sup_hog_train.npz`, `sup_hog_val.npz`, `sup_hog_dt.joblib` (depth=2) and `sup_hog_dt_deep.joblib` (depth=None) so the result is reproducible without re-extraction.

---

## Paradigm 1 — Supervised: Gradient Boosting

**Notebook:** `museum_supervised_dt_boosting.ipynb`

### Algorithm

`sklearn.ensemble.GradientBoostingClassifier` fits an additive ensemble of shallow decision trees by minimizing the binomial deviance loss in a stagewise fashion. Each new tree fits the pseudo-residuals of the current ensemble (negative gradient of the loss), corrected by a learning rate `η`.

> Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine.* Annals of Statistics, 29(5), 1189–1232.

### 7 Configurations (2 DT baselines + 5 Gradient Boosting)

The 2 plain Decision Tree baselines establish the performance floor (without boosting) so the gain from the ensemble can be quantified directly. The 5 GB configurations vary **five hyperparameters** simultaneously: `n_estimators`, `learning_rate`, `max_depth`, `subsample` (stochastic GB), and `min_samples_leaf`.

| Config | `n_estimators` | `learning_rate` | `max_depth` | `subsample` | `min_samples_leaf` | Role |
|---|---|---|---|---|---|---|
| **DT_depth1_gini**             | — (single DT) | — | 1 | — | — | Baseline floor (stump) |
| **DT_depth2_gini**             | — (single DT) | — | 2 | — | — | Baseline floor (depth-2) |
| GB_n10_lr01_d2_sub07           | 10  | 0.10 | 2 | 0.70 | 1 | Minimal GB — under-fit reference |
| GB_n50_lr005_d3_sub07          | 50  | 0.05 | 3 | 0.70 | 1 | Slow learner — many small steps |
| GB_n100_lr01_d2_sub05          | 100 | 0.10 | 2 | 0.50 | 1 | Aggressive stochastic GB |
| GB_n50_lr05_d3_leaf5           | 50  | 0.50 | 3 | 1.00 | 5 | High lr + leaf regularization |
| GB_n75_lr01_d4_sub07_leaf3     | 75  | 0.10 | 4 | 0.70 | 3 | Balanced depth-4 with leaf reg |

Each config is evaluated with **5-fold Stratified Cross-Validation** on the training set, then retrained on the **full** training set and evaluated on the held-out validation set. This guarantees the final reported metrics come from a model trained on 100% of the available labeled data.

### Key Functions

| Function | Purpose |
|---|---|
| `extract_resnet_features()` | Runs frozen ResNet18 over a dataset; loads from `.npz` checkpoint if available |
| `extract_hog_features()` | Computes HOG descriptors; caches to `.npz` |
| `run_pipeline()` | Scales features → fits all configs with 5-fold CV → saves checkpoint |
| `f1_confidence_interval()` | Bootstrap resampling (n=2 000) to compute a 95% CI on macro-F1 |
| `build_report()` | Generates the multi-panel dark-theme figure (bars, ROC, heatmap, CM, per-class P/R, table) |
| `predict_test()` | Loads best model by val-F1; writes per-image predictions to CSV |
| `predict_single_image()` | Live single-image demo with probability bar chart (bonus) |

### Confidence Interval

The margin of error on macro-F1 is estimated via **non-parametric bootstrap** (2 000 resamples with replacement), using the percentile method:

```python
def f1_confidence_interval(y_true, y_pred, n_boot=2000, alpha=0.05, seed=42):
    rng  = np.random.default_rng(seed)
    n    = len(y_true)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot.append(f1_score(y_true[idx], y_pred[idx], average='macro', zero_division=0))
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return (hi - lo) / 2
```

---

## Paradigm 2 — Semi-supervised: Manual Self-Training

**Notebook:** `museum_semisupervised_dt.ipynb`

### Algorithm

`ManualSelfTrainingDT` implements Yarowsky-style iterative self-training with an explicit Python loop (no sklearn wrapper), making each iteration fully observable and diagnosable.

> Yarowsky, D. (1995). *Unsupervised Word Sense Disambiguation Rivaling Supervised Methods.* ACL.

**Algorithm steps per iteration:**

1. Train a `DecisionTreeClassifier` on the current labeled pool
2. Predict class probabilities on the unlabeled pool
3. Score each sample: `score = max(proba)`
4. Select candidates above `threshold`; optionally keep only top `k_best`
5. Move accepted samples to the labeled pool with their pseudo-labels
6. Check convergence — stop if any criterion fires

**Four convergence criteria:**

| Criterion | Meaning |
|---|---|
| `no_confident_samples` | No unlabeled sample exceeded the threshold |
| `predictions_stable` | Fixed-size snapshot of predictions unchanged from previous iter |
| `unlabeled_exhausted` | Unlabeled pool is empty |
| `max_iter_reached` | Hard cap on iterations fired |

The `predictions_stable` check uses a **fixed-size snapshot array** indexed by original positions (not the shrinking `X_unl` array) to guarantee the comparison is valid across iterations as confident samples are removed from the pool.

### Semi-supervised Split

`make_semisup_split(y, labeled_ratio, rng_seed)` performs a **stratified split** of the training set: a given fraction per class receives real labels; the rest are masked as unlabeled (`y = -1`). The validation set is always fully labeled and used only for final evaluation.

### 5 Configurations

| Config | `labeled_ratio` | `dt_depth` | `criterion` | `threshold` | `k_best` | Tests |
|---|---|---|---|---|---|---|
| ST_thr075_d2_lab20   | 20% | 2 | gini    | 0.75 | —   | Baseline |
| ST_thr095_d6_lab10   | 10% | 6 | gini    | 0.95 | —   | Low data + strict threshold |
| ST_kbest500_d4_lab20 | 20% | 4 | gini    | 0.75 | 500 | Gradual top-K convergence |
| ST_entropy_d4_lab30  | 30% | 4 | entropy | 0.60 | —   | Entropy criterion |
| ST_thr060_d6_lab30   | 30% | 6 | gini    | 0.60 | —   | Aggressive pseudo-labeling |

### Key Functions

| Function | Purpose |
|---|---|
| `ManualSelfTrainingDT.fit()` | Full self-training loop with iteration-level convergence logging |
| `ManualSelfTrainingDT.predict()` / `predict_proba()` | Delegates to `self.clf_` (final DT) |
| `make_semisup_split()` | Stratified labeled/unlabeled split |
| `run_semisup_pipeline()` | Scales → runs all configs → incremental checkpoint |
| `build_report()` | 6-row dashboard: metrics, ΔF1 heatmap, ROC, pseudo-label accuracy, per-class P/R, CM, convergence counts, summary table |
| `plot_self_training_convergence()` | Labeled-pool growth and confident-samples-per-iter curves |
| `predict_single_image()` | Live single-image demo with probability bar chart |

### ΔF1 — Measuring the True Gain of Self-Training

Each config is compared against a **labeled-only baseline** DT (same depth and criterion, trained only on the initial labeled fraction):

```
ΔF1 = F1(semi-supervised model) − F1(labeled-only baseline)
```

A positive ΔF1 indicates the self-training loop genuinely improved over what labeled data alone could achieve.

---

## Results Summary — Feature-Based Paradigms

### Supervised (ResNet18 + Gradient Boosting)

All metrics on the held-out validation set (n = 200, balanced 100/100). 95% bootstrap confidence interval on macro-F1 ≈ ±0.022.

| Config | Val Acc | Val F1 (macro) | Val ROC-AUC | CV F1 ± std | Time |
|---|---|---|---|---|---|
| DT_depth1_gini             | 0.9100 | 0.9098 | 0.9100 | 0.8777 ± 0.0039 | 8 s |
| DT_depth2_gini             | 0.9150 | 0.9147 | 0.9563 | 0.8858 ± 0.0053 | 10 s |
| GB_n10_lr01_d2_sub07       | 0.9750 | 0.9750 | 0.9948 | 0.9608 ± 0.0034 | 472 s |
| GB_n50_lr005_d3_sub07      | 0.9750 | 0.9750 | 0.9924 | 0.9523 ± 0.0035 | 493 s |
| GB_n100_lr01_d2_sub05      | 0.9600 | 0.9600 | 0.9948 | 0.9626 ± 0.0035 | 455 s |
| **GB_n50_lr05_d3_leaf5**   | **0.9750** | **0.9750** | **0.9961** | 0.9623 ± 0.0040 | 733 s |
| GB_n75_lr01_d4_sub07_leaf3 | 0.9700 | 0.9700 | 0.9964 | 0.9668 ± 0.0016 | 942 s |

**Best supervised model:** `GB_n50_lr05_d3_leaf5` — highest val F1 (0.9750) tied with two others, top-tier ROC-AUC (0.9961), and 28 % faster than the next-best regularized config. Gain over the strongest DT baseline: **ΔF1 ≈ +0.060**, clearly above the bootstrap CI of ±0.022 → statistically significant.

The 3 configs that hit val F1 = 0.9750 (`GB_n10_lr01_d2_sub07`, `GB_n50_lr005_d3_sub07`, `GB_n50_lr05_d3_leaf5`) sit at the val-set noise ceiling; the choice between them is dominated by ROC-AUC and CV F1 stability.

### HOG vs ResNet18 — Same Decision Tree

To isolate the contribution of the feature representation, the same `DecisionTreeClassifier(max_depth=2, gini)` is trained on both feature spaces. The performance gap quantifies how much the deep representation contributes beyond classical edge descriptors.

*(Run the supervised notebook to populate the exact numbers — figure saved to `outputs_supervised/hog_vs_resnet.png`.)*

### Semi-supervised (ResNet18 + ManualSelfTrainingDT)

| Config | Val F1 | ΔF1 | Stop Reason | Iterations |
|---|---|---|---|---|
| **ST_thr060_d6_lab30**   | **0.9449** | −0.005 | no_confident_samples | 5 |
| ST_kbest500_d4_lab20     | 0.9300 | +0.010 | no_confident_samples | 17 |
| ST_entropy_d4_lab30      | 0.9247 | +0.000 | unlabeled_exhausted  | 3 |
| ST_thr075_d2_lab20       | 0.9200 | +0.000 | no_confident_samples | 3 |
| ST_thr095_d6_lab10       | 0.9050 | +0.010 | no_confident_samples | 5 |

**Best semi-supervised model:** `ST_thr060_d6_lab30` — highest val F1 (0.9449), highest pseudo-label accuracy (0.924). The marginal negative ΔF1 reflects that the labeled-only DT baseline with 30 % of labels already nears the ceiling of what a depth-6 DT can extract from ResNet features; the self-training loop merely confirms this rather than substantially improving it.

**Best convergence demonstration:** `ST_kbest500_d4_lab20` — 16 iterations of +500 samples each, with `avg_score` decaying monotonically from 0.956 to 0.711 until `max_score < threshold` fires `no_confident_samples`. This config provides the strongest empirical demonstration of the rubric's "we can see the loop" requirement.

---

## End-to-End CNN — Custom Network vs ResNet18 Backbone

**Notebook:** `museum_cnn_colab.ipynb` · runs end-to-end on **Google Colab** (dataset mounted from Google Drive, per-epoch checkpointing to Drive).

Unlike the feature-extraction paradigms above, this approach learns the representation and the classifier jointly, straight from pixels. The headline experiment is a **controlled comparison**: a CNN trained from scratch versus an ImageNet-pretrained ResNet18, with the two pipelines differing in *only one thing* — the feature extractor.

### Notebook Structure

| Section | Content |
|---|---|
| I | Exploratory data analysis |
| II | Preprocessing, augmentation, automated **stratified** train/val split |
| III | Custom CNN architecture |
| IV | Training / evaluation utilities (resumable checkpointing, early stopping) |
| V | Baseline custom-CNN training |
| VI | Hyperparameter search — *within-family* comparison |
| VII | ResNet18 controlled comparison |
| VIII | Final evaluation on the held-out test set |
| VIII-b | *(optional)* sensitivity check — ResNet18 at native 224 px |
| IX | Load a saved model & run batch inference on the test folder |
| X | Application mode — single-image / live demo |

### Dataset Layout (Google Drive)

The notebook expects the dataset under `MyDrive/appliedAI/` (same images as the [Dataset](#dataset) section above):

```
MyDrive/appliedAI/
├── training/            ← labeled; auto-split 80/20 into train/val (stratified) → 8 000 / 2 000
│   ├── museum-indoor/
│   └── museum-outdoor/
├── museum_validation/   ← held-out TEST set (200 images, balanced 100/100)
└── test/                ← unlabeled images (application mode → predictions CSV)
```

Trained models are written to `MyDrive/appliedAI/models/` and resumable per-epoch checkpoints to `MyDrive/appliedAI/checkpoints/`, so a Colab disconnect never loses progress — re-running the same cell resumes from the last completed epoch.

### Custom CNN Architecture

`MuseumCNN` is built in two explicit parts:

- **Feature extractor** — `n_blocks` convolutional blocks, each `Conv3×3 → BatchNorm → ReLU → (MaxPool)`, channels doubling per block (`base_channels=32`). He (Kaiming) initialization for ReLU.
- **Classifier head** — the shared `make_classifier` factory (see below).

`n_blocks`, `n_pools`, `base_channels` and `dropout` are configurable, which drives the within-family hyperparameter search in Section VI (depth, pooling, learning rate, optimizer, batch size). The best config is **auto-selected** by validation accuracy, retrained on the full split, and saved.

### Controlled Comparison — Custom CNN vs ResNet18 Extractor

Section VII is engineered so the two pipelines differ in **only the feature extractor** — same head, same input, same training regime:

```python
def make_classifier(in_ch, n_classes=2, dropout=0.5):
    # AdaptiveAvgPool(4×4) → Flatten → Dropout → Linear(in_ch*16 → 128)
    #                      → ReLU → Dropout → Linear(128 → n_classes)
```

| | Custom CNN | ResNet18 pipeline |
|---|---|---|
| Feature extractor | conv blocks trained from scratch | ResNet18 (ImageNet), `conv1…layer4`, avgpool/fc removed → 512-ch map |
| `in_ch` into the head | 128 | 512 |
| Classifier head | shared `make_classifier` | **same** shared `make_classifier` |
| Input | 128×128, ImageNet normalization | **same** 128×128 loaders & normalization |

`AdaptiveAvgPool2d((4,4))` makes the head independent of spatial size, so the identical head plugs onto either extractor. ImageNet normalization is applied to **both** pipelines — neutral for the from-scratch CNN (it uses BatchNorm) and the native format ResNet was pretrained on — guaranteeing identical inputs. Because the head, input and optimization are shared, any performance gap isolates the contribution of the **feature extractor** itself (learned-from-scratch vs ImageNet transfer).

A `FREEZE_BACKBONE` flag controls the ResNet regime:

| `FREEZE_BACKBONE` | Behavior |
|---|---|
| `False` *(default)* | Fine-tune ResNet end-to-end — mirrors the custom CNN, which is also trained end-to-end |
| `True` | Freeze the backbone, train only the head |

### Sensitivity Check — Native 224 px

Section VIII-b is optional (`RUN_SENSITIVITY = True`): it retrains the ResNet pipeline at its native **224×224** resolution — *only the input resolution changes* — and reports **ΔF1 (224 vs 128)**. A near-zero ΔF1 confirms that running the comparison at 128 px did not penalize the pretrained backbone, i.e. the gap measured in Section VII really comes from the extractor, not the input format.

### Saving, Loading & Application Mode

Per the assignment's deployment requirements:

- **Save** — each trained model is serialized with its architecture kwargs, weights, normalization stats and class names: `museum_cnn_best.pt` (custom CNN) and `museum_resnet18.pt` (ResNet pipeline).
- **Load + batch test** (Section IX) — `load_model()` reconstructs the network from the checkpoint; `predict_folder()` runs it over the unlabeled `test/` folder and writes a per-image **predictions CSV** with softmax probabilities.
- **Application mode** (Section X) — `predict_image()` classifies a single image on disk and displays the predicted class with per-class probabilities.

All models are built in **standard PyTorch** (`torchvision` counts as PyTorch); scikit-learn is used **only** for the stratified split and evaluation metrics.

### CNN Results

**Within-family hyperparameter search (Section VI)** — 7 configs, 5 short epochs each, ranked by best validation accuracy on the 2 000-image val split:

| Experiment | Variation | Best Val Acc | Params |
|---|---|---|---|
| **blocks_4**     | 4 conv blocks (winner) | **0.9250** | 914 050 |
| pool_removed     | one pooling layer removed | 0.9235 | 356 226 |
| baseline_3blocks | 3 conv blocks (reference) | 0.9195 | 356 226 |
| batch_32         | batch size 32 | 0.9150 | 356 226 |
| sgd_momentum     | SGD + momentum, lr 1e-2 | 0.9125 | 356 226 |
| lr_1e-4          | Adam, lr 1e-4 | 0.9015 | 356 226 |
| blocks_2         | 2 conv blocks | 0.8995 | 151 042 |

The 4-block CNN is auto-selected, retrained on the full split, and saved as `museum_cnn_best.pt`.

**Final held-out test (Section VII–VIII)** — 200 images, balanced 100/100:

| Model | Accuracy | F1 (macro) |
|---|---|---|
| Custom CNN (best, 4 blocks) | 0.9400 | 0.9399 |
| **ResNet18 (backbone + shared head, 128 px)** | **0.9750** | **0.9750** |

With the head, input and training regime held identical, the ImageNet-pretrained extractor beats the from-scratch CNN by **ΔF1 ≈ +0.035** — a clean measure of what transfer learning contributes, since the extractor is the only thing that changed.

**Sensitivity check (Section VIII-b)** — retraining ResNet18 at native 224 px yields the *same* F1 (0.9750), so **ΔF1 (224 − 128) = +0.0000**: the 128 px choice did not bias the comparison against the pretrained backbone.

---

## Checkpoints

Two zip archives are included to skip re-extraction and re-training on subsequent runs:

| File | Contents | Size |
|---|---|---|
| `drive-download-...-910Z.zip` | ResNet18 train/val features (`.npz`) + supervised GB models + semi-supervised models (`.joblib`) | ~19 MB |
| `drive-download-...-329Z.zip` | HOG train/val features (`.npz`) + DT baseline models (`.joblib`) | ~55 MB |

**To use:** extract the contents into the `checkpoints/` folder inside your Google Drive `appliedAI/` directory before running either notebook. Both notebooks detect these files automatically and skip the corresponding extraction or training steps.

Specifically, the supervised notebook expects:
```
checkpoints/
├── sup_resnet_train.npz        (ResNet18 features for training set)
├── sup_resnet_val.npz          (ResNet18 features for validation set)
├── sup_hog_train.npz           (HOG features for training set)
├── sup_hog_val.npz             (HOG features for validation set)
├── sup_hog_dt.joblib           (DT depth=2 trained on HOG)
├── sup_hog_dt_deep.joblib      (DT depth=None trained on HOG)
└── sup_models_resnet_gb.joblib (5 GB + 2 DT trained on ResNet18)
```

And the semi-supervised notebook reuses `sup_resnet_*.npz` plus its own `semi_models_resnet_v4.joblib`.

---

## How to Run

### Feature-based notebooks

1. Upload your dataset to `MyDrive/appliedAI/` on Google Drive following the structure above
2. Extract the checkpoint zips into `MyDrive/appliedAI/checkpoints/`
3. Open either notebook (`museum_supervised_dt_boosting.ipynb` / `museum_semisupervised_dt.ipynb`) in **Google Colab** and run all cells top to bottom

The notebooks detect the environment automatically (Colab vs. local) and mount Google Drive if needed.

For a live demo, set `IMAGE_PATH` in the "Bonus — Live Single-Image Prediction" cell of either notebook to point to any image file on disk; the cell extracts ResNet18 features, runs the best model, and displays the predicted class with per-class probability bars.

### End-to-end CNN notebook (`museum_cnn_colab.ipynb`)

1. Upload your dataset to `MyDrive/appliedAI/` following the same structure above
2. Open `museum_cnn_colab.ipynb` in **Google Colab** and run all cells top to bottom — it auto-mounts Google Drive

This notebook trains directly on pixels at 128 × 128 with ImageNet normalization, so it needs **no** precomputed feature zips. Two toggles control the experiments:

- `FREEZE_BACKBONE` (Section VII) — `False` fine-tunes ResNet18 end-to-end (default), `True` freezes the backbone and trains only the head.
- `RUN_SENSITIVITY` (Section VIII-b) — `True` retrains ResNet18 at its native 224 px to report ΔF1 vs the 128 px result.

Saved models go to `MyDrive/appliedAI/models` (`museum_cnn_best.pt`, `museum_resnet18.pt`); per-epoch checkpoints are written to `MyDrive/appliedAI/checkpoints`, so a disconnected session resumes from the last completed epoch on re-run.

---

## Dependencies

| Package | Role |
|---|---|
| `torch` + `torchvision` | ResNet18 feature extraction |
| `scikit-learn` | GradientBoostingClassifier, DecisionTreeClassifier, metrics, CV |
| `scikit-image` | HOG descriptor |
| `Pillow` | Image loading and resizing |
| `numpy` | Feature arrays and bootstrap sampling |
| `matplotlib` + `seaborn` | Report figures |
| `joblib` | Checkpoint serialization |

---

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep Residual Learning for Image Recognition.** *CVPR.* https://arxiv.org/abs/1512.03385
2. Friedman, J. H. (2001). **Greedy Function Approximation: A Gradient Boosting Machine.** *Annals of Statistics, 29*(5), 1189–1232.
3. Yarowsky, D. (1995). **Unsupervised Word Sense Disambiguation Rivaling Supervised Methods.** *ACL*, 189–196.
4. Dalal, N., & Triggs, B. (2005). **Histograms of Oriented Gradients for Human Detection.** *CVPR*, 886–893.
5. Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., & Torralba, A. (2018). **Places: A 10 million image database for scene recognition.** *IEEE TPAMI, 40*(6), 1452–1464.
6. Pedregosa, F., et al. (2011). **Scikit-learn: Machine Learning in Python.** *JMLR, 12*, 2825–2830. *(library)*
