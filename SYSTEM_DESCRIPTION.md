# System Description

## Overview

This project is a few-shot image similarity system built around a Siamese Neural Network in PyTorch. Its main goal is to learn an embedding space in which:

- images from the same class are close together
- images from different classes are farther apart

Instead of training a standard classifier that predicts a fixed class ID directly, the system learns a reusable similarity function. That makes it suitable for one-shot or few-shot recognition, where the model must compare a query image against a small support set.

The repository supports both:

- a command-line training pipeline
- a Streamlit dashboard for training, inspecting metrics, and exploring image similarity

## What The System Does

At a high level, the system:

1. Loads one of the supported datasets.
2. Converts that dataset into positive and negative image pairs.
3. Trains a Siamese network with contrastive loss.
4. Saves the best model checkpoint based on validation loss.
5. Evaluates the trained model with one-shot N-way trials.
6. Exports metrics, figures, failure analysis, and a report-ready markdown summary.

## Supported Datasets

The current supported dataset names are:

- `mnist`
- `fashion_mnist`
- `fashion_products_small`
- `amharic_handwritten`
- `ethiopian_textile_local`

### Dataset Types

| Dataset | Source | Input Channels | Image Size | Notes |
|---|---|---:|---:|---|
| `mnist` | `torchvision.datasets.MNIST` | 1 | 28x28 | Auto-download |
| `fashion_mnist` | `torchvision.datasets.FashionMNIST` | 1 | 28x28 | Auto-download |
| `fashion_products_small` | Hugging Face `ashraq/fashion-product-images-small` | 3 | 96x96 | Filtered to apparel and top 8 article types |
| `amharic_handwritten` | Hugging Face `Yaredoffice/geez-characters` | 1 | 32x32 | Downloaded as zip and extracted locally |
| `ethiopian_textile_local` | Local `ImageFolder` dataset | 3 | 128x128 | Requires `train/` and `test/` folders |

## Main Entry Points

### Command-line entrypoint

File: `main.py`

Purpose:

- parse a small set of command-line overrides
- build a `Config`
- prepare the runtime environment
- run the full experiment

Typical usage:

```bash
python main.py
python main.py --dataset fashion_mnist
python main.py --epochs 8 --batch-size 128 --learning-rate 0.0005 --trials 300
```

### Streamlit frontend

File: `app.py`

Purpose:

- provide a UI for training
- surface saved metrics and CSV outputs
- display generated figures
- let the user compare two test images through the learned embedding distance

Typical usage:

```bash
streamlit run app.py
```

## Architecture Summary

The system is organized into small modules with clear responsibilities:

| File | Responsibility |
|---|---|
| `config.py` | Central configuration, dataset validation, naming conventions |
| `dataset.py` | Dataset loading, preprocessing, pair generation, dataloader creation |
| `model.py` | Siamese encoder and embedding network |
| `loss.py` | Contrastive loss |
| `train.py` | Training loop, validation loop, checkpoint saving, early stopping |
| `evaluate.py` | Embedding extraction, one-shot trials, failure analysis |
| `visualize.py` | Loss curve, sample pairs, t-SNE, one-shot example plots |
| `report.py` | Report markdown generation |
| `pipeline.py` | End-to-end orchestration |
| `utils.py` | Reproducibility, JSON/CSV helpers, directory creation |
| `main.py` | CLI launcher |
| `app.py` | Streamlit frontend |

## End-to-End Execution Flow

The full experiment path is:

1. `main.py` or `app.py` creates a `Config`.
2. `pipeline.prepare_environment()` creates required directories and fixes random seeds.
3. `dataset.create_datasets()` builds raw datasets plus Siamese pair dataloaders.
4. `pipeline.run_experiment()` creates:
   - `SiameseNetwork`
   - `ContrastiveLoss`
   - Adam optimizer
5. A sample training batch is plotted as an example-pair figure.
6. `train.train_model()` runs training and validation across epochs.
7. The best checkpoint is reloaded from disk.
8. `evaluate.evaluate_model()` runs one-shot evaluation and embedding extraction.
9. `visualize.py` produces the figures.
10. Metrics, CSV tables, failure summaries, and markdown reports are written to disk.

## Configuration Model

The `Config` dataclass is the central control surface for the project.

### Important default values

| Setting | Default |
|---|---:|
| `seed` | 42 |
| `dataset_name` | `mnist` |
| `batch_size` | 64 |
| `num_workers` | 0 |
| `train_pairs_per_epoch` | 12000 |
| `val_pairs` | 2000 |
| `embedding_dim` | 32 |
| `learning_rate` | 0.001 |
| `weight_decay` | 0.0001 |
| `epochs` | 5 |
| `margin` | 1.0 |
| `early_stopping_patience` | 3 |
| `one_shot_trials` | 200 |
| `one_shot_way` | 5 |
| `tsne_samples` | 1000 |
| `log_interval` | 50 |

### Config behavior

`Config` also:

- normalizes dataset names by lowercasing and replacing `-` with `_`
- rejects unsupported dataset names with a `ValueError`
- sets image size and input channels based on dataset
- generates dataset-specific output filenames such as:
  - `siamese_<dataset>.pt`
  - `training_history_<dataset>.json`
  - `evaluation_metrics_<dataset>.json`
  - `results_table_<dataset>.csv`

## Dataset Layer

The dataset layer does more than just load images. It transforms ordinary labeled datasets into pairwise training data suitable for Siamese learning.

### Raw dataset loading

There are three dataset-loading strategies in the code:

- `torchvision` digit/fashion datasets for `mnist` and `fashion_mnist`
- Hugging Face datasets for `fashion_products_small`
- local `ImageFolder` datasets for `amharic_handwritten` and `ethiopian_textile_local`

### Transform pipeline

All datasets use the `get_transforms()` pipeline:

1. resize to the dataset-specific size
2. convert to grayscale when `input_channels == 1`
3. convert to tensor
4. normalize using dataset-specific mean and standard deviation

### Pair generation

The class `SiamesePairDataset` wraps a standard labeled dataset and generates pairs on demand.

For each requested sample:

1. choose a random anchor image
2. randomly decide whether to form a positive or negative pair
3. if positive, choose another image from the same class
4. if negative, choose an image from a different class

Label semantics are:

- `0.0` means positive pair, same class
- `1.0` means negative pair, different class

That convention is important because the contrastive loss uses it directly.

### Deterministic sampling

Pair generation is deterministic for a given configuration because each sample uses:

- `np.random.default_rng(self.seed + index)`

That means the same dataset index produces the same pair for a fixed seed, which improves reproducibility.

### Dataset bundle

Each loader function returns a `DatasetBundle` containing:

- `train_loader`
- `val_loader`
- `train_dataset`
- `val_dataset`
- `raw_train`
- `raw_test`
- `class_names`

This lets later stages use both pairwise dataloaders and the original raw dataset for one-shot evaluation and visualization.

## Model Architecture

The learning model is defined in `model.py`.

### EmbeddingNetwork

This is the shared encoder used by both Siamese branches.

Architecture:

1. `Conv2d(input_channels, 32, kernel_size=3, padding=1)`
2. `ReLU`
3. `MaxPool2d(2)`
4. `Conv2d(32, 64, kernel_size=3, padding=1)`
5. `ReLU`
6. `MaxPool2d(2)`
7. `Conv2d(64, 128, kernel_size=3, padding=1)`
8. `ReLU`
9. `AdaptiveAvgPool2d((1, 1))`
10. `Flatten`
11. `Linear(128, 64)`
12. `ReLU`
13. `Linear(64, embedding_dim)`
14. L2 normalization with `F.normalize(..., p=2, dim=1)`

### SiameseNetwork

The Siamese wrapper contains one encoder instance and exposes:

- `forward_once(x)` for single-image embedding extraction
- `forward(left, right)` for paired embedding computation

Because both images pass through the same encoder weights, the network learns a consistent metric space.

## Loss Function

The training objective is contrastive loss, implemented in `loss.py`.

For a pair of embeddings:

- positive pairs are penalized by squared distance
- negative pairs are penalized only when their distance is smaller than the margin

In code terms:

- positive term: `(1 - label) * distance^2`
- negative term: `label * max(0, margin - distance)^2`

Default margin:

- `1.0`

## Training Loop

Training logic lives in `train.py`.

### Epoch processing

`run_epoch()` handles both training and validation modes.

For each batch:

1. move the left and right images plus pair labels to the target device
2. run the model on the pair
3. compute contrastive loss
4. backpropagate and update parameters if in training mode
5. compute embedding distances
6. convert distances into binary predictions using a fixed threshold

### Pair accuracy rule

The current pair-accuracy decision rule is:

- predict negative (`1`) when `distance >= 0.5`
- predict positive (`0`) when `distance < 0.5`

This threshold is hard-coded and independent of the loss margin.

### Checkpointing and early stopping

`train_model()`:

- tracks train and validation loss and accuracy
- saves the best model whenever validation loss improves
- stops early after `early_stopping_patience` consecutive non-improving epochs

The best checkpoint is saved to:

- `outputs/models/<model_name>`

The epoch-wise history is saved to:

- `outputs/results/<history_name>`

## Evaluation Logic

Evaluation is implemented in `evaluate.py`.

### One-shot trials

The system evaluates by running repeated one-shot N-way matching tasks.

Per trial:

1. sample `N` candidate classes, where `N = one_shot_way`
2. pick one query class among classes that have at least two samples
3. choose one query image
4. build a support set with one support image from each candidate class
5. embed the query and every support image
6. compute Euclidean distances
7. predict the label of the nearest support image

### Effective N-way setting

The implementation computes:

- `effective_way = min(requested_way, number_of_classes)`

This prevents invalid sampling when the dataset has fewer classes than requested.

### Embedding extraction

For t-SNE visualization, the system also extracts embeddings from up to:

- `config.tsne_samples`

### Failure analysis

The evaluation stage builds:

- a confusion summary over failed one-shot trials
- a list of the 10 hardest failures, sorted by smallest incorrect distance

This is useful for understanding which classes are most visually confusable in the learned embedding space.

## Visualization Layer

The project generates several figures through `visualize.py`.

### Generated plots

1. `sample_pairs_<dataset>.png`
   - shows example positive and negative training pairs
2. `loss_curve_<dataset>.png`
   - plots train and validation contrastive loss over epochs
3. `tsne_plot_<dataset>.png`
   - shows a 2D projection of embeddings
4. `one_shot_examples_<dataset>.png`
   - shows query/support examples from one-shot evaluation

### How figures are saved

Each figure is saved twice:

- once in `outputs/figures/`
- once in `report_assets/`

This is handled by a simple copy helper after the main file is written.

### Current implementation note

The t-SNE plotting function currently uses a colorbar labeled `Digit Class` with ticks `0..9`, even though the system now supports non-digit datasets as well. That is a presentation limitation, not a training limitation.

## Report Generation

The file `report.py` produces a markdown report for each run.

The generated report includes:

- project summary
- experiment settings
- quantitative results
- learning dynamics explanation
- embedding-space explanation
- one-shot failure analysis summary
- discussion and reproducibility notes

Reports are saved both to:

- the project root as `report_<dataset>.md`
- `report_assets/report_<dataset>.md`

## Frontend Behavior

The Streamlit app in `app.py` turns the backend pipeline into an interactive workflow.

### Sidebar controls

The UI exposes:

- dataset choice
- epochs
- batch size
- learning rate
- one-shot trials
- N-way setting
- train pairs per epoch
- validation pairs
- embedding dimension

### Dashboard sections

The interface contains:

- a results tab for saved metrics and tables
- a similarity explorer for comparing two test images
- a figures tab for generated assets

### Similarity explorer

The similarity explorer:

1. loads the raw test dataset
2. lets the user choose two test indices
3. shows both images
4. indicates whether they are the same class
5. if a trained checkpoint exists, computes the embedding distance between them

Smaller distances mean the model considers the images more similar.

### Dataset caching

The app uses `@st.cache_resource` for dataset loading so repeated interaction does not recreate the dataset bundle every time.

## Output Files

The main output directories are:

### `outputs/models/`

- best saved model checkpoint per dataset

### `outputs/results/`

- training history JSON
- evaluation metrics JSON
- results table CSV
- failure summary JSON
- one-shot failure analysis CSV

### `outputs/figures/`

- sample pair figure
- loss curve
- t-SNE projection
- one-shot example montage

### `report_assets/`

- copies of the main figures
- copies of results tables
- copies of generated reports

## Reproducibility Features

The project contains several reproducibility measures:

- Python `random`, NumPy, and PyTorch seeds are fixed
- CUDA seeds are fixed when CUDA is available
- cuDNN is configured for deterministic behavior
- pair generation is deterministic per index for a fixed seed
- output filenames are dataset-specific and predictable

## Design Strengths

The current system has a few notable strengths:

- modular separation between data, model, training, evaluation, visualization, and reporting
- support for both grayscale and RGB datasets
- simple CPU-friendly architecture
- direct one-shot evaluation instead of only reporting pairwise training metrics
- automatic export of artifacts for coursework, demos, and reports
- both CLI and web UI interfaces

## Current Limitations And Implementation Notes

This section describes the current system as it is, not as an idealized version.

### Modeling limitations

- The encoder is intentionally small and may underfit more complex real-world datasets.
- There is no hard-negative mining.
- There is no learning-rate scheduler.
- There is no data augmentation pipeline beyond resize and normalization.

### Evaluation limitations

- Pair accuracy uses a fixed distance threshold of `0.5`.
- One-shot evaluation is based on random trials rather than a fixed benchmark split protocol.
- The one-shot support set uses one sample per class only.

### Visualization limitations

- The t-SNE colorbar is still digit-oriented in its label/ticks.
- The figures are designed more for interpretation and reporting than for scientific benchmarking rigor.

### UI and system limitations

- The Streamlit app only trains on demand; it does not manage experiments as background jobs.
- The app loads saved checkpoints from dataset-specific filenames, so changing embedding dimension without retraining can make the saved model incompatible with the requested config.
- The CLI exposes fewer overrides than the Streamlit app.

## Typical User Workflows

### Training from the command line

1. Install dependencies from `requirements.txt`.
2. Run `python main.py --dataset <name>`.
3. Wait for training, evaluation, and artifact generation to complete.
4. Inspect `outputs/` and the generated `report_<dataset>.md`.

### Training from the dashboard

1. Run `streamlit run app.py`.
2. Choose dataset and hyperparameters in the sidebar.
3. Click `Run Training`.
4. Review metrics, failure tables, and figures.

### Using the system as a similarity explorer

1. Train at least one checkpoint for a chosen dataset.
2. Open the Streamlit dashboard.
3. Go to the Similarity Explorer tab.
4. Select query and support indices.
5. Inspect the displayed distance.

## File Naming Convention

Most saved artifacts are dataset-specific, for example:

- `siamese_mnist.pt`
- `siamese_fashion_mnist.pt`
- `evaluation_metrics_amharic_handwritten.json`
- `results_table_ethiopian_textile_local.csv`

This keeps experiments from different datasets separate without requiring a separate run directory per experiment.

## Dependencies

The main Python dependencies are:

- `torch`
- `torchvision`
- `matplotlib`
- `scikit-learn`
- `numpy`
- `pandas`
- `streamlit`
- `datasets`
- `huggingface-hub`

## In One Sentence

This system is a modular Siamese few-shot learning pipeline that trains on pairwise similarity, evaluates through one-shot matching, and packages the results into both a command-line workflow and an interactive Streamlit dashboard.
