# Few-Shot Learning with Siamese Networks for Image Similarity

This project implements a university-level few-shot learning pipeline using a Siamese Neural Network trained with contrastive loss. The goal is to learn an embedding space where same-class images stay close together and different-class images move farther apart, making one-shot recognition possible with very limited labeled examples.

The repository is designed to be:

- Easy to run on a CPU or free Google Colab environment
- Modular enough for academic discussion and experimentation
- Ready to use as a GitHub submission or final project artifact

## Project Features

- Siamese CNN encoder implemented in PyTorch
- Contrastive loss for metric learning
- Positive and negative pair generation from MNIST
- Train and validation pair accuracy tracking
- One-shot `N`-way recognition evaluation
- t-SNE visualization of learned embeddings
- Sample pair visualization and qualitative one-shot examples
- One-shot failure analysis export for error discussion
- Optional Fashion-MNIST experiment path
- Public fashion product image dataset support
- Amharic handwritten character dataset support
- Ethiopian cultural clothing and textile local dataset support
- Auto-generated report markdown for submission support
- Output figures and result tables suitable for reports

## Project Structure

```text
siamese-few-shot-learning/
|
|-- README.md
|-- requirements.txt
|-- main.py
|-- app.py
|-- config.py
|-- dataset.py
|-- model.py
|-- loss.py
|-- pipeline.py
|-- train.py
|-- evaluate.py
|-- visualize.py
|-- utils.py
|
|-- outputs/
|   |-- figures/
|   |-- models/
|   `-- results/
|
`-- report_assets/
    |-- loss_curve.png
    |-- tsne_plot.png
    |-- sample_pairs.png
    |-- one_shot_examples.png
    `-- results_table.csv
```

The files inside `report_assets/` are generated automatically after running training and evaluation.

## Installation

Create a virtual environment if you want isolation, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Run the complete pipeline:

```bash
python main.py
```

Optional overrides:

```bash
python main.py --epochs 8 --batch-size 128 --learning-rate 0.0005 --trials 300
```

Run the Fashion-MNIST extension:

```bash
python main.py --dataset fashion_mnist
```

Run a public fashion image dataset extension:

```bash
python main.py --dataset fashion_products_small
```

Run the Amharic handwritten extension:

```bash
python main.py --dataset amharic_handwritten
```

Run the Ethiopian textile extension after placing images in the local folder structure:

```bash
python main.py --dataset ethiopian_textile_local
```

## Running the Frontend

Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

The frontend lets you:

- Start training from a sidebar control panel
- Review saved metrics and results tables
- Browse generated figures
- Compare two MNIST test images using the learned embedding distance
- Switch between MNIST and Fashion-MNIST experiments
- Switch to a public clothing-image dataset
- Switch to an Amharic handwritten character dataset
- Switch to a local Ethiopian cultural clothing or textile dataset
- Inspect one-shot failure analysis tables

## Methodology

### 1. Siamese Architecture

The Siamese network contains two identical convolutional branches with shared weights. Each image is mapped into a low-dimensional embedding vector. Shared weights ensure that both inputs are processed consistently, which is essential for similarity learning.

### 2. Contrastive Loss

Given two embeddings, the model minimizes the Euclidean distance for positive pairs and penalizes negative pairs that are closer than a chosen margin:

```text
L = (1 - y) * D^2 + y * max(0, margin - D)^2
```

Where:

- `y = 0` for similar pairs
- `y = 1` for dissimilar pairs
- `D` is the Euclidean distance between the two embeddings

### 3. Pair Construction

Each training batch contains a mix of:

- Positive pairs: two images from the same digit class
- Negative pairs: two images from different digit classes

This lets the encoder learn class-aware similarity without training a standard 10-class classifier.

For the Ethiopian extensions:

- `fashion_products_small` uses a public Hugging Face fashion product image dataset with apparel article labels
- `amharic_handwritten` uses a class-labeled handwritten Ge'ez/Amharic character dataset from Hugging Face
- `ethiopian_textile_local` expects local train/test folders where each subfolder is a textile or clothing class

### 4. One-Shot Evaluation

One-shot recognition is evaluated as a 5-way matching task:

1. Sample one query image
2. Build a support set with one candidate image from each of five classes
3. Compute embedding distances
4. Select the nearest support image as the prediction

This approximates the core few-shot setting where only a handful of labeled examples are available.

## Expected Outputs

After running the project, you will obtain:

- `outputs/models/siamese_mnist.pt`: best saved model
- `outputs/results/training_history.json`: epoch-wise training history
- `outputs/results/evaluation_metrics.json`: final metrics
- `outputs/results/results_table.csv`: metrics table for the report
- `outputs/figures/sample_pairs.png`: example positive and negative pairs
- `outputs/figures/loss_curve.png`: loss curves
- `outputs/figures/tsne_plot.png`: t-SNE visualization
- `outputs/figures/one_shot_examples.png`: qualitative one-shot examples
- `outputs/results/one_shot_failure_analysis_<dataset>.csv`: confusion-style failure breakdown
- `report_<dataset>.md`: report-ready markdown summary

The same key visual outputs are copied into `report_assets/` to make report preparation easier.

## Result Interpretation Guide

### Pair Accuracy

Pair accuracy measures whether the model can correctly distinguish same-class pairs from different-class pairs using embedding distance.

### One-Shot Accuracy

One-shot accuracy is a better indicator of few-shot learning usefulness because it tests whether the learned embedding generalizes to similarity-based recognition rather than ordinary classification.

### t-SNE Analysis

If training is effective, the t-SNE projection should show visible clustering of digit classes. Some overlap is expected for digits with similar shapes such as `4` and `9`, or `3` and `5`.

### Failure Analysis

The failure analysis export highlights which classes are most often confused during one-shot recognition. This is useful for discussing whether errors come from visual similarity, insufficient hard negatives, or a limited encoder capacity.

## Discussion for Final Report

### Why Siamese Networks Matter

Siamese networks are useful when labeled data is scarce or when new classes may appear after training. Instead of learning only fixed-class decision boundaries, they learn a reusable similarity space. This is especially valuable in:

- Signature verification
- Face verification
- Medical imaging with rare conditions
- Industrial anomaly matching
- Handwritten character matching for low-resource scripts

### Strengths

- Works with limited labeled examples
- Generalizes through metric learning
- Naturally supports retrieval and matching tasks
- More flexible than closed-set classifiers

### Limitations

- Performance depends on high-quality pair sampling
- Negative mining can matter for harder datasets
- MNIST is simple, so results may overestimate performance on real-world tasks
- t-SNE is useful for intuition but not a formal proof of class separability

### Future Improvements

- Replace the CNN backbone with a stronger encoder
- Use triplet loss or supervised contrastive loss
- Add hard negative mining
- Move from MNIST to Fashion-MNIST, Omniglot, or custom data
- Evaluate `k`-shot prototypes instead of only one-shot matching
- Curate a larger labeled Ethiopian cultural clothing benchmark with more balanced regional coverage

## Suggested Report Sections

You can reuse this structure in your written report:

1. Introduction
2. Problem Statement
3. Related Background on Few-Shot Learning
4. Siamese Network Architecture
5. Contrastive Loss Formulation
6. Experimental Setup
7. Results and Visual Analysis
8. Discussion
9. Limitations and Future Work
10. Conclusion

## Reproducibility Notes

- Random seeds are fixed in `utils.py`
- MNIST downloads automatically through `torchvision.datasets`
- Paths and training defaults are centralized in `config.py`
- The Amharic dataset is downloaded from Hugging Face on first use
- The fashion product dataset is downloaded from Hugging Face on first use
- The Ethiopian textile dataset is user-supplied under `data/ethiopian_textile_local/`

## How to Reproduce Results

1. Install the dependencies with `pip install -r requirements.txt`
2. Run `python main.py --dataset mnist`
3. Inspect `outputs/results/`, `outputs/figures/`, and `report_assets/`
4. Optionally run `python main.py --dataset fashion_mnist` for the extension experiment
5. Optionally run `python main.py --dataset fashion_products_small`
6. Optionally run `python main.py --dataset amharic_handwritten`
7. Optionally place local textile images under `data/ethiopian_textile_local/` and run `python main.py --dataset ethiopian_textile_local`
8. Launch `streamlit run app.py` to interactively review the model outputs

## Dataset Sources

- `mnist`: torchvision MNIST
- `fashion_mnist`: torchvision Fashion-MNIST
- `fashion_products_small`: [ashraq/fashion-product-images-small](https://huggingface.co/datasets/ashraq/fashion-product-images-small)
- `amharic_handwritten`: [Yaredoffice/geez-characters](https://huggingface.co/datasets/Yaredoffice/geez-characters)
- `ethiopian_textile_local`: local folder dataset intended for Ethiopian cultural clothing or textile classes you curate

The Amharic dataset source describes 287 handwritten character classes and an image-classification folder layout.

## License

This project is suitable for educational and academic use.
#   F e w s h o t l e a r n  
 
