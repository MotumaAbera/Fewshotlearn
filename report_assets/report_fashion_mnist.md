# Few-Shot Learning with Siamese Networks Report

## Project Summary

This experiment trains a Siamese Neural Network with contrastive loss on Fashion-MNIST. The model learns an embedding space where same-class images stay close together while different-class images are pushed apart. The learned embedding is then evaluated through one-shot recognition rather than conventional closed-set classification.

## Experimental Setup

- Dataset: Fashion-MNIST
- Epochs run: 1
- Batch size: 64
- Train pairs per epoch: 1200
- Validation pairs: 240
- Embedding dimension: 32
- Contrastive margin: 1.0
- One-shot setting: 5-way, 20 trials

## Quantitative Results

- Best validation loss: 0.2280
- Final train pair accuracy: 0.6267
- Final validation pair accuracy: 0.6167
- One-shot accuracy: 0.6500

## Learning Dynamics

The training curve is saved in `report_assets/loss_curve_fashion_mnist.png`. A healthy run should show decreasing training loss together with stable or improving validation loss. If validation loss stops improving quickly, that usually indicates either a small model capacity ceiling or that the sampled negative pairs are too easy.

## Embedding Space Analysis

The t-SNE projection saved in `report_assets/tsne_plot_fashion_mnist.png` provides a low-dimensional view of the learned embedding space. Effective metric learning should create visible clusters for individual classes, while visually similar classes remain closer to each other. This behavior is expected because t-SNE preserves local neighborhood structure more than global geometry.

## One-Shot Failure Analysis

The model made 7 errors out of 20 one-shot trials.

Most common confusion patterns:
- Trouser -> Dress: 2 cases
- Pullover -> Shirt: 1 cases
- Dress -> Trouser: 1 cases
- Dress -> Shirt: 1 cases
- Ankle boot -> Sneaker: 1 cases

Hardest failure cases are saved in `outputs/results/one_shot_failure_analysis_fashion_mnist.csv`.

## Discussion

Few-shot learning is useful when annotated data is limited, expensive, or constantly changing. Instead of learning only a fixed classifier boundary, the Siamese network learns a reusable similarity function. That is especially useful for verification, retrieval, anomaly matching, and low-resource recognition tasks.

For this project, Fashion-MNIST offers a fast and controlled benchmark. MNIST-style datasets let the model converge quickly on CPU, which makes them practical for university coursework and rapid experimentation. The main limitation is that these datasets are relatively simple compared with real-world image variation, so they can overestimate how well a method will scale without harder negative mining or stronger feature extractors.

## Reproducibility

Run the full experiment with:

```bash
python main.py --dataset fashion_mnist
```

Launch the frontend with:

```bash
streamlit run app.py
```
