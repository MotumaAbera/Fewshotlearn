# Few-Shot Learning with Siamese Networks Report

## Project Summary

This experiment trains a Siamese Neural Network with contrastive loss on Fashion Product Images Small. The model learns an embedding space where same-class images stay close together while different-class images are pushed apart. The learned embedding is then evaluated through one-shot recognition rather than conventional closed-set classification.

## Experimental Setup

- Dataset: Fashion Product Images Small
- Epochs run: 5
- Batch size: 64
- Train pairs per epoch: 12000
- Validation pairs: 2000
- Embedding dimension: 32
- Contrastive margin: 1.0
- One-shot setting: 5-way, 200 trials

## Quantitative Results

- Best validation loss: 0.1242
- Final train pair accuracy: 0.8103
- Final validation pair accuracy: 0.8315
- One-shot accuracy: 0.6900

## Learning Dynamics

The training curve is saved in `report_assets/loss_curve_fashion_products_small.png`. A healthy run should show decreasing training loss together with stable or improving validation loss. If validation loss stops improving quickly, that usually indicates either a small model capacity ceiling or that the sampled negative pairs are too easy.

## Embedding Space Analysis

The t-SNE projection saved in `report_assets/tsne_plot_fashion_products_small.png` provides a low-dimensional view of the learned embedding space. Effective metric learning should create visible clusters for individual classes, while visually similar classes remain closer to each other. This behavior is expected because t-SNE preserves local neighborhood structure more than global geometry.

## One-Shot Failure Analysis

The model made 62 errors out of 200 one-shot trials.

Most common confusion patterns:
- Tops -> Shirts: 7 cases
- Jeans -> Trousers: 7 cases
- Shorts -> Jeans: 7 cases
- Tshirts -> Shirts: 7 cases
- Tops -> Tshirts: 5 cases

Hardest failure cases are saved in `outputs/results/one_shot_failure_analysis_fashion_products_small.csv`.

## Discussion

Few-shot learning is useful when annotated data is limited, expensive, or constantly changing. Instead of learning only a fixed classifier boundary, the Siamese network learns a reusable similarity function. That is especially useful for verification, retrieval, anomaly matching, and low-resource recognition tasks.

For this project, Fashion Product Images Small offers a fast and controlled benchmark. MNIST-style datasets let the model converge quickly on CPU, which makes them practical for university coursework and rapid experimentation. The main limitation is that these datasets are relatively simple compared with real-world image variation, so they can overestimate how well a method will scale without harder negative mining or stronger feature extractors.

## Reproducibility

Run the full experiment with:

```bash
python main.py --dataset fashion_products_small
```

Launch the frontend with:

```bash
streamlit run app.py
```
