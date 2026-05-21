# Report Submission-Ready Sections

## Source Basis

All numerical values and interpretations below were derived from the saved project artifacts in this repository, primarily:

- `outputs/results/evaluation_metrics_*.json`
- `outputs/results/training_history_*.json`
- `outputs/results/failure_summary_*.json`
- `outputs/results/one_shot_failure_analysis_*.csv`
- `outputs/figures/*.png`
- `report_*.md`

Important constraint:

- The historical runs do not persist the learning rate in the output artifacts. The implementation default in `config.py` is `0.001`, but that value cannot be independently verified for each saved run unless the experiment is rerun with explicit hyperparameter serialization.

## 1. Experimental Metrics Summary

### Table 1. Recorded Experimental Metrics Across Saved Runs

| Dataset | Epochs Run | Best Epoch | Batch Size | Train Pairs | Val Pairs | Final Train Loss | Final Val Loss | Final Train Pair Acc. | Final Val Pair Acc. | One-Shot Acc. | Embedding Dim. | Margin | Effective N-Way | Trials | One-Shot Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNIST | 5 | 5 | 64 | 12000 | 2000 | 0.0447 | 0.0496 | 0.9697 | 0.9505 | 0.9700 | 32 | 1.0 | 5 | 200 | 6 |
| Fashion-MNIST | 1 | 1 | 64 | 1200 | 240 | 0.2419 | 0.2280 | 0.6267 | 0.6167 | 0.6500 | 32 | 1.0 | 5 | 20 | 7 |
| Fashion Product Images Small | 5 | 5 | 64 | 12000 | 2000 | 0.1328 | 0.1242 | 0.8103 | 0.8315 | 0.6900 | 32 | 1.0 | 5 | 200 | 62 |
| Amharic Handwritten Characters | 5 | 5 | 64 | 12000 | 2000 | 0.1643 | 0.1801 | 0.7623 | 0.7365 | 0.5250 | 32 | 1.0 | 5 | 200 | 95 |
| Ethiopian Cultural Clothing and Textile | 1 | 1 | 2 | 12 | 6 | 0.2320 | 0.0029 | 0.6667 | 1.0000 | 1.0000 | 32 | 1.0 | 2 | 6 | 0 |

### Learning-Rate Note

The saved experiment artifacts do not contain a serialized learning-rate field. The code default is `0.001`, but the final report should not present that value as a historically verified run-time metric unless the experiments are rerun with configuration logging enabled.

### Concise Academic Interpretation

The saved experiments show a clear performance hierarchy. MNIST is the strongest and most stable run, reaching a final validation pair accuracy of `0.9505` and a one-shot accuracy of `0.9700`. Fashion Product Images Small occupies a middle position, with `0.8315` validation pair accuracy and `0.6900` one-shot accuracy, indicating that the model learns meaningful similarity structure but struggles with fine-grained apparel categories. Amharic Handwritten Characters is the most difficult large-scale experiment, with `0.7365` validation pair accuracy and `0.5250` one-shot accuracy despite five epochs of optimization, suggesting that the current shallow encoder is insufficient for a highly multi-class handwritten character problem. Fashion-MNIST and Ethiopian textile results must be interpreted cautiously because both runs were performed with reduced protocols: Fashion-MNIST used only `20` one-shot trials and one training epoch, while the textile experiment used only `3` test images and an effective `2`-way evaluation.

## 2. Figure Explanations

### 2.1 Training Loss Curves

The loss-curve figures document the optimization behavior of the contrastive-loss objective over training. The MNIST curve is the cleanest example of successful convergence, with training loss decreasing from `0.1589` to `0.0447` and validation loss decreasing from `0.1021` to `0.0496` over five epochs. The close tracking between training and validation curves indicates that the model improved without obvious overfitting on this benchmark. The Fashion Product Images Small and Amharic runs show the same general downward trend, but with higher terminal losses (`0.1242` and `0.1801`, respectively), which is consistent with their more difficult one-shot recognition outcomes.

By contrast, the Fashion-MNIST and Ethiopian textile loss figures should be interpreted as limited snapshots rather than fully developed convergence trajectories, because each saved run contains only one epoch. In the Fashion-MNIST case, the single recorded epoch is insufficient to support strong claims about convergence or saturation. In the textile case, the extremely low validation loss (`0.0029`) is likely driven by the very small validation set rather than by robust generalization.

### 2.2 t-SNE Embedding Plots

The t-SNE figures provide qualitative evidence about the geometry of the learned embedding space. The MNIST plot is the strongest visual result: it shows compact, well-separated class clusters with only minor local overlap, which agrees with the `0.9700` one-shot accuracy and suggests that the learned metric is highly class-discriminative on digit data. The Fashion-MNIST t-SNE plot is substantially less organized. While some coarse grouping is visible, multiple clothing categories overlap, especially among upper-body garments, which is consistent with the modest `0.6500` one-shot accuracy.

The Fashion Product Images Small t-SNE plot shows partial structure rather than full separation. Several regions are clearly populated by dominant category clusters, but there is also extensive overlap among visually adjacent clothing types. This agrees with the failure summaries, which show repeated confusions such as `Tops -> Shirts`, `Tshirts -> Shirts`, and `Jeans -> Trousers`. The Amharic handwritten t-SNE plot is the weakest qualitative embedding result: class colors are widely intermixed across the manifold, indicating that the current embedding space does not form clean character-specific neighborhoods at this class scale. This visual observation is consistent with the `0.5250` one-shot accuracy and the diffuse failure distribution. The Ethiopian textile t-SNE plot is not analytically reliable because it was generated from only `3` test images across `2` classes; it confirms that the pipeline ran, but it does not provide a meaningful basis for claims about class geometry.

### 2.3 Positive and Negative Pair Figures

The sample-pair figures verify that the Siamese training data are being constructed correctly. In the MNIST and Amharic pair figures, positive pairs show recognizable within-class similarity, while negative pairs contrast visually distinct symbols or characters. In the Fashion Product Images Small figure, the positive pairs are already quite heterogeneous in pose, color, garment cut, and human presentation, which demonstrates that the system is solving a more difficult metric-learning problem than simple pixel-level resemblance. This is academically useful evidence because it shows that the model must learn class-level similarity under meaningful intra-class variation.

The textile pair figure provides a different kind of evidence: it shows that even positive pairs may differ substantially in crop, framing, and scene composition. That observation supports two conclusions. First, the local textile task is visually nontrivial in principle. Second, the current textile experiment is too small for its perfect score to be taken as a robust performance claim.

### 2.4 One-Shot Prediction Figures

The one-shot example grids are among the most informative figures in the repository because they expose actual support-set distances. In MNIST, correct supports are typically much closer to the query than incorrect alternatives. For example, one query digit `2` is matched to the correct support at distance `0.20`, while competing supports lie near `0.98-1.11`. Likewise, a query digit `6` is correctly matched at distance `0.13`, while distractors lie between `0.60` and `1.55`. This large margin between correct and incorrect supports explains the high one-shot accuracy.

In the harder datasets, the same figure reveals the embedding weaknesses directly. In Fashion-MNIST, a query from class `2` has an incorrect support from class `6` at distance `0.12`, which is even closer than the correct class-`2` support at distance `0.21`. In Fashion Product Images Small, a query from class `1` is closer to class `0` at distance `0.14` than to its own class at distance `0.35`, which indicates fine-grained semantic overlap in the embedding space. In Amharic handwriting, one query from class `70` is closer to class `3` at distance `0.32` than to its true class at distance `0.41`, showing that visually similar handwritten symbols can collapse together under the current encoder. The textile one-shot examples show very confident matches, with correct supports at distance `0.02` and incorrect supports at `0.91`, but the repeated reuse of the same few images confirms that the dataset is too small for this figure to demonstrate generality.

## 3. Failure Analysis

### 3.1 Dataset-Specific Failure Patterns

MNIST exhibits very few one-shot failures (`6/200`), and the confusion pairs are all visually plausible: `9 -> 8`, `8 -> 5`, `6 -> 5`, and `3 -> 0`. These errors are consistent with stroke-level similarity and indicate that the remaining mistakes occur near natural class boundaries rather than due to overall embedding collapse.

Fashion-MNIST produces `7` failures in only `20` trials, which is a nontrivial error rate given the small evaluation budget. The most notable confusions are `Trouser -> Dress`, `Pullover -> Shirt`, `Dress -> Shirt`, and `Ankle boot -> Sneaker`. These categories share similar global silhouettes in low-resolution grayscale images, so the observed errors suggest that the encoder is relying strongly on coarse shape cues and has not yet learned sufficiently discriminative fine-scale garment features.

Fashion Product Images Small shows the clearest evidence of fine-grained semantic confusion. The top failure modes are `Tops -> Shirts` (`7` cases), `Tshirts -> Shirts` (`7` cases), `Jeans -> Trousers` (`7` cases), and `Shorts -> Jeans` (`7` cases). These are not arbitrary mistakes; they are confusions among adjacent apparel categories with overlapping appearance distributions. The hardest recorded error is `Jeans -> Trousers` at a distance of only `0.0869`, which implies that some incorrect garment pairs are embedded almost as tightly as correct matches.

Amharic Handwritten Characters produces the largest absolute number of failures (`95/200`), but unlike Fashion Product Images Small, its errors are not dominated by a few recurring class pairs. The top listed confusion, `114 -> 184`, occurs only twice, while many other errors occur once. This diffuse pattern is important: it suggests a broad representational weakness rather than a small number of unstable boundaries. The model is not simply confusing one or two character families; it is struggling to maintain consistent separation across a large alphabet with substantial writer variability.

The Ethiopian textile experiment records zero one-shot failures, but that result is not statistically persuasive because the test split contains only `3` images in total and the effective evaluation setting is only `2`-way over `6` trials. The absence of failures should therefore be described as a successful pipeline demonstration rather than as definitive evidence of strong generalization.

### 3.2 Critical Interpretation of Failure Behavior

The failure artifacts collectively show that dataset complexity, class granularity, and sample scale strongly shape embedding quality. On MNIST, the class manifold is simple enough that the shallow Siamese encoder can learn stable class neighborhoods. On Fashion-MNIST, the shift from digits to garments introduces shape overlap that degrades discrimination. On Fashion Product Images Small, the problem becomes fine-grained and semantically structured, so the encoder must distinguish categories that are genuinely adjacent in visual space. On Amharic handwriting, the difficulty increases again because the task combines a large number of classes with handwriting variation, noise, stroke thickness differences, and subtle graphemic distinctions. The current system therefore succeeds most clearly on low-complexity, low-intra-class-variance data and degrades as the recognition problem becomes more realistic.

## 4. Experimental Discussion Paragraphs

### 4.1 Training Dynamics

The saved training histories indicate that the optimization procedure is stable on the longer five-epoch runs. MNIST improves monotonically from `0.1589` to `0.0447` in training loss and from `0.1021` to `0.0496` in validation loss, while validation pair accuracy rises to `0.9505`. Fashion Product Images Small also improves steadily, with validation accuracy increasing from `0.6625` to `0.8315`, which shows that the Siamese objective remains learnable on a more varied RGB dataset. Amharic handwriting follows the same downward loss trend, but the final validation loss remains comparatively high (`0.1801`), indicating that convergence alone does not guarantee a sufficiently discriminative embedding when the class space is large. Overall, the training curves support the claim that the system is learning meaningful similarity structure, but they also show that dataset difficulty determines how much representational value is gained from that optimization.

### 4.2 Embedding-Space Quality

The embedding-space quality is strongest on MNIST and progressively weaker on the more realistic datasets. The MNIST t-SNE plot displays compact and clearly separated clusters, which is exactly the qualitative structure expected from a well-trained metric-learning model. The Fashion Product Images Small plot shows partially separated but overlapping regions, indicating that the encoder captures broad category structure without fully resolving fine-grained garment distinctions. The Amharic t-SNE figure shows substantial intermixing across classes, which suggests that the current 32-dimensional embedding produced by the shallow CNN is not expressive enough for a 287-class handwritten recognition setting. Taken together, the figures and metrics show that the learned space is useful, but its discriminative power diminishes as the problem moves from coarse inter-class differences toward subtle intra-family variation.

### 4.3 One-Shot Evaluation Interpretation

The one-shot results demonstrate that pairwise discrimination and few-shot retrieval are related but not identical capabilities. MNIST reaches `0.9700` one-shot accuracy, which closely matches its strong pairwise validation performance and confirms that the embedding generalizes well to support-set matching. Fashion Product Images Small, however, reaches `0.8315` validation pair accuracy but only `0.6900` one-shot accuracy, indicating that a model can separate random positive and negative pairs reasonably well while still failing on fine-grained nearest-neighbor decisions among semantically adjacent classes. Amharic handwriting shows this gap even more clearly: the model achieves `0.7365` validation pair accuracy but only `0.5250` one-shot accuracy. This suggests that one-shot evaluation is the more meaningful performance measure for this project because it tests the actual downstream use case rather than only the pairwise training surrogate.

### 4.4 Pairwise Distance Interpretation

The saved one-shot example figures make the distance scale interpretable. In successful MNIST cases, correct supports often appear at distances around `0.13-0.42`, whereas incorrect supports frequently lie near or above `0.80`. This indicates a healthy margin in the learned space. In the more difficult datasets, incorrect classes sometimes appear at distances below `0.20`, which is a strong sign of embedding overlap. For example, the hardest Fashion Product Images Small error occurs at `0.0869`, and a hard Amharic error occurs at `0.0920`. These values show that the system can place semantically or structurally different items into nearly identical local neighborhoods. Consequently, the current fixed pairwise decision threshold of `0.5` is serviceable on simpler data but becomes less reliable as class granularity increases.

### 4.5 Strengths of the Implemented System

The system has several clear strengths. First, it is modular and academically transparent: data loading, pair generation, model definition, loss computation, training, evaluation, visualization, and report generation are separated cleanly. Second, it produces not only scalar metrics but also interpretive evidence, including t-SNE plots, pair examples, one-shot examples, and failure summaries. Third, it generalizes across multiple dataset types, including grayscale digits, grayscale clothing silhouettes, RGB product images, handwritten characters, and a local cultural-image dataset. Fourth, the system uses a true one-shot matching protocol rather than only reporting pairwise training success, which makes the evaluation more faithful to the stated few-shot learning objective.

### 4.6 Weaknesses and Trade-Offs

The main weaknesses are directly visible in the saved experiments. The encoder is shallow, so it performs well on simple benchmarks but loses discrimination on fine-grained and high-cardinality datasets. The current contrastive-learning setup uses random positive and negative pairs without hard-negative mining, which means the model may spend too much training time on easy separations rather than on ambiguous boundaries. The evaluation protocols are also uneven across datasets: Fashion-MNIST was run for only one epoch and `20` trials, while the textile experiment was evaluated on only `3` test images. These trade-offs make the system excellent for coursework demonstration and methodological clarity, but less suitable as a final high-confidence benchmark unless the harder datasets are rerun with stronger protocols and fully serialized hyperparameters.

## 5. Research Outlook

Several concrete research extensions follow naturally from the observed results. A first priority is hard negative mining. The current pair generator samples negative pairs uniformly, yet the failure analysis shows that the dominant errors arise from visually adjacent classes such as `Jeans` versus `Trousers`, `Tshirts` versus `Shirts`, and closely related handwritten characters. Hard-negative mining would force the encoder to allocate more capacity to these difficult boundaries rather than repeatedly solving already easy contrasts.

A second extension is to replace or complement contrastive loss with triplet loss or supervised contrastive loss. The present implementation optimizes pairwise separation, but the observed gap between pairwise accuracy and one-shot accuracy on the harder datasets suggests that a richer relative-ranking objective may better support nearest-neighbor retrieval. Triplet-style objectives could explicitly encourage the query to remain closer to a positive support than to a hard negative by a controlled margin.

A third direction is encoder scaling. The current model is intentionally lightweight and CPU-friendly, which is appropriate for demonstration purposes, but the Amharic and fine-grained apparel results indicate that the representational capacity is limited. A deeper convolutional backbone or a pretrained visual encoder would likely improve invariance to stroke variation, pose, texture, and background clutter.

Explainability is another worthwhile continuation. Since the system already produces embeddings and nearest-neighbor decisions, future work could add saliency visualization, class-activation mapping, or embedding-neighborhood inspection to clarify why particular errors occur. This would be especially valuable for cultural-image applications, where interpretability may matter alongside accuracy.

Dataset expansion is essential for the cultural-image component. The current Ethiopian textile experiment is best interpreted as a proof of pipeline compatibility rather than a mature benchmark, because it contains only `4` training images and `3` test images across two classes. A larger and better-balanced collection of Ethiopian cultural clothing and textile images would enable meaningful few-shot experiments with real intra-class and inter-class diversity.

Finally, future evaluation protocols should be strengthened. The saved results would be more publication-ready if every run stored the full hyperparameter configuration, used standardized trial counts, reported confidence intervals across repeated seeds, and evaluated on class-balanced support/query splits. These changes would improve both experimental rigor and reproducibility without changing the fundamental Siamese-learning formulation already implemented here.

## 6. Figure Captions

### Caption 1. Architecture Diagram

Figure X. Siamese network architecture used in the study. Two input images are processed by a shared-weight convolutional encoder, projected into a 32-dimensional L2-normalized embedding space, and compared through Euclidean distance under a contrastive-loss objective.

### Caption 2. Loss Curve

Figure X. Training and validation contrastive loss across epochs for the selected dataset. A consistent downward trend indicates successful metric-space optimization, while the relative gap between the two curves reflects the degree of generalization achieved by the Siamese encoder.

### Caption 3. t-SNE Plot

Figure X. t-SNE projection of test-set embeddings learned by the Siamese network. Well-separated clusters indicate strong class-wise organization in the learned metric space, whereas overlap between clusters suggests residual ambiguity among visually similar classes.

### Caption 4. Similarity Examples

Figure X. Representative positive and negative image pairs sampled for Siamese training. Positive pairs belong to the same class and illustrate intra-class variation, while negative pairs belong to different classes and provide the dissimilarity signal required by contrastive learning.

### Caption 5. One-Shot Prediction Examples

Figure X. Example one-shot N-way matching episodes. Each row shows a query image and one support image per candidate class, together with the learned embedding distance; the nearest support determines the predicted class.

## 7. Final Submission Checklist

### Verified from Repository Artifacts

- [x] Dataset-specific metrics JSON files exist for MNIST, Fashion-MNIST, Fashion Product Images Small, Amharic Handwritten Characters, and Ethiopian Cultural Clothing and Textile.
- [x] Dataset-specific training-history JSON files exist.
- [x] Loss-curve, t-SNE, sample-pair, and one-shot-example figures exist in `outputs/figures/`.
- [x] Dataset-specific report markdown files exist.
- [x] Saved model checkpoints exist for all five experiments.
- [x] Failure summaries exist for all completed experiments; the textile run correctly has no failure CSV because no failures were recorded.

### Requires Manual Insertion into the Final Report

- [ ] Insert the experimental summary table from this document into the main report.
- [ ] Insert at least one loss-curve figure and one t-SNE figure for the primary reported experiment.
- [ ] Insert representative sample-pair and one-shot-example figures.
- [ ] Insert the failure-analysis discussion and research-outlook section.
- [ ] Insert the professional figure captions.
- [ ] Add an explicit GitHub repository link. No GitHub URL was found in the repository markdown files.
- [ ] Add an architecture diagram. A caption has been provided, but no architecture diagram image was found in the repository.

### Requires Clarification or Regeneration for Maximum Rigor

- [ ] If the final report must state the learning rate as a historical run-time fact, rerun the experiments with saved hyperparameter logging, because the current output artifacts do not serialize it.
- [ ] If Fashion-MNIST is to be treated as a serious comparative experiment, rerun it with more than one epoch and more than `20` one-shot trials.
- [ ] If the Ethiopian textile result is to be emphasized, expand the dataset first; the current split contains only `4` training images and `3` test images.
- [ ] If citation formatting is being checked for final submission, verify IEEE style in the main report document itself, because the repository artifacts do not expose the external final manuscript layout.

## Recommended Paste-In Summary

If only one concise comparative paragraph is needed, the following can be inserted directly:

The saved experiments show that the implemented Siamese pipeline is highly effective on simple benchmark data but progressively challenged by fine-grained and high-cardinality recognition tasks. MNIST produced the strongest result, reaching `0.9505` validation pair accuracy and `0.9700` one-shot accuracy after five epochs, with a clearly separated embedding space in the corresponding t-SNE visualization. Fashion Product Images Small improved steadily during training and reached `0.8315` validation pair accuracy, but one-shot accuracy remained lower at `0.6900`, reflecting confusion among semantically adjacent garment categories such as Tops, Shirts, Tshirts, Jeans, and Trousers. The Amharic handwritten experiment converged numerically but remained difficult, achieving only `0.5250` one-shot accuracy and showing substantial class overlap in the t-SNE plot, which suggests that the current shallow encoder is underpowered for a 287-class handwritten recognition problem. The local Ethiopian textile run achieved perfect one-shot accuracy, but because the evaluation used only `3` test images and an effective `2`-way protocol, it should be interpreted as a proof of pipeline functionality rather than as a conclusive benchmark.
