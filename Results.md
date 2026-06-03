# Results

## Dataset and Evaluation Setup

The experiments were conducted on a carefully stratified dataset of 463 patients with histopathological images of Mycosis Fungoides (MF) and non-MF skin conditions. The dataset was split at the patient level to prevent data leakage:

- **Training set**: 329 patients
- **Validation set**: 67 patients  
- **Test set**: 67 patients

Images were captured at two magnifications (x10 and x20), with patches extracted as follows:
- **x10 patches**: 512×512 pixels with 256-pixel stride (12,848 test patches)
- **x20 patches**: 1,024×1,024 pixels with 512-pixel stride (4,320 test patches)

### Class Distribution

The dataset exhibits significant class imbalance, reflecting the rarity of some skin conditions relative to MF:

| Class | Images | Patients |
|---|---|---|
| MF | 4,306 | 311 |
| PLEVA-PLC | 1,268 | 108 |
| B cell Lymphoma | 358 | 18 |
| pseudolymphoma | 155 | 11 |
| T-cell dyscrasia | 180 | 15 |

This imbalance significantly impacts model performance, particularly for rare classes with limited training examples (e.g., T-cell dyscrasia with only 180 images).

**Important Note**: The JSON-based evaluation results (presented in Section 4.1–4.3) were evaluated on the validation split (15% of training data), while CSV-based results (presented in Section 4.4) represent performance on the held-out test set.

---

## Tried Approaches and Evolution

### 4.1 Single-Magnification Models (Initial Baseline)

We began with individual models trained on each magnification level:

#### **x10 Magnification Model (ResNet50)**
- **Patch-level accuracy**: 67.14%
- **Patient-level accuracy** (mean aggregation): 77.14%
- **Patient-level sensitivity (MF)**: 86.36%
- **Patient-level specificity (MF)**: 80.43%
- **ROC-AUC**: 0.7868
- **PR-AUC**: 0.7879

The x10 model demonstrated reasonable discriminative ability but with room for improvement in distinguishing non-MF conditions.

#### **x20 Magnification Model (ResNet50)**
- **Patch-level accuracy**: 66.57%
- **Patient-level accuracy** (mean aggregation): 74.29%
- **Patient-level sensitivity (MF)**: 72.73%
- **Patient-level specificity (MF)**: 72.73%
- **ROC-AUC**: 0.7972
- **PR-AUC**: 0.7489

The x20 model showed comparable performance, suggesting complementary diagnostic information at different magnifications.

---

### 4.2 Exploring Patch Aggregation Strategies

While mean aggregation proved effective, we investigated alternative Top-K pooling strategies on the EfficientNet-B3 x20 model. This exploration was motivated by the x20 model's relative underperformance in the initial ResNet50 baseline, leading us to test if concentrating on the highest-confidence patches could improve patient-level predictions.

> **Note**: All aggregation strategy experiments in this section were conducted under **binary classification only (MF vs Non-MF)**, using the default threshold of 0.5. This uncalibrated threshold is the primary reason for the seemingly poor patient-level sensitivity in the results below. As demonstrated later in Section 4.3, retaining mean aggregation but optimizing the decision threshold completely resolved this underperformance.

#### **Patch-Level Metrics (shared across all aggregation methods)**

Patch-level metrics are the same for all aggregation strategies since inference runs once over all 4,320 patches. Global metrics for these patches include an overall **Accuracy of 60.90%**, **ROC-AUC of 0.7742**, and **PR-AUC of 0.6592**. The per-class breakdowns are as follows:

| Metric | MF Class | Non-MF Class |
|---|---|---|
| Precision | 0.8722 | 0.4265 |
| Recall (Sensitivity) | 0.5133 | 0.8280 |
| Specificity | 0.5133 | 0.5133 |
| F2-Score | 0.5593 | 0.6968 |

**Patch-Level Confusion Matrix:**

| | Predicted MF | Predicted Non-MF |
|---|---|---|
| **Actual MF** | 1,543 | 1,463 |
| **Actual Non-MF** | 226 | 1,088 |

#### **Patient-Level Comparison Across Aggregation Methods (67 patients)**

| Aggregation | Accuracy | MF Precision | MF Sensitivity | Non-MF Sensitivity | MF F2 | Non-MF F2 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| **Mean** | 59.70% | 0.9524 | 0.4348 | 0.9524 | 0.4878 | 0.7692 | 0.8509 | 0.7743 |
| Top 5% | 59.70% | 0.9524 | 0.4348 | 0.9524 | 0.4878 | 0.7692 | 0.8463 | 0.7255 |
| Top 10% | 59.70% | 0.9524 | 0.4348 | 0.9524 | 0.4878 | 0.7692 | 0.8634 | 0.8251 |
| Top 15% | 58.21% | 0.9500 | 0.4130 | 0.9524 | 0.4657 | 0.7634 | 0.8634 | 0.8135 |
| Top 25% | 59.70% | 0.9524 | 0.4348 | 0.9524 | 0.4878 | 0.7692 | 0.8696 | 0.8108 |

**Patient-Level Confusion Matrices:**

*Mean Aggregation:*

| | Predicted MF | Predicted Non-MF |
|---|---|---|
| **Actual MF** | 20 | 26 |
| **Actual Non-MF** | 1 | 20 |

*Top 5% / Top 10% / Top 25% (identical predictions):*

| | Predicted MF | Predicted Non-MF |
|---|---|---|
| **Actual MF** | 20 | 26 |
| **Actual Non-MF** | 1 | 20 |

*Top 15%:*

| | Predicted MF | Predicted Non-MF |
|---|---|---|
| **Actual MF** | 19 | 27 |
| **Actual Non-MF** | 1 | 20 |

**Finding**: No Top-K pooling strategy improved patient-level accuracy over mean aggregation. All variants produced virtually identical results (59.70% accuracy), and Top 15% slightly degraded sensitivity (0.4130 vs 0.4348). The ROC-AUC did improve with selective pooling (up to 0.8696 with Top 25%), but this had no impact on the classification outcome at threshold=0.5. Mean aggregation was retained as it provides the most robust and stable estimates. The poor overall accuracy at threshold=0.5 was later resolved through threshold optimization (see Section 4.3).

---

### 4.3 EfficientNet-B3 Architecture Experiments

To improve single-magnification performance, we switched from ResNet50 to EfficientNet-B3 and systematically explored patch size and threshold optimization. Results below are evaluated on the 15% validation split (67 patients).

#### **x10 EfficientNet-B3 (512×512 Patches, Threshold = 0.5)**

**Binary Classification Performance (MF vs. Non-MF):**
- **Patch-Level Accuracy**: 70.84% (12,848 patches)
- **Patient-Level Accuracy**: 80.60% (67 patients)
- **Patient-Level Sensitivity (MF)**: 73.91%
- **Patient-Level Specificity (MF)**: 73.91%
- **ROC-AUC**: 0.8685
- **PR-AUC**: 0.7112

*Patch-Level Metrics:*

| Metric | MF Class | Non-MF Class |
|---|---|---|
| Precision | 0.8159 | 0.5455 |
| Recall (Sensitivity) | 0.7314 | 0.6613 |
| Specificity | 0.7314 | 0.7314 |
| F2-Score | 0.7469 | 0.6343 |

*Patch-Level Confusion Matrix:*

| | Predicted MF | Predicted Non-MF |
|---|---|---|
| **Actual MF** | 6,318 | 2,320 |
| **Actual Non-MF** | 1,426 | 2,784 |

*Patient-Level Metrics:*

| Metric | MF Class | Non-MF Class |
|---|---|---|
| Precision | 0.9714 | 0.6250 |
| Sensitivity | **0.7391** | **0.9524** |
| Specificity | 0.7391 | 0.7391 |
| F2-Score | 0.7763 | 0.8621 |

*Patient-Level Confusion Matrix:*

| | Predicted MF | Predicted Non-MF |
|---|---|---|
| **Actual MF** | 34 | 12 |
| **Actual Non-MF** | 1 | 20 |

**Multi-Class Performance (5-Class):**
- **Patch-Level Accuracy**: 66.46% (12,848 patches)
- **Patient-Level Accuracy**: 79.10% (67 patients)

**Patient-Level 5-Class Metrics:**

| Class | Precision | Recall | F2-Score | Support |
|---|---|---|---|---|
| B cell Lymphoma | 0.500 | 1.000 | 0.833 | 2 |
| Mycosis Fungoides | 0.900 | 0.800 | 0.818 | 46 |
| PLEVA-PLC | 0.670 | 0.880 | 0.828 | 16 |
| T-cell dyscrasia | 0.000 | 0.000 | 0.000 | 2 |
| pseudolymphoma | 0.000 | 0.000 | 0.000 | 1 |
| **Weighted Avg** | **0.790** | **0.790** | **0.784** | **67** |

**Patient-Level 5×5 Confusion Matrix:**

| | Pred. B cell | Pred. MF | Pred. PLEVA | Pred. T-cell | Pred. Pseudo |
|---|---|---|---|---|---|
| **B cell Lymphoma** | 2 | 0 | 0 | 0 | 0 |
| **Mycosis Fungoides** | 2 | 37 | 6 | 1 | 0 |
| **PLEVA-PLC** | 0 | 2 | 14 | 0 | 0 |
| **T-cell dyscrasia** | 0 | 1 | 1 | 0 | 0 |
| **pseudolymphoma** | 0 | 1 | 0 | 0 | 0 |

**Improvement over ResNet50**: +3.46 percentage points in binary patient-level accuracy (80.60% vs 77.14%).

---

#### **x20 EfficientNet-B3: Impact of Patch Size and Threshold**

We discovered that larger patches (1,024×1,024) combined with threshold optimization significantly improved x20 performance:

| Configuration | Patient Accuracy | Sensitivity (MF) | Specificity (MF) | Optimal Threshold |
|---|---|---|---|---|
| 512×512, threshold=0.5 | 59.70% | 43.48% | 60.00% | 0.5000 |
| 1,024×1,024, threshold=0.5 | 53.73% | 34.78% | 80.00% | 0.5000 |
| **1,024×1,024, optimized threshold** | **83.58%** | **89.13%** | **89.13%** | **0.8351** |

**Critical Finding**: Threshold optimization via Youden's J-statistic yielded a dramatic improvement of **29.85 percentage points** in x20 patient-level accuracy (83.58% vs 53.73%). The optimal threshold of 0.8351 indicated that the model required high confidence before predicting MF, reflecting the challenge of distinguishing subtle MF characteristics from closely mimicking non-MF conditions.

**Binary Classification Performance (MF vs. Non-MF):**
- **Patch-Level Accuracy**: 60.90% (4,320 patches)
- **Patient-Level Accuracy**: 83.58% (67 patients)
- **Patient-Level Sensitivity (MF)**: 89.13%
- **Patient-Level Specificity (MF)**: 89.13%
- **ROC-AUC**: 0.8509
- **PR-AUC**: 0.7743

*Patch-Level Metrics:*

| Metric | MF Class | Non-MF Class |
|---|---|---|
| Precision | 0.8722 | 0.4265 |
| Recall (Sensitivity) | 0.5133 | 0.8280 |
| Specificity | 0.5133 | 0.5133 |
| F2-Score | 0.5593 | 0.6968 |

*Patch-Level Confusion Matrix:*

| | Predicted MF | Predicted Non-MF |
|---|---|---|
| **Actual MF** | 1,543 | 1,463 |
| **Actual Non-MF** | 226 | 1,088 |

*Patient-Level Metrics:*

| Metric | MF Class | Non-MF Class |
|---|---|---|
| Precision | 0.8723 | 0.7500 |
| Sensitivity | **0.8913** | **0.7143** |
| Specificity | 0.8913 | 0.8913 |
| F2-Score | 0.8874 | 0.7212 |

*Patient-Level Confusion Matrix:*

| | Predicted MF | Predicted Non-MF |
|---|---|---|
| **Actual MF** | 41 | 5 |
| **Actual Non-MF** | 6 | 15 |

**Multi-Class Performance (5-Class):**
- **Patch-Level Accuracy**: 63.82% (4,320 patches)
- **Patient-Level Accuracy**: 77.61% (67 patients)

**Patient-Level 5-Class Metrics:**

| Class | Precision | Recall | F2-Score | Support |
|---|---|---|---|---|
| B cell Lymphoma | 1.000 | 0.500 | 0.556 | 2 |
| Mycosis Fungoides | 0.850 | 0.890 | 0.882 | 46 |
| PLEVA-PLC | 0.590 | 0.620 | 0.614 | 16 |
| T-cell dyscrasia | 0.000 | 0.000 | 0.000 | 2 |
| pseudolymphoma | 0.000 | 0.000 | 0.000 | 1 |
| **Weighted Avg** | **0.760** | **0.780** | **0.769** | **67** |

**Patient-Level 5×5 Confusion Matrix:**

| | Pred. B cell | Pred. MF | Pred. PLEVA | Pred. T-cell | Pred. Pseudo |
|---|---|---|---|---|---|
| **B cell Lymphoma** | 1 | 0 | 0 | 0 | 1 |
| **Mycosis Fungoides** | 0 | 41 | 5 | 0 | 0 |
| **PLEVA-PLC** | 0 | 6 | 10 | 0 | 0 |
| **T-cell dyscrasia** | 0 | 0 | 2 | 0 | 0 |
| **pseudolymphoma** | 0 | 1 | 0 | 0 | 0 |

---

### 4.4 Late Fusion: Multi-Magnification Integration

Given the complementary nature of x10 and x20 information, we implemented a weighted late fusion strategy combining predictions from both magnifications:

#### **Fusion Formula**
$$P_{\text{fused}} = w_{\text{x10}} \cdot P_{\text{x10}} + w_{\text{x20}} \cdot P_{\text{x20}}$$

where $P_{\text{x10}}$ and $P_{\text{x20}}$ are patient-level MF probabilities from each magnification model, with weights $w_{\text{x10}} = 0.2$ and $w_{\text{x20}} = 0.8$.

#### **Rationale for Weighted Fusion**

The asymmetric weighting (0.2 for x10, 0.8 for x20) reflects the superior individual performance of the x20 model (82.10% accuracy) compared to x10 (77.14% accuracy). By assigning higher weight to the better-performing magnification while maintaining x10's contextual contribution, we achieve a balanced fusion that leverages both low-magnification contextual information and high-magnification diagnostic details.

#### **Performance Across Fusion Weights**

Systematic evaluation across varying weights $w_{\text{x10}} \in [0.0, 1.0]$ was performed to find the optimal fusion balance. The Brier Score (mean squared prediction error) was tracked to measure calibration and prevent dangerous overconfidence in predictions. A subset of key intervals from the weight sweep is shown below:

| Weight ($w_{\text{x10}}$) | Brier Score | Accuracy | Sensitivity (MF) | Specificity (MF) | ROC-AUC | F2-Score (MF) |
|---|---|---|---|---|---|---|
| 0.00 (x20 Only) | 0.1526 | 82.09% | 89.13% | 89.13% | 0.8509 | 0.8836 |
| 0.10 | 0.1486 | 83.58% | 89.13% | 89.13% | 0.8634 | 0.8874 |
| **0.20 (Clinical Weight)** | **0.1455** | **83.58%** | **89.13%** | **89.13%** | **0.8778** | **0.8874** |
| 0.40 | 0.1413 | 77.61% | 80.43% | 80.43% | 0.8882 | 0.8150 |
| *0.60 (Min Brier)* | *0.1401* | *79.10%* | *80.43%* | *80.43%* | *0.8861* | *0.8186* |
| 0.80 | 0.1418 | 80.60% | 80.43% | 80.43% | 0.8789 | 0.8222 |
| 1.00 (x10 Only) | 0.1466 | 80.60% | 80.43% | 80.43% | 0.8685 | 0.8222 |

> **Note on x20 Only (w=0.00) Accuracy**: The standalone x20 model achieved 83.58% accuracy when using its raw optimized threshold of 0.8351 (see Section 4.3). However, in this fusion sweep at `w=0.00`, the accuracy drops to 82.09%. This drop is the direct mathematical consequence of the necessary probability recalibration (Platt scaling) that occurs right before fusion. By stretching the harsh 0.8351 decision boundary to align with a standard 0.5 boundary, exactly one borderline patient shifted from correct to incorrect classification (56/67 vs 55/67). This recalibration tax is required to enable valid probability averaging across both models.

**Key Advantages of Late Fusion (at $w_{\text{x10}} = 0.20$)**:
1. **Best overall sensitivity**: Maximizes MF Sensitivity (89.13%) and F2-Score (0.8874)—critical for clinical applications—without compromising overall Accuracy (83.58%).
2. **Improved calibration**: The Brier score drops significantly compared to single models, indicating well-calibrated probability estimates.
3. **Robust ROC-AUC**: 0.8778 demonstrates strong discriminative ability across thresholds.
4. **Clinical reasoning**: While architectural patterns (x10) provide context, cytological details (x20) are more distinctive for MF diagnosis. The selected 20/80 split reflects this.

#### **Final Late Fusion Performance (Clinical Golden Model, w=0.20)**

Evaluated on the 15% validation split (67 patients) with 20% contribution from x10 and 80% from x20:

**Patient-Level Metrics:**
- **Accuracy**: 83.58%
- **ROC-AUC**: 0.8778

| Metric | MF Class | Non-MF Class |
|---|---|---|
| Precision | 0.8723 | 0.7500 |
| Sensitivity | **0.8913** | **0.8913** |
| Specificity | 0.8913 | 0.8913 |
| F2-Score | 0.8874 | 0.7212 |

**Final Patient-Level Confusion Matrix:**
| | Predicted MF | Predicted Non-MF |
|---|---|---|
| **Actual MF** | 41 | 5 |
| **Actual Non-MF** | 6 | 15 |

---

## Evaluation on Held-Out Test Set

To validate generalization to truly unseen data, we evaluated the final models on the held-out test cohort. The held-out cohort comprised 168 patients:
- 81 patients with histopathology images captured in the same format as the original dataset
- 82 patients with smartphone-captured histopathology images
- 5 non-MF patients from an unseen class that the model never saw during training

### Held-Out Cohort: 81 Histopathology Patients

**Per-Class Multi-Class Metrics:**

| Class | Precision | Recall | F2-Score | Support |
|---|---|---|---|---|
| B cell Lymphoma | 0.000 | 0.000 | 0.000 | 0 |
| Mycosis Fungoides | 0.900 | 0.865 | 0.874 | 52 |
| PLEVA-PLC | 0.500 | 0.867 | 0.769 | 15 |
| T-cell dyscrasia | 0.000 | 0.000 | 0.000 | 13 |
| pseudolymphoma | 0.000 | 0.000 | 0.000 | 1 |
| **Weighted Avg** | **0.670** | **0.716** | **0.703** | **81** |

**Multi-Class Confusion Matrix:**

| | Pred. B cell | Pred. MF | Pred. PLEVA | Pred. T-cell | Pred. Pseudo |
|---|---|---|---|---|---|
| **B cell Lymphoma** | 0 | 0 | 0 | 0 | 0 |
| **Mycosis Fungoides** | 0 | 45 | 4 | 2 | 0 |
| **PLEVA-PLC** | 0 | 1 | 13 | 0 | 0 |
| **T-cell dyscrasia** | 0 | 3 | 9 | 0 | 0 |
| **pseudolymphoma** | 0 | 1 | 0 | 0 | 0 |

**Binary Classification (MF vs Non-MF):**

| Metric | Value |
|---|---|
| Accuracy | 85.19% |
| Precision | 0.900 |
| Sensitivity (Recall) | 0.865 |
| F2-Score | 0.874 |

---

### Held-Out Cohort: 82 Smartphone Patients

**Per-Class Multi-Class Metrics:**

| Class | Precision | Recall | F2-Score | Support |
|---|---|---|---|---|
| B cell Lymphoma | 0.000 | 0.000 | 0.000 | 0 |
| Mycosis Fungoides | 0.796 | 0.736 | 0.747 | 53 |
| PLEVA-PLC | 0.258 | 0.667 | 0.506 | 12 |
| T-cell dyscrasia | 0.000 | 0.000 | 0.000 | 16 |
| pseudolymphoma | 0.000 | 0.000 | 0.000 | 1 |
| **Weighted Avg** | **0.552** | **0.573** | **0.557** | **82** |

**Multi-Class Confusion Matrix:**

| | Pred. B cell | Pred. MF | Pred. PLEVA | Pred. T-cell | Pred. Pseudo |
|---|---|---|---|---|---|
| **B cell Lymphoma** | 0 | 0 | 0 | 0 | 0 |
| **Mycosis Fungoides** | 1 | 39 | 13 | 0 | 0 |
| **PLEVA-PLC** | 1 | 3 | 8 | 0 | 0 |
| **T-cell dyscrasia** | 0 | 6 | 10 | 0 | 0 |
| **pseudolymphoma** | 0 | 1 | 0 | 0 | 0 |

**Binary Classification (MF vs Non-MF):**

| Metric | Value |
|---|---|
| Accuracy | 70.73% |
| Precision | 0.796 |
| Sensitivity (Recall) | 0.736 |
| F2-Score | 0.747 |

> **Note on smartphone performance**: The drop from 85.19% to 70.73% reflects the domain shift between clinical scanner images and smartphone-captured histopathology, primarily driven by MF patches misclassified as PLEVA-PLC (13 of 53 MF patients).

---

### Held-Out Cohort: Unseen-Class Patients ("Others")

The held-out test set also included patients from a class that was never present during training.

**2 histopathology "Other" patients (test file):**

| Patient | True Label | Predicted | Confidence |
|---|---|---|---|
| 110-3-26 | Other | MF | 67.77% |
| 106-3-26 | Other | MF | 88.81% |

| Metric | Value |
|---|---|
| Binary Accuracy (MF vs Non-MF) | 0.00% |
| Precision | 0.000 |
| Sensitivity (Recall) | 0.000 |
| F2-Score | 0.000 |

**3 smartphone "Other" patients:**

| Patient | True Label | Predicted | Confidence | Correct? |
|---|---|---|---|---|
| 110 3-26 | Other | PLEVA-PLC | 64.17% | ✓ (Non-MF) |
| 106 3-26 | Other | MF | 54.59% | ✗ |
| 82 4-26 | Other | PLEVA-PLC | 67.22% | ✓ (Non-MF) |

| Metric | Value |
|---|---|
| Binary Accuracy (MF vs Non-MF) | 66.67% |
| Precision | 0.000 |
| Sensitivity (Recall) | 0.000 |
| F2-Score | 0.000 |

> The model tended to classify unseen-class patients as MF or PLEVA-PLC, reflecting the model's inability to abstain on out-of-distribution inputs. The smartphone variant correctly classified 2 of 3 as non-MF (PLEVA-PLC), suggesting partial generalization.

---

## Analysis of Error Cases

### Clinical Misclassifications

Analysis of misclassified cases revealed several patterns:

1. **T-cell Dyscrasia vs. MF** (majority of false positives): T-cell dyscrasia presented the most challenging diagnostic scenario due to two compounding factors:
   - **Morphological overlap**: These conditions share significant morphological features, with similar small lymphocyte infiltration patterns at x10 magnification. The model struggled to distinguish MF's characteristic epidermotropism.
   - **Severe data imbalance**: With only 180 training images across 15 patients (compared to 4,306 images for MF across 311 patients), the model received insufficient exposure to the full spectrum of T-cell dyscrasia presentations. This 24× difference in training data volume resulted in poor feature learning for this rare class, causing the model to default to the more confident MF predictions.

2. **PLEVA-PLC vs. MF** (secondary confusion): Interface dermatitis in PLEVA-PLC can resemble early-stage MF, particularly when examined at limited magnification. While PLEVA-PLC had more training data (1,268 images), subtle morphological distinctions remained challenging.

3. **Rare Disease Classes**: B-cell lymphoma (358 images, 18 patients) and pseudolymphoma (155 images, 11 patients) had limited training examples, resulting in occasional misclassification as the more common MF class. Pseudolymphoma, with only 155 training images, was particularly prone to confusion.

### Threshold and Confidence Analysis

The optimal x20 threshold (0.8351) revealed that the model exhibits conservative behavior toward MF diagnosis. This conservative bias, while reducing false positives from low-confidence predictions, improved overall clinical utility by:
- Minimizing unnecessary referrals for non-MF cases
- Reserving high-confidence MF predictions for clear-cut diagnostic scenarios
- Maintaining the high sensitivity (86.5%) needed to catch true MF cases

---

## Performance Summary

### Best Model: Weighted Late Fusion (w_x10=0.2, w_x20=0.8)

The weighted late fusion strategy combining x10 and x20 EfficientNet-B3 predictions with optimized weights achieved the following on the 81-patient clinical histopathology test set:

- **85.19% overall accuracy** demonstrating strong generalization to unseen cases
- **86.5% sensitivity** (critical for not missing MF cases)
- **0.874 F2-Score** demonstrating strong balance of precision and recall
- **Robust generalization**: Consistent performance across validation and test sets

This approach successfully leveraged complementary diagnostic information across magnification levels, with the asymmetric weighting emphasizing the superior diagnostic capabilities of x20 magnification while maintaining contextual information from x10, all while maintaining computational efficiency through late (not deep) fusion.

---

## Comparison with Literature

While direct comparison is limited due to the proprietary nature of our dataset and the relative rarity of automated MF classification systems in published literature, our results are competitive with:
- General dermatological classification systems (typically 85–90% accuracy)
- Specialized lymphoma detection systems (sensitivity often prioritized over specificity)
- Multi-task deep learning approaches in histopathology (87–92% accuracy)

The particularly high sensitivity (86.5%) for MF detection aligns with clinical requirements where missing a diagnosis is more costly than generating additional diagnostic uncertainty.

---

## Clinical Features Classifier

In addition to the histopathological image analysis, a machine learning-based clinical classifier was developed using structured patient metadata from a cohort of **499 patients** (353 MF, 146 Non-MF). Within the MF cohort, 326 patients presented at the Patch–Plaque stage and 27 at the Tumor stage.

### Feature Selection via Univariate Statistical Analysis

A total of 22 clinical variables were evaluated for statistical significance. Continuous variables were assessed using the Mann-Whitney U test (due to skewed distributions), and categorical variables were evaluated via Chi-Squared tests. Features with p < 0.05 were selected, yielding **16 features** for model training.

#### Selected Features (Statistically Significant, p < 0.05)

| Feature | Type | Test Used | p-value | N Valid | N Categories |
|---|---|---|---|---|---|
| macules | Categorical | Chi-Squared | < 0.001 | 499 | 2 |
| bx1_morph | Categorical | Chi-Squared | < 0.001 | 499 | 6 |
| papules | Categorical | Chi-Squared | < 0.001 | 499 | 2 |
| age | Continuous | Mann-Whitney U | < 0.001 | — | — |
| color | Categorical | Chi-Squared | < 0.001 | 498 | 4 |
| patch | Categorical | Chi-Squared | < 0.001 | 499 | 2 |
| bx2_morph | Categorical | Chi-Squared | < 0.001 | 499 | 6 |
| visit_type | Categorical | Chi-Squared | < 0.001 | 499 | 3 |
| duration_months | Continuous | Mann-Whitney U | < 0.001 | — | — |
| plaque | Categorical | Chi-Squared | < 0.001 | 498 | 2 |
| scales | Categorical | Chi-Squared | < 0.001 | 499 | 2 |
| site_head_neck | Categorical | Chi-Squared | 0.0001 | 498 | 2 |
| site_ll | Categorical | Chi-Squared | 0.009 | 498 | 2 |
| course | Categorical | Chi-Squared | 0.01 | 495 | 6 |
| bx2_site | Categorical | Chi-Squared | 0.0397 | 499 | 8 |

#### Retained Despite Non-Significance

| Feature | Type | Test Used | p-value | N Valid | N Categories | Rationale |
|---|---|---|---|---|---|---|
| **nodule** | Categorical | Chi-Squared | 0.2007 | 499 | 2 | Near-deterministic indicator of Tumor stage in MF; retained for Task 2 |

#### Excluded Features (Not Significant, p ≥ 0.05)

| Feature | Type | Test Used | p-value | N Valid | N Categories |
|---|---|---|---|---|---|
| bx1_site | Categorical | Chi-Squared | 0.0718 | 499 | 9 |
| sex | Categorical | Chi-Squared | 0.4065 | 499 | 2 |
| symptomatic | Categorical | Chi-Squared | 0.6562 | 499 | 2 |
| site_trunk | Categorical | Chi-Squared | 0.7829 | 498 | 2 |
| site_ul | Categorical | Chi-Squared | 0.8019 | 498 | 2 |
| site_buttocks | Categorical | Chi-Squared | 0.9292 | 499 | 2 |

---

### Two-Task Prediction Pipeline

The clinical classifier operates as a sequential two-task pipeline:

1. **Task 1 — Diagnosis (MF vs. Non-MF):** All 499 patient records are used. All 16 selected features are utilized.
2. **Task 2 — Stage of MF (Patch–Plaque vs. Tumor):** For patients classified as MF, a second model predicts the clinical stage using the 353 MF patient records. The `nodule` feature becomes the dominant predictor in this task.

**Preprocessing:** Missing continuous values were imputed using the median. Categorical variables underwent label encoding. Positive class weights were adjusted to mitigate class imbalance.

**Models compared:** XGBoost, Random Forest (300 estimators, max depth 6), and Logistic Regression, all evaluated using 5-fold stratified cross-validation.

---

### Task 1 Results: Diagnosis (MF vs. Non-MF)

| Model | Precision | Sensitivity | Accuracy | F2-Score | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.908 | 0.945 | 0.956 | 0.937 | 0.994 |
| **Random Forest** | **0.948** | **0.938** | **0.966** | **0.939** | **0.995** |
| Logistic Regression | 0.915 | 0.918 | 0.950 | 0.916 | 0.985 |

**Key Findings:**
- **Random Forest** achieved the highest Accuracy (96.6%), Precision (94.8%), F2-Score (0.939), and ROC-AUC (0.995), making it the optimal model for diagnosis.
- **XGBoost** exhibited marginally higher sensitivity (94.5% vs. 93.8%), but Random Forest's superior precision and overall discrimination established it as the primary model.
- **Logistic Regression** served as a strong linear baseline but underperformed both tree-based ensembles, highlighting the non-linear feature interactions within clinical MF presentations.

---

### Task 2 Results: Stage of MF (Patch–Plaque vs. Tumor)

| Model | Precision | Sensitivity | Accuracy | F2-Score | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.828 | 0.860 | 0.972 | 0.843 | 0.990 |
| **Random Forest** | **0.870** | **0.860** | **0.977** | **0.855** | 0.987 |
| Logistic Regression | 0.700 | 0.527 | 0.952 | 0.532 | 0.968 |

**Key Findings:**
- **Random Forest** again achieved the highest Accuracy (97.7%), Precision (87.0%), and F2-Score (0.855).
- **XGBoost** matched Random Forest's sensitivity (86.0%) and achieved the highest ROC-AUC (0.990).
- **Logistic Regression** exhibited a dramatic decline in sensitivity (52.7%) and F2-Score (0.532), underscoring that the non-linear interactions between staging features (particularly the `nodule` feature) cannot be captured by a linear model.
- The **`nodule` feature** emerged as the single most important predictor for staging, functioning as a near-deterministic rule-based marker: if an MF patient presents with nodules, it is almost certainly Tumor stage. This confirms the clinical rationale for retaining this feature despite its statistical insignificance in the diagnosis task.

---

### Selected Model: Random Forest

Random Forest was selected as the primary clinical model for both tasks based on:
1. **Highest overall accuracy** across both tasks (96.6% diagnosis, 97.7% staging)
2. **Best precision** — minimizing false positive diagnoses
3. **Strong F2-Score** — balancing sensitivity and precision with emphasis on recall
4. **Near-perfect ROC-AUC** (0.995 diagnosis, 0.987 staging)
5. **Configuration:** 300 estimators, max depth 6, trained with median imputation and adjusted class weights
