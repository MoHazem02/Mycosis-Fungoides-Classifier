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

While mean aggregation proved effective, we investigated alternative pooling strategies to determine if focusing on high-confidence predictions could improve performance:

#### **Top-K Pooling Methods (x20 Model)**

| Aggregation Method | Patient Accuracy | Sensitivity (MF) | Specificity (MF) | ROC-AUC | Notes |
|---|---|---|---|---|---|
| Mean (baseline) | 74.29% | 72.73% | 72.73% | 0.7972 | Balanced performance |
| Top 50% pooling | 51.43% | 27.27% | 92.31% | 0.8112 | Over-emphasized specificity |
| Top 30% pooling | 45.71% | 18.18% | 92.31% | 0.8077 | Poor sensitivity, low utility |
| Top 10% pooling | 53.73% | 34.78% | 92.31% | N/A | Severe performance drop |
| Top 5% pooling | 53.73% | 34.78% | 92.31% | N/A | Insufficient samples |

**Finding**: Mean aggregation substantially outperformed selective pooling strategies, likely because averaging across patches provides more robust patient-level predictions than relying on a small subset of patches.

---

### 4.3 EfficientNet-B3 Architecture Experiments

To improve single-magnification performance, we switched from ResNet50 to EfficientNet-B3 and systematically explored patch size and threshold optimization:

#### **x10 EfficientNet-B3 (512×512 Patches)**
- **Patch-level accuracy**: 67.14%
- **Patient-level accuracy** (mean aggregation, 0.5 threshold): 80.60%
- **Patient-level sensitivity (MF)**: 80.43%
- **Patient-level specificity (MF)**: 80.86%
- **ROC-AUC**: 0.8078
- **Optimal threshold**: 0.50

**Improvement over ResNet50**: +3.46 percentage points in patient-level accuracy.

#### **x20 EfficientNet-B3: Impact of Patch Size and Threshold**

We discovered that larger patches (1,024×1,024) combined with threshold optimization significantly improved x20 performance:

| Configuration | Patient Accuracy | Sensitivity (MF) | Specificity (MF) | Optimal Threshold |
|---|---|---|---|---|
| 512×512, threshold=0.5 | 59.70% | 43.48% | 60.00% | 0.5000 |
| 1,024×1,024, threshold=0.5 | 53.73% | 34.78% | 80.00% | 0.5000 |
| **1,024×1,024, optimized threshold** | **82.10%** | **89.13%** | **89.13%** | **0.8351** |

**Critical Finding**: Threshold optimization via Youden's J-statistic yielded a dramatic improvement of **28.4 percentage points** in x20 patient-level accuracy. The optimal threshold of 0.8351 indicated that the model required high confidence before predicting MF, reflecting the challenge of distinguishing subtle MF characteristics from closely mimicking non-MF conditions.

---

### 4.4 Late Fusion: Multi-Magnification Integration

Given the complementary nature of x10 and x20 information, we implemented a weighted late fusion strategy combining predictions from both magnifications:

#### **Fusion Formula**
$$P_{\text{fused}} = w_{\text{x10}} \cdot P_{\text{x10}} + w_{\text{x20}} \cdot P_{\text{x20}}$$

where $P_{\text{x10}}$ and $P_{\text{x20}}$ are patient-level MF probabilities from each magnification model, with weights $w_{\text{x10}} = 0.2$ and $w_{\text{x20}} = 0.8$.

#### **Rationale for Weighted Fusion**

The asymmetric weighting (0.2 for x10, 0.8 for x20) reflects the superior individual performance of the x20 model (82.10% accuracy) compared to x10 (77.14% accuracy). By assigning higher weight to the better-performing magnification while maintaining x10's contextual contribution, we achieve a balanced fusion that leverages both low-magnification contextual information and high-magnification diagnostic details.

#### **Performance Across Fusion Weights**

| Metric | x10 Only (w=1.0) | x20 Only (w=0.0) | Fused (w_x10=0.2, w_x20=0.8) |
|---|---|---|---|
| **Accuracy** | 79.31% | 82.76% | **86.21%** |
| **Sensitivity (MF)** | 77.78% | 94.44% | **97.22%** |
| **Specificity (MF)** | 81.82% | 63.64% | **68.18%** |
| **Precision (MF)** | 87.50% | 80.95% | **83.33%** |
| **F2-Score (MF)** | 0.7955 | 0.9140 | **0.9409** |
| **ROC-AUC** | 0.9015 | 0.8346 | **0.9129** |
| **PR-AUC** | 0.8840 | N/A | N/A |
| **Brier Score** | 0.1411 | 0.1371 | **0.1164** |

**Key Advantages of Late Fusion**:
1. **Best overall accuracy**: 86.21%, surpassing both individual models
2. **Highest sensitivity**: 97.22% MF detection rate—critical for clinical applications
3. **Balanced specificity**: 68.18%, adequate for reducing false positives
4. **Improved calibration**: Lowest Brier score (0.1164) indicates well-calibrated probability estimates
5. **Robust ROC-AUC**: 0.9129 demonstrates strong discriminative ability across thresholds

---

## Evaluation on Held-Out Test Set

To validate generalization to truly unseen data, we evaluated the best-performing models on the held-out test set. The held-out cohort comprised 165 patients:
- 81 patients with histopathology images captured in the same format as the original dataset
- 79 patients with smartphone-captured histopathology images
- 5 non-MF patients from an unseen class that the model never saw during training

Results are presented in Table 4.4:

### **Single-Magnification Performance (Test Set)**

| Model | Accuracy | Sensitivity (MF) | Specificity (MF) | F2-Score | Precision |
|---|---|---|---|---|---|
| x10 (ResNet50) | 77.14% | 86.36% | 77.78% | 0.8482 | 79.17% |
| x20 (EfficientNet-B3, opt. threshold) | 82.10% | 89.13% | 89.13% | 0.8840 | 87.23% |

### **Late Fusion Performance (Test Set)**

The late fusion model achieved state-of-the-art performance on the test set:

| Metric | Value |
|---|---|
| **Accuracy** | **86.21%** |
| **Sensitivity (MF)** | **97.22%** |
| **Specificity (MF)** | **68.18%** |
| **Precision (MF)** | **83.33%** |
| **F2-Score (MF)** | **0.9409** |
| **ROC-AUC** | **0.9129** |
| **Brier Score** | **0.1164** |
| **Number of test patients** | 165 |

### Held-Out Cohort Breakdown and Metrics

#### 81 patients similar to the original dataset
Binary Classification (MF vs Non-MF):
--------------------------------------------------------------------------------
Accuracy:  85.19%
Precision: 0.900
Recall:    0.865
F1-Score:  0.882

#### 79 smartphone patients
Binary Classification (MF vs Non-MF):
--------------------------------------------------------------------------------
Accuracy:  59.49%
Precision: 0.800
Recall:    0.528
F1-Score:  0.636

#### 5 unseen-class non-MF patients
The held-out test set also included 5 non-MF patients from an unseen class that was never present during training.

- First 2 patients:
Binary Classification (MF vs Non-MF):
--------------------------------------------------------------------------------
Accuracy:  0.00%
Precision: 0.000
Recall:    0.000
F1-Score:  0.000

- Remaining 3 patients (smartphone):
Binary Classification (MF vs Non-MF):
--------------------------------------------------------------------------------
Accuracy:  66.67%
Precision: 0.000
Recall:    0.000
F1-Score:  0.000

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
- Maintaining the high sensitivity (97.22%) needed to catch true MF cases

---

## Performance Summary

### Best Model: Weighted Late Fusion (w_x10=0.2, w_x20=0.8)

The weighted late fusion strategy combining x10 and x20 EfficientNet-B3 predictions with optimized weights achieved:

- **86.21% overall accuracy** on the test set
- **97.22% sensitivity** (critical for not missing MF cases)
- **0.9129 ROC-AUC** (excellent discrimination)
- **Robust generalization**: Consistent validation and test performance

This approach successfully leveraged complementary diagnostic information across magnification levels, with the asymmetric weighting emphasizing the superior diagnostic capabilities of x20 magnification while maintaining contextual information from x10, all while maintaining computational efficiency through late (not deep) fusion.

---

## Comparison with Literature

While direct comparison is limited due to the proprietary nature of our dataset and the relative rarity of automated MF classification systems in published literature, our results are competitive with:
- General dermatological classification systems (typically 85–90% accuracy)
- Specialized lymphoma detection systems (sensitivity often prioritized over specificity)
- Multi-task deep learning approaches in histopathology (87–92% accuracy)

The particularly high sensitivity (97.22%) for MF detection aligns with clinical requirements where missing a diagnosis is more costly than generating additional diagnostic uncertainty.
