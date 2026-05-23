# Methodology: Hierarchical Mycosis Fungoides Classification System

## 1. Overview

A hierarchical deep learning framework was developed to diagnose Mycosis Fungoides (MF) and differentiate it from disease mimics using multi-magnification histopathological image analysis combined with clinical features. The system employs **late fusion** of two independent convolutional neural networks trained at different microscopic magnifications (10× and 20×), integrated with a machine learning-based clinical notes classifier.

---

## 2. Histopathological Image Analysis

### 2.1 Preprocessing Pipeline

#### 2.1.1 Patch Extraction

Whole slide images were preprocessed using a sliding window approach to extract diagnostically relevant tissue patches:

**10× Magnification:**
- Patch size: 512 × 512 pixels
- Stride: 256 pixels (50% overlap)
- Maximum patches per image: 150

**20× Magnification:**
- Patch size: 1024 × 1024 pixels
- Stride: 512 pixels (50% overlap)
- Maximum patches per image: 150

#### 2.1.2 Tissue Foreground Detection

To eliminate non-diagnostic regions (background, artifacts, staining anomalies), each patch underwent automated quality filtering:

**Process:**
1. Convert patch from RGB to HSV color space
2. Extract saturation channel to identify stained tissue
3. Apply threshold: foreground pixels = saturation > 20
4. Compute foreground ratio: 

$$\text{fg\_ratio} = \frac{N_{\text{foreground}}}{N_{\text{total}}}$$

5. **Inclusion criterion**: $\text{fg\_ratio} \geq 0.285$ (28.5% minimum tissue threshold)

This tissue detection mechanism ensures only patches containing sufficient cellular material are processed, automatically rejecting background and edge artifacts.

#### 2.1.3 Normalization

All patches were normalized using ImageNet pretraining statistics:

$$\text{Normalized Patch} = \frac{\text{Original} - \mu}{\sigma}$$

Where:
- **Mean**: $\mu = [0.485, 0.456, 0.406]$
- **Std Dev**: $\sigma = [0.229, 0.224, 0.225]$

---

### 2.2 Deep Learning Architecture: Dual-Magnification Ensemble

A two-branch architecture was implemented with independent models trained for each magnification level, capturing complementary diagnostic information.

#### 2.2.1 10× Classifier (Architectural Features)

| Parameter | Value |
|---|---|
| **Base Architecture** | TF-EfficientNet-B3 |
| **Input Resolution** | 512 × 512 pixels |
| **Output Classes** | 5 (MF, PLEVA-PLC, Pseudo-Lymphoma, T-cell dyscrasia, B-cell Lymphoma) |
| **Dropout Rate** | 0.40 |
| **Inference Batch Size** | 8 patches |
| **Model Parameters** | ~10.3M |

**Rationale:** The 10× magnification captures broader tissue architecture and glandular distribution patterns, enabling assessment of infiltration depth and architectural distortion characteristic of MF.

**Training Configuration:**
- Loss function: Cross-entropy with focal loss weighting
- Optimizer: AdamW with gradient clipping
- Learning rate: Cosine annealing with warm restart
- Augmentations: Random rotation (±20°), H/V flips, color jittering, mixup
- Validation metric: F2-score (emphasizing sensitivity)
- Early stopping: Patience = 15 epochs

#### 2.2.2 20× Classifier (Cytological Features)

| Parameter | Value |
|---|---|
| **Base Architecture** | TF-EfficientNet-B3 |
| **Input Resolution** | 512 × 512 pixels |
| **Native Patch Size** | 1024 × 1024 pixels |
| **Output Classes** | 5 |
| **Dropout Rate** | 0.40 |
| **Inference Batch Size** | 2 patches |
| **Model Parameters** | ~10.3M |

**Rationale:** The 20× magnification emphasizes cytological details—nuclear morphology, chromatin patterns, mitotic figures, and immune infiltrates—essential for distinguishing MF from benign disease mimics.

**Training Configuration:**
- Augmentations: Elastic distortions, CLAHE (contrast enhancement), color normalization
- Sampling strategy: Oversampling rare cytological variants
- Focal loss α: Weighted toward ambiguous class pairs
- Validation metric: ROC-AUC (balancing sensitivity and specificity)

---

### 2.3 Patch-to-Patient Aggregation: Mean Probability

Raw patch-level outputs from both models were aggregated to patient-level predictions using **mean probability averaging**:

$$P_{\text{patient}}(c) = \frac{1}{N} \sum_{i=1}^{N} P_{\text{patch}_i}(c)$$

Where:
- $P_{\text{patient}}(c)$ = patient-level probability for class $c$
- $N$ = total number of extracted patches
- $P_{\text{patch}_i}(c)$ = softmax output for patch $i$ and class $c$

**Justification:** This approach assumes that aggregate patch characteristics reflect overall patient pathology, providing robustness against local artifacts while preserving diagnostic signal across the tissue sample.

**Output:** While the individual models can output 5-class probability vectors, for the purpose of the binary fusion (MF vs Non-MF), we extract the scalar **Non-MF probability** for each patient:
- $P^{10x}_{\text{non-MF}} \in [0,1]$
- $P^{20x}_{\text{non-MF}} \in [0,1]$

---

## 3. Decision Thresholding Strategy: Youden's J Optimization

Rather than defaulting to 0.5, optimal decision thresholds were computed independently for each model using **Youden's J statistic**, which balances sensitivity and specificity:

$$J(\theta) = \text{TPR}(\theta) - \text{FPR}(\theta) = \text{Sensitivity} + \text{Specificity} - 1$$

The optimal threshold maximizes this statistic:

$$\theta^* = \arg\max_{\theta} J(\theta)$$

### 3.1 10× Model Threshold

- **Optimal Threshold**: $\theta_{10x}^* = 0.50$
- **Derivation**: ROC analysis on training set (N=487 patients)
- **Decision Rule**: $P(\text{Non-MF}) > 0.50 \rightarrow \text{Non-MF}$; otherwise $\rightarrow \text{MF}$
- **Sensitivity**: 82.35% | **Specificity**: 93.41%

### 3.2 20× Model Threshold

- **Optimal Threshold**: $\theta_{20x}^* = 0.8351$
- **Derivation**: Youden's J maximization on validation cohort
- **Decision Rule**: $P(\text{Non-MF}) > 0.8351 \rightarrow \text{Non-MF}$; otherwise $\rightarrow \text{MF}$
- **Sensitivity**: 84.21% | **Specificity**: 95.10%

**Clinical Interpretation:** The elevated 20× threshold (0.8351 vs 0.50) reflects that cytological mimicry is more prevalent; hence stronger confidence is required to exclude MF. This asymmetry protects against false negatives—critical in cancer diagnosis.

---

## 4. Late Fusion with Probability Calibration

### 4.1 The Calibration Problem

Directly averaging probabilities from both models introduces systematic bias because they operate under different decision calibrations:
- 10× model: 0.5 ≈ indifference point
- 20× model: 0.8351 ≈ indifference point

Without correction, the fused probabilities would be skewed toward the 20× model's calibration.

### 4.2 Recalibration Transformation

The 20× Non-MF probability is recalibrated to align with the standard 0.5 scale using **piecewise-linear transformation**:

$$p_{20x}^{\text{non-MF, cal}} = \begin{cases}
p_{20x}^{\text{non-MF}} \cdot \dfrac{0.5}{0.8351} & \text{if } p_{20x}^{\text{non-MF}} < 0.8351 \\[0.5em]
0.5 + \left(p_{20x}^{\text{non-MF}} - 0.8351\right) \cdot \dfrac{0.5}{1.0 - 0.8351} & \text{if } p_{20x}^{\text{non-MF}} \geq 0.8351
\end{cases}$$

**Transformation Effects:**
1. Maps $[0, 0.8351] \rightarrow [0, 0.5]$ (stretching lower probabilities)
2. Maps $[0.8351, 1.0] \rightarrow [0.5, 1.0]$ (stretching upper probabilities)
3. Ensures $0.8351 \rightarrow 0.5$ (calibration point alignment)

**Result:** Both models now operate on comparable probability scales, enabling valid weighted fusion.

---

### 4.3 Weighted Fusion via Brier Score Optimization

The calibrated Non-MF probabilities are combined via weighted averaging:

$$P_{\text{fused}}^{\text{non-MF}} = w \cdot P_{10x}^{\text{non-MF}} + (1-w) \cdot P_{20x}^{\text{non-MF, cal}}$$

The Brier Score (mean squared prediction error) was tracked to measure calibration:

$$\text{BS} = \frac{1}{N} \sum_{i=1}^{N} (P_{\text{fused},i} - y_i)^2$$

**Rationale for Brier Score:** Unlike accuracy (binary outcome), Brier score penalizes both:
- Incorrect predictions
- Overconfident correct predictions

In medical AI, this prevents dangerous overconfidence in predictions. While the strictly optimal weight minimizing the Brier Score was found to be $w = 0.60$ (Brier: 0.1401), a **clinical golden weight** of $w^* = 0.20$ was manually selected to prioritize sensitivity.

#### 4.3.1 Weight Sweep Results

Systematic evaluation across $w \in [0.0, 1.0]$ with step 0.05. A subset of key intervals is shown:

| Weight | Brier Score | Accuracy | Sensitivity (MF) | Specificity (MF) | ROC-AUC | F2 (MF) |
|---|---|---|---|---|---|---|
| 0.00 | 0.1526 | 82.09% | 89.13% | 89.13% | 0.8509 | 0.8836 |
| 0.10 | 0.1486 | 83.58% | 89.13% | 89.13% | 0.8634 | 0.8874 |
| **0.20** | **0.1455** | **83.58%** | **89.13%** | **89.13%** | **0.8778** | **0.8874** |
| 0.40 | 0.1413 | 77.61% | 80.43% | 80.43% | 0.8882 | 0.8150 |
| *0.60* | *0.1401* | *79.10%* | *80.43%* | *80.43%* | *0.8861* | *0.8186* |
| 0.80 | 0.1418 | 80.60% | 80.43% | 80.43% | 0.8789 | 0.8222 |
| 1.00 | 0.1466 | 80.60% | 80.43% | 80.43% | 0.8685 | 0.8222 |

**Selected Clinical Weight:** $w^* = 0.20$

**Interpretation:**
- **20% contribution** from 10× architectural model
- **80% contribution** from 20× cytological model
- **Clinical reasoning**: While architectural patterns provide context, cytological details are more distinctive for MF diagnosis. The weight $w = 0.20$ maximizes MF Sensitivity (89.13%) and F2-Score (0.8874) without compromising overall Accuracy (83.58%), providing the safest clinical threshold.
- **Performance at w=0.20**: Brier: 0.1455 | Acc: 83.58% | Sens: 89.13% | Spec: 89.13% | AUC: 0.8778

---

## 5. Multi-Class Prediction Logic

> [!NOTE]
> The late fusion pipeline primarily focuses on the binary MF vs Non-MF decision using the fused scalar probabilities. The multi-class refinement described below is a theoretical capability of the underlying individual models (which can output 5-class vectors), but it is not applied during the scalar late fusion step.

### 5.1 Binary Decision (MF vs Non-MF)

Using the fused Non-MF probability:

$$\hat{y} = \begin{cases}
\text{MF} & \text{if } P_{\text{fused}}^{\text{non-MF}} \leq 0.5 \\
\text{Non-MF} & \text{if } P_{\text{fused}}^{\text{non-MF}} > 0.5
\end{cases}$$

### 5.2 Multi-Class Refinement

For Non-MF cases, the specific disease mimic is determined:

**Algorithm:**
1. **Mask MF class**: Set $P_{\text{fused}}(1) = -\infty$ to prevent MF assignment
2. **Select maximum**: 
   $$c^* = \arg\max_{c \in \{0,2,3,4\}} P_{\text{fused}}(c)$$
3. **Assign**: Predicted class = $\text{CLASS\_NAMES}[c^*]$

**Class Mapping** (5-way):
- Index 0: **B-cell Lymphoma**
- Index 1: **Mycosis Fungoides (MF)**
- Index 2: **PLEVA-PLC**
- Index 3: **T-cell dyscrasia**
- Index 4: **Pseudolymphoma**

---

## 6. Clinical Features Classifier: Machine Learning Module

### 6.1 Clinical Feature Engineering

A complementary classifier was trained on patient metadata to leverage diagnostic information beyond histology.

#### 6.1.1 Feature Selection (16 Selected Features)

Univariate feature screening on a larger initial pool yielded 16 predictive features (used after filtering down to numeric representations):

| Feature | Type | Domain | Values/Description |
|---|---|---|---|
| **Age** | Continuous | Demographics | Patient age (years) |
| **Duration (Months)** | Continuous | Disease history | Disease duration in months |
| **Course** | Categorical | Disease trajectory | Progressive, Stationary, Remitting, etc. |
| **Visit Type** | Categorical | Visit context | New, Follow-up, Recurrent |
| **Site: Head and Neck** | Binary | Anatomic distribution | Head/Neck involvement (Yes/No) |
| **Site: Lower Limbs** | Binary | Limb involvement | Upper/Lower limb (Yes/No) |
| **Lesion Color** | Categorical | Lesion phenotype | Erythematous, Hyperpigmented, etc. |
| **Macules** | Binary | Morphology | Presence (Yes/No) |
| **Patch** | Binary | Morphology | Presence (Yes/No) |
| **Papules** | Binary | Morphology | Presence (Yes/No) |
| **Plaque** | Binary | Morphologic stage | Presence (Yes/No) |
| **Nodule** | Binary | Morphologic stage | Presence (Yes/No) |
| **Scales** | Binary | Surface features | Presence (Yes/No) |
| **Biopsy 1 Morphology** | Categorical | Biopsy findings | Patch, Plaque, Nodule, Macule |
| **Biopsy 2 Morphology** | Categorical | Biopsy findings | Patch, Plaque, Nodule, Macule |
| **Biopsy 2 Site** | Categorical | Biopsy location | Multiple site options |

**Selection Rationale:** Features selected based on both statistical significance and clinical plausibility (confirmed by dermatopathology experts).

#### 6.1.2 Data Preprocessing

- **Continuous features**: Median imputation for missingness
- **Categorical features**: Label encoding (alphabetical order)
- **Binary features**: 0/1 encoding (No=0, Yes=1)
- **Class weighting**: $\text{scale\_pos\_weight} = 12$ (addressing MF/Non-MF imbalance)

---

### 6.2 Three-Model Comparison Framework

Three distinct algorithms were systematically evaluated using **5-fold stratified cross-validation** to identify the optimal clinical classifier:

#### 6.2.1 Model 1: XGBoost (Gradient Boosting)

**Hyperparameters:**
```python
n_estimators       = 300
max_depth          = 4
learning_rate      = 0.05
subsample          = 0.8
colsample_bytree   = 0.8
scale_pos_weight   = 12
eval_metric        = 'logloss'
```

**Strengths:**
- Captures non-linear feature interactions
- Built-in feature importance via gain/cover metrics
- Inherent handling of missing values
- Robust to outliers

**Clinical Relevance:** Captures complex disease phenotype patterns (e.g., specific site + morphology combinations).

#### 6.2.2 Model 2: Random Forest

**Hyperparameters:**
```python
n_estimators = 300
max_depth    = 6
criterion    = 'gini'
```

**Strengths:**
- Excellent generalization via ensemble averaging
- Reduced overfitting vs single tree
- Interpretable feature importances
- Minimal hyperparameter sensitivity

**Role in Comparison:** Validates that top features from XGBoost are broadly predictive across different tree algorithms.

#### 6.2.3 Model 3: Logistic Regression

**Hyperparameters:**
```python
C           = 0.5
solver      = 'lbfgs'
max_iter    = 1000
penalty     = 'l2'
```

**Strengths:**
- Direct probabilistic interpretation (odds ratios)
- Fast inference
- Transparent, linear feature relationships
- Baseline for non-linearity benefit quantification

**Role in Comparison:** Linear baseline establishing additive value of tree-based non-linearity.

---

### 6.3 Comparative Evaluation

#### 6.3.1 Evaluation Metrics

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Sensitivity (Recall)} = \frac{TP}{TP + FN}$$

$$F_2 = \frac{5 \cdot \text{Precision} \cdot \text{Sensitivity}}{4 \cdot \text{Precision} + \text{Sensitivity}}$$

(F2-score emphasizes sensitivity with $\beta=2$ to minimize missed MF diagnoses)

#### 6.3.2 Results on Diagnosis Task (MF vs Non-MF)

| Metric | XGBoost | Random Forest | Logistic Regression |
|---|---|---|---|
| **Accuracy** | 0.956 | **0.966** | 0.950 |
| **Precision** | 0.908 | **0.948** | 0.915 |
| **Sensitivity** | **0.945** | 0.938 | 0.918 |
| **F2-Score** | 0.937 | **0.939** | 0.916 |
| **ROC-AUC** | 0.994 | **0.995** | 0.985 |

**Key Findings:**
- **Random Forest dominant**: Random Forest outperforms other models on 4 out of 5 metrics (Accuracy, Precision, F2-Score, and ROC-AUC), making it the strongest overall model.
- **High Discrimination**: Random Forest achieves an exceptional ROC-AUC of 0.995 and Accuracy of 0.966.
- **Sensitivity**: XGBoost has slightly higher sensitivity (0.945 vs 0.938), but Random Forest's superior precision (0.948 vs 0.908) makes it a more balanced and reliable classifier.
- **Selected Model**: **Random Forest** chosen as the primary clinical classifier for its reliability, superior generalization, and interpretability.

#### 6.3.3 Feature Importance Analysis

**Diagnosis Task (MF vs Non-MF):**

To provide interpretable feature rankings, feature importance was visualized using XGBoost with SHAP (SHapley Additive exPlanations) values, which provide theoretically principled feature importance estimates:

![!\[SHAP Feature Importance - XGBoost, Diagnosis Task\](shap_importance.png)](<Clinical Notes Classifier/Graphs/shap_importance_task1.png>)

**Top 5 Predictive Features:**

| Rank | Feature | Importance | Clinical Significance |
|---|---|---|---|
| 1 | **macules** | 0.30 | Primary morphologic indicator |
| 2 | **papules** | 0.15 | Secondary morphology |
| 3 | **patch** | 0.13 | Early MF morphology |
| 4 | **age** | 0.11 | Demographic risk factor |
| 5 | **duration_months** | 0.10 | Disease chronicity |

**Stage of MF Prediction (Task 2):**

For MF patients, feature importance in predicting disease stage (Patch-Plaque vs Tumor):

![!\[XGBoost Feature Importance - Stage of MF\](fi_Stage_of_MF.png)](<Clinical Notes Classifier/Graphs/fi_Stage_of_MF.png>)

**Top Feature:** **nodule** is most predictive of advanced tumor stage (importance: 0.65), which aligns with nodular morphology being a hallmark of tumor-stage MF.

**Clinical Validation:** These feature rankings align with established dermatopathologic knowledge and MF disease progression patterns.

---

## 7. Hierarchical Classification Architecture

The complete diagnostic system operates via a four-stage pipeline:

### Stage 1: Histopathological Analysis
1. Extract patches at 10× and 20× from patient specimen
2. Independent inference: EfficientNet-B3 for each magnification
3. Aggregate to patient-level via mean pooling
4. Apply magnitude-specific thresholds (0.50, 0.8351)

### Stage 2: Probability Calibration & Fusion
1. Recalibrate 20× probabilities to standard 0.5 scale (piecewise-linear transformation)
2. Weighted fusion: 
   $$P_{\text{fused}} = 0.20 \cdot P_{10x} + 0.80 \cdot P_{20x}^{\text{cal}}$$

### Stage 3: Clinical Integration
1. Extract 15 clinical features
2. Random Forest inference for supporting evidence
3. Augment image-based confidence

### Stage 4: Final Decision
1. **Binary classification**: MF vs Non-MF (threshold = 0.5)
2. **Multi-class refinement**: Assign specific mimic class if Non-MF

```
┌─────────────────────────────────────────────────────────────┐
│ Input: Histopathology Images (10x, 20x) + Clinical Data    │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
    [Image Path]              [Clinical Path]
         │                           │
         ├─10x Model            Random Forest
         │ (EfficientNet-B3)      (300 trees)
         │                      Features: 16
         ├─20x Model                 │
         │ (EfficientNet-B3)     MF vs Non-MF
         │                           │
         └────────┬──────────────────┘
                  │
         ┌────────▼────────┐
         │  Late Fusion    │
         │  w=0.20 (clinic)│
         │  Calibration    │
         └────────┬────────┘
                  │
         ┌────────▼──────────────────┐
         │ Binary Prediction         │
         │ (MF vs Non-MF)            │
         │ threshold = 0.5           │
         └────────┬──────────────────┘
                  │
         ┌────────▼──────────────────┐
         │ Multi-Class Refinement    │
         │ (if Non-MF: classify as) │
         │ B-cell, PLEVA, T-cell,   │
         │ or Pseudo-Lymphoma       │
         └────────┬──────────────────┘
                  │
         ┌────────▼──────────────────┐
         │ Final Output:             │
         │ - Diagnosis               │
         │ - Confidence Score        │
         │ - Per-Class Probabilities │
         │ - Feature Attribution     │
         └───────────────────────────┘
```

---

## 8. Key Methodological Innovations

| Innovation | Description |
|---|---|
| **Dual-Magnification Architecture** | Complementary 10× (architecture) and 20× (cytology) models capture multi-scale pathology |
| **Youden's J Thresholding** | Optimal decision thresholds (0.50, 0.8351) derived from ROC analysis rather than default 0.5 |
| **Probability Recalibration** | Novel piecewise-linear transformation enables meaningful weighted averaging of differently-calibrated model outputs |
| **Clinical Weight Selection** | While Brier score optimization found w=0.60, a clinical golden weight (w=0.20) was manually selected to prioritize and maximize sensitivity. |
| **Hierarchical Classification** | Binary MF/Non-MF decision followed by multi-class mimic differentiation |
| **Clinico-Histologic Integration** | Late fusion of deep learning image analysis with machine learning-based clinical features classifier |
| **Rigorous Model Selection** | Three-model comparison (XGBoost vs Random Forest vs Logistic Regression); Random Forest selected for robustness and interpretability |

---

## 9. Implementation Details

**Software Stack:**
- **Deep Learning:** PyTorch 2.0+, timm (EfficientNet)
- **Classical ML:** scikit-learn, XGBoost
- **Image Processing:** OpenCV, Pillow
- **Metrics:** torchmetrics, sklearn.metrics
- **GPU Support:** CUDA 12.1 (with CPU fallback)

**Data Specifications:**
- **Total patients:** 499 (353 MF, 146 Non-MF)
  - **Training set:** 328 patients
  - **Validation set:** 67 patients (used for threshold optimization and hyperparameter tuning)
  - **Test set:** 148 patients (67 + 81 from separate test cohorts)

- **10× Magnification Patches:**
  - Training: 73,824 patches
  - Validation: 14,804 patches
  - Test: 13,190 patches

- **20× Magnification Patches:**
  - Training: 23,062 patches
  - Validation: 4,728 patches
  - Test: 4,407 patches

- **Test set composition:** Multiple disease subtypes (MF, PLEVA-PLC, T-cell dyscrasia, Pseudo-Lymphoma, B-cell Lymphoma)

**Reproducibility:** All hyperparameters, random seeds, and cross-validation splits are documented for complete reproducibility.

---

## 10. Summary Table

| Component | Method | Key Parameter(s) | Performance |
|---|---|---|---|
| **10× Model** | EfficientNet-B3 | Input: 512×512, Dropout: 0.4 | Sensitivity: 82.35% |
| **20× Model** | EfficientNet-B3 | Input: 512×512, Dropout: 0.4 | Sensitivity: 84.21% |
| **Aggregation** | Mean pooling | Patch → Patient level | Preserves diagnostic signal |
| **Threshold (10×)** | Youden's J | θ = 0.50 | Balanced sensitivity/specificity |
| **Threshold (20×)** | Youden's J | θ = 0.8351 | High specificity for cytology |
| **Calibration** | Piecewise-linear | Maps 0.8351 → 0.5 | Enables valid fusion |
| **Fusion** | Weighted averaging | w = 0.20 (clinical) | Brier: 0.1455, Acc: 83.58%, Sens: 89.13% |
| **Clinical** | Random Forest | 16 features, max_depth: 6, 300 trees | Sensitivity: 93.8%, Accuracy: 96.6% |
| **Binary Decision** | Threshold | θ = 0.5 | Separates MF vs Non-MF |
| **Multi-class** | Argmax masking | Mask MF, select max | Differentiates 5 classes |

---

**End of Methodology Section**
