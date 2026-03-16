# EXPERIMENTS — Text Authorship Detection (MALTO Hackathon)

Tracking every run: what changed, what the CV F1 was, what the Kaggle leaderboard F1 was.

**Metric**: Macro F1 (all 6 classes weighted equally)
**Classes**: Human=0, DeepSeek=1, Grok=2, Claude=3, Gemini=4, ChatGPT=5
**Class counts (train)**: Human=1520, DeepSeek=80, Grok=160, Claude=80, Gemini=240, ChatGPT=320
**Key challenge**: DeepSeek (80 samples) recall was near 0 before run 2.

---

## Run 1 — Baseline TF-IDF + Linear Models
**Date**: 2026-03-16  
**Kaggle public LB**: 0.85942 (rank 15/16, 1st place = 0.95081)  
**Best CV model**: sgd_hinge — 0.8943 macro F1

### Config
- Word TF-IDF: (1,2) ngrams, 100k features
- Char TF-IDF: (3,5) ngrams, 100k features
- No stylometric features
- Models: logistic_regression, logistic_regression_balanced, linear_svc,
  linear_svc_balanced, calibrated_svc, sgd_log, sgd_hinge, ridge_classifier,
  passive_aggressive, complement_nb

### CV Results
| Model                        | CV Macro F1 |
|------------------------------|-------------|
| sgd_hinge                    | 0.8943      |
| logistic_regression_balanced | ~0.883      |
| ridge_classifier              | ~0.882      |

### Issues identified
- DeepSeek recall ≈ 0.07 (catastrophic — most predicted as Grok)
- CV (0.8943) >> Kaggle LB (0.8594) — 0.035 gap

---

## Run 2 — Stylometric (34 features) + Soft Ensemble + Threshold Optimization
**Date**: 2026-03-16  
**Kaggle public LB**: (submit artifacts/submissions/submission_latest.csv)  
**Best CV model**: logistic_regression_balanced — **0.9140** macro F1  
**After threshold opt (OOF)**: **0.9192** macro F1

### Changes from Run 1
1. **Char TF-IDF widened**: (3,5) → **(2,6)** — captures more subword patterns
2. **Stylometric features added (34 features)** — up from 0:
   - Base: char/word counts, avg sentence/word length, type-token ratio, punctuation rates
   - NEW: `very_short_sent_ratio` (< 5 words) — DeepSeek tends to write short sentences
   - NEW: `very_long_sent_ratio` (> 30 words) — Grok tends longer sentences
   - NEW: `numbered_list_rate` — DeepSeek loves numbered lists
   - NEW: `markdown_header_rate` — Gemini/Claude use markdown headers
   - NEW: `code_block_rate`, `bold_rate`, `italic_rate`, `link_rate`
   - NEW: `parenthesis_rate`, `ellipsis_rate`, `dash_rate`, `caps_word_ratio`
   - NEW: `avg_para_len_chars`, `starts_with_i_ratio`
3. **ensemble_soft model** (new): soft voting with calibrated models
   - Components: CalibratedClassifierCV(sgd_hinge) + CalibratedClassifierCV(ridge) +
     LogisticRegression(balanced) + CalibratedClassifierCV(LinearSVC)
4. **Per-class threshold optimizer** (new post-processing):
   - Grid-searches per-class probability scale factors on OOF data
   - Optimizes macro F1 directly
   - Thresholds [1.25, 1.25, 1.0, 1.0, 1.25, 1.75] saved to artifacts/thresholds.json
   - Applied automatically at inference time

### CV Results (all models, this run)
| Model                        | CV Macro F1 | OOF Macro F1 |
|------------------------------|-------------|--------------|
| logistic_regression_balanced | **0.9140**  | 0.9145       |
| ensemble_soft                | 0.9025      | 0.9026       |
| ensemble_top3                | 0.9006      | 0.9011       |
| sgd_hinge                    | 0.8936      | 0.8938       |
| ridge_classifier              | 0.8892      | 0.8904       |
| calibrated_svc               | 0.8719      | 0.8735       |

### Per-class F1 for logistic_regression_balanced (OOF)
| Class   | Before threshold opt | After threshold opt |
|---------|---------------------|---------------------|
| Human   | 0.998               | 0.999               |
| DeepSeek| **0.637** (was 0.07)| **0.671**           |
| Grok    | 0.865               | 0.852               |
| Claude  | 1.000               | 1.000               |
| Gemini  | 0.994               | 0.998               |
| ChatGPT | 0.992               | 0.995               |
| **Macro**| **0.9145**         | **0.9192**          |

### Analysis
- **DeepSeek recall massively improved**: 0.07 → 0.637
  - Root cause: stylometric features (numbered_list_rate, very_short_sent_ratio) give
    clear signal to separate DeepSeek from Grok
- **logistic_regression_balanced > sgd_hinge**: LR with LBFGS solver can find a better
  global optimum; the balanced weights matter more with the richer feature set
- **Overfit gap still present** (~0.10): models memorize training features; gap suggests
  room for regularization improvement

### Final submission
- Model: `logistic_regression_balanced`
- Thresholds: [1.25, 1.25, 1.0, 1.0, 1.25, 1.75]
- Submission: `artifacts/submissions/submission_latest.csv`

---

## Run 3 — Reduced Overfit + Seed Averaging + More DeepSeek Features
**Date**: 2026-03-16
**Kaggle public LB**: (pending — run python main_train.py then main_infer.py)
**Target**: > 0.91 LB

### Changes from Run 2
1. **37 stylometric features** (up from 34) — 3 new DeepSeek vs Grok discriminators:
   - `punct_variety`: ratio of distinct punctuation chars (DeepSeek uses very few)
   - `sent_length_cv`: coefficient of variation of sentence lengths (DeepSeek is uniform)
   - `transition_word_rate`: "however/therefore/moreover/..." per sentence
2. **Reduced feature space to fight CV-LB gap**:
   - word TF-IDF: max_features 100k -> 50k, min_df 2 -> 3
   - char TF-IDF: max_features 100k -> 50k, min_df 3 -> 4
   - Reason: overfit gap ~0.048 (CV 0.9140 vs LB 0.8660) — model memorises rare words
3. **More regularization**: LR C: 1.0 -> 0.5 (less memorization of training text)
4. **`lr_seed_avg` model** (new): runs LR 5x with seeds [42,123,456,789,2024],
   averages predicted probabilities — reduces variance on minority classes
5. **`use_best_fold_model: true`**: saves the best CV fold's fitted pipeline and
   uses it as the final model (instead of retraining on all 2400 samples)
   — the best fold model saw only 1920 samples, so it generalizes better to test

### Hypothesis
- Reducing max_features + increasing regularization will close the LB gap
- Seed averaging will give more stable DeepSeek predictions
- Best-fold model avoids the overfit that comes from training on all data


### Bug fixes applied before Run 3
Three pipeline bugs found and fixed:
1. **Model mismatch**: `best_model` in config was used for final training, but thresholds
   were computed for the CV winner. Fixed: CV winner now ALWAYS overrides config for final
   training, so model and thresholds always match.
2. **Thresholds applied to wrong model**: threshold_optimizer uses OOF proba of the CV
   winner; applying those thresholds to a different model breaks the calibration.
   Fixed by fix #1 above.
3. **`use_best_fold_model`**: was True, meaning only 80% of data was used for final model.
   Set to False — retrain on all 2400 samples for the submitted model.
4. **Submission not saved to experiment folder**: added shutil copy of submission CSV into
   each experiment's directory for full reproducibility.

### How to run
```
cd d:\hachaton	ext-authorship-detection
.env\Scriptsctivate
python main_train.py --config configs/config.yaml
python main_infer.py --config configs/config.yaml
```
Then submit artifacts/submissions/submission_latest.csv to Kaggle.

---

## Run 4 — Two-Stage Classifier + DeepSeek Boost + Function-Word TF-IDF
**Date**: 2026-03-16
**Kaggle public LB**: **0.91089** (rank 10/16) — submitted ensemble_soft (CV winner)
**Target**: > 0.90 LB (fix DeepSeek recall=0.59, 31 DS->Grok errors)

### Root cause of Run 3 errors (GPT analysis)
- 31/80 DeepSeek samples predicted as Grok
- DeepSeek F1 ≈ 0.59 — the single worst class
- DeepSeek and Grok have similar AI writing style; the 6-class model can't sharply
  distinguish them using the same boundary as all other classes

### Changes from Run 3
1. **`two_stage_lr` model** (new): Two-stage classifier
   - Stage 1: full 6-class LR(balanced, C=0.5) trained on all 2400 samples
   - Stage 2: binary DeepSeek-vs-Grok LR(balanced, C=2.0) trained only on
     DeepSeek+Grok samples (240 samples, simpler 2-class problem)
   - At inference: stage-1 predicts; if prediction is DS or Grok → stage-2 overrides
   - predict_proba: redistributes DS+Grok probability mass via binary stage
   - Hypothesis: a dedicated decision boundary beats sharing it with 4 other classes

2. **`lr_deepseek_boost` model** (new): Custom class weights
   - Balanced weights + DeepSeek weight ×2.0 = class 1 weight ≈ 10.0
   - Forces the model to focus harder on DeepSeek recall
   - Config: `models.deepseek_boost: 2.0`

3. **Function-word TF-IDF** (new 4th FeatureUnion component)
   - Fixed vocabulary of ~130 function/connector words
   - Captures style fingerprint (how author connects ideas) not topic
   - Different LLMs have very distinct rates: DeepSeek heavy on "however/therefore/moreover",
     Grok more varied, Human uses "and/but/so" far more
   - Config: `features.function_word_tfidf.enabled: true`

4. **char TF-IDF min_df: 4 → 2**
   - Exposes rare but class-specific char n-grams
   - e.g., DeepSeek's typical punctuation sequences like ": \n" after bullets

5. **Report**: `_feature_importance.csv` saved alongside the best model — shows top-50
   TF-IDF features per class (training-only, zero inference overhead)

### How to run
```
cd d:\hachaton\text-authorship-detection
.env\Scripts\activate
python main_train.py --config configs/config.yaml
python main_infer.py --config configs/config.yaml
```
Then submit `artifacts/submissions/submission_latest.csv` to Kaggle.

### Run 4 actual results (OOF CV)
| Model                        | CV Macro F1 | DS→Grok | Grok→DS |
|------------------------------|-------------|---------|---------|
| ensemble_soft  (submitted)   | **0.9321**  | 27      | 8       |
| logistic_regression_balanced | 0.9300      | —       | —       |
| lr_seed_avg                  | 0.9300      | —       | —       |
| two_stage_lr                 | 0.9295      | 19 (-8) | 19 (+11)|
| lr_deepseek_boost            | 0.9275      | —       | —       |

**Diagnosis**: two_stage_lr reduced DS→Grok errors but over-reclassified 11 extra Grok→DS.
Root cause: binary stage with class_weight='balanced' + C=2.0 was too aggressive toward DS.

---

## Run 5 — Smart Margin Trigger + ensemble_two_stage
**Date**: 2026-03-16
**Kaggle public LB**: (pending)
**Target**: > 0.93 LB, rank top-5

### Root cause fix for Run 4 over-correction
Run 4's two_stage_lr triggered stage-2 for ALL predicted DS/Grok samples, even when the
base model was highly confident it was Grok (e.g. P(Grok)=0.85). This led to 11 Grok
samples being wrongly reclassified as DS.

Fix: **margin trigger** — only invoke stage-2 when |P(DS)-P(Grok)| < 0.40 (base uncertain).
- Clear Grok (margin > 0.40): trust base, don't reclassify
- Uncertain (margin < 0.40): use binary specialist to refine the boundary

### Changes from Run 4
1. **`TwoStageClassifier` updated**:
   - New param: `margin_trigger_gap` (default None, set to 0.40 for conservative)
   - New param: `binary_ds_threshold` (default 0.50, 0.52 for conservative)
   - Default binary: C=1.0, class_weight=None (natural 1:2 DS:Grok ratio)
   - All existing two_stage_lr behaviour preserved (no margin trigger = old behaviour)

2. **`two_stage_conservative` model** (new):
   - margin_trigger_gap=0.40: only refine when base uncertain
   - binary_ds_threshold=0.52: slightly conservative DS prediction
   - binary class_weight=None: natural 1:2 ratio avoids over-predicting DS

3. **`ensemble_two_stage` model** (new — likely best):
   - VotingClassifier(soft) of 3 diverse models:
     - `ensemble_soft`: best overall calibration, best Grok recall (0.9321 CV)
     - `two_stage_conservative`: smarter DS/Grok boundary
     - `lr_deepseek_boost`: nudges DS probabilities up across the board
   - Averaging reduces variance; all 3 have predict_proba → proper soft voting

4. **`main_submit.py` removed** — everything done via main_train.py + main_infer.py

5. **`main_infer.py` updated**:
   - `--list-models`: lists all .joblib files under artifacts/ with their sizes
   - `--model <path>`: already existed, now clearly documented in docstring

### Config tuning knobs (try adjusting these)
- `models.two_stage_margin_gap: 0.40`  — raise → less conservative, lower → more conservative
- `models.two_stage_ds_threshold: 0.52` — raise → harder for binary to predict DS

### How to run
```
cd d:\hachaton\text-authorship-detection
.env\Scripts\activate
python main_train.py --config configs/config.yaml
python main_infer.py --config configs/config.yaml

# Run inference with a specific model from a previous run:
python main_infer.py --model artifacts/experiments/2026-03-16_161401_run/best_model.joblib

# List all saved models:
python main_infer.py --list-models
```

---

## Run 6 — DS/Grok Targeted Improvements + Pair Threshold
**Date**: 2026-03-16
**Kaggle public LB**: (pending — run main_train.py then main_infer.py)
**Target**: > 0.93 LB — fix DeepSeek recall from 0.62

### Root cause analysis of DS->Grok=27 errors
From error_analysis.csv, the 27 false-negative DeepSeek samples split into two buckets:
1. **Short factual/definition style** (majority): 1-2 sentence factual texts about
   science, history, geography. No markdown, no lists. Visually indistinguishable
   from Grok's concise factual style.
2. **Long encyclopedic style** (minority): Multi-paragraph historical texts without
   DeepSeek's typical structured output (headers, numbered lists).
Both types lack DeepSeek's primary style markers -> the model falls back to topic,
which is unreliable since DS and Grok cover the same topics.

### Changes from Run 5
1. **43 stylometric features** (up from 37) — 6 new DS/Grok discriminators:
   - `first_sent_words`: DS short texts often start with a precise one-line definition
   - `proper_noun_density`: factual DS texts tend to reference more named entities
   - `hedge_rate`: uncertainty markers (may/might/could/possibly) — Grok hedges more
   - `question_per_sent`: DS almost never asks rhetorical questions
   - `sent_range`: max-min sentence word count (DS more length-uniform)
   - `text_len_log`: log of text length (length bucket is a style discriminator)

2. **`two_stage_top2` model** (new):
   - Trigger: fires ONLY when DS AND Grok are BOTH in the top-2 predicted classes
   - More precise than margin trigger — doesn't fire when the runner-up is Claude/Human
   - Binary stage: LR(balanced, C=1.5) — less regularised than conservative variant
   - Config: `binary_ds_threshold=0.50` (balanced DS/Grok prediction)
   - Hypothesis: current margin trigger was firing on ambiguous DS-vs-Claude/Human cases
     that the binary DS/Grok specialist then forced into DS or Grok incorrectly

3. **Pair-specific DS/Grok threshold** (new post-processing):
   - After global classification, apply ratio threshold `P(DS)/(P(DS)+P(Grok))`
   - Optimised on OOF data via grid search [0.25, 0.75] to maximise macro F1
   - Only affects samples where `P(DS)+P(Grok) > 0.15` — safe, no other-class leakage
   - Saved to `artifacts/thresholds.json` as `ds_grok_pair_threshold`
   - Applied automatically in `main_infer.py`

4. **Bug fix (High)**: stale threshold warning in `main_infer.py`
   - Now warns when loaded threshold model name differs from `--model` path
   - Prevents silent application of wrong-model thresholds

5. **Bug fix (Medium)**: single-model CV branch now computes and saves thresholds
   - Previously the `elif run_cv` branch skipped threshold computation
   - Now saves `thresholds.json` + pair threshold in both paths

6. **Removed from run_models**: `two_stage_conservative`, `ensemble_two_stage`,
   `two_stage_lr` — all confirmed worse than `ensemble_soft` in Run 4/5

### How to run
```
cd d:\hachaton\text-authorship-detection
.env\Scripts\activate
python main_train.py --config configs/config.yaml
python main_infer.py --config configs/config.yaml
```
Then submit `artifacts/submissions/submission_latest.csv` to Kaggle.

---

## Run 7 — Net DS/Grok Error Reduction (5 models)
**Date**: 2026-03-16
**Kaggle public LB**: (pending — same CV winner two_stage_top2, submit to confirm)
**Target**: Reduce total DS+Grok errors below 30

### Run 6 diagnosis
Run 6 two_stage_top2 improved DS recall (0.61→0.70) but traded:
- DS→Grok: 27 → 21 (-6 good)
- Grok→DS: 8 → 14 (+6 bad)
Net: 35 total errors, same as before. Only 5/600 test labels changed → no LB jump.
Root cause: the balanced-weight binary (C=1.5) is too aggressive — it corrects true
DS samples but also over-claims ambiguous Grok samples as DS.

### Three targeted hypotheses (GPT + own analysis)
1. **Conservative binary** (my suggestion): Keep top2 trigger but use natural 1:2
   DS:Grok weights in the binary. Natural ratio biases toward Grok when uncertain,
   which should recover the 6 extra Grok→DS errors without losing DS recall.

2. **Combined trigger** (GPT suggestion): Fire only when BOTH top2 AND margin<0.30.
   This is the most selective trigger possible — only fires on truly ambiguous cases
   where the model has barely any preference between DS and Grok. Near-zero chance
   of creating Grok→DS false positives.

3. **Ensemble v2** (my suggestion): Soft average of two_stage_top2 + ensemble_soft.
   The two models are complementary: top2 has better DS recall (0.70), soft has better
   Grok recall (0.93) and overall precision. Averaging should get the best of both.

### Changes from Run 6
1. **`two_stage_top2_conservative`** (new): top2_trigger=True + binary C=0.8 +
   class_weight=None (natural 1:2 DS:Grok) + ds_threshold=0.52
2. **`two_stage_combined`** (new): top2_trigger=True AND margin_trigger_gap=0.30 +
   binary C=1.5 + class_weight="balanced". AND logic: most selective trigger.
3. **`ensemble_v2`** (new): VotingClassifier(soft) of [two_stage_top2 + ensemble_soft]
4. **Bug fix**: `TwoStageClassifier.predict()` now correctly fetches base_proba when
   top2_trigger=True (was only fetching for margin_trigger_gap previously)
5. **Enhancement**: `_compute_trigger_mask` now uses AND logic when both top2_trigger
   AND margin_trigger_gap are set — enables the two_stage_combined model

### Run 7 actual results (OOF CV)
| Model                       | CV Macro F1 | DS→Grok | Grok→DS | DS recall |
|-----------------------------|-------------|---------|---------|-----------|
| two_stage_top2              | **0.9328**  | 21      | 14      | 0.70      |
| ensemble_v2                 | 0.9292      | 26      | 12      | 0.64      |
| two_stage_combined          | 0.9277      | 20      | 19      | 0.71      |
| ensemble_soft               | 0.9272      | 28      | 10      | 0.61      |
| two_stage_top2_conservative | 0.9061      | **40**  | **5**   | **0.46**  |

### Run 7 diagnosis — linear ceiling confirmed
- **two_stage_top2_conservative FAILED**: natural binary weights (1:2 DS:Grok ratio)
  with C=0.8 massively over-biased toward Grok. DS recall dropped from 0.70 → 0.46.
  Root cause: C=0.8 + class_weight=None + ds_threshold=0.52 = triple Grok bias.
- **two_stage_combined (GPT suggestion)**: AND trigger (top2 AND margin<0.30) achieved
  best DS→Grok count (20) but worst Grok→DS (19). The AND logic is so selective that
  it only fires on the most ambiguous cases — where the binary has least information,
  so it over-corrects toward DS to compensate, creating Grok→DS false positives.
- **ensemble_v2**: Averaging two_stage_top2 with ensemble_soft diluted the DS recall
  improvement (0.70 → 0.64). The soft ensemble smoothed out the two_stage corrections.
- **Conclusion**: All linear models are capped at CV ~0.9328. Every trigger/weight
  combination just trades DS→Grok errors for Grok→DS errors without reducing the total.
  The DS/Grok boundary is **non-linearly inseparable** in TF-IDF space for short texts
  that cover the same topics. A neural network is needed.

### How to run
```
cd d:\hachaton\text-authorship-detection
.env\Scripts\activate
python main_train.py --config configs/config.yaml
python main_infer.py --config configs/config.yaml
```

---

## Run 8 — Neural Network: TruncatedSVD + MLP
**Date**: 2026-03-16
**Kaggle public LB**: (pending — run main_train.py then main_infer.py)
**Target**: Break linear ceiling (CV > 0.9328), reduce total DS+Grok errors below 30

### Why a neural network?
All linear models in Runs 1-7 are capped at CV ~0.9328. The core issue is that the
27 false-negative DeepSeek texts (short factual statements) are **linearly inseparable**
from Grok in TF-IDF space when they cover the same topics. Linear classifiers (LR, SVC,
Ridge) can only draw hyperplane boundaries in the feature space — they cannot learn
INTERACTIONS like `(short text) AND (no hedging words) AND (specific char n-gram pattern)
→ DeepSeek`. A neural network with hidden layers can learn these non-linear combinations.

### Architecture: TfidfMLPClassifier
```
Sparse TF-IDF input (~100k features)
  → TruncatedSVD(500)         # dense 500-dim representation (≈70% variance retained)
  → MLPClassifier(512, 256)   # 2 hidden layers, ReLU activation
  → 6-class softmax output
```
Key design choices:
- **TruncatedSVD first**: MLP with 100k sparse inputs would have 51M weight parameters
  for just the first layer (100k × 512). With 2400 training samples this massively overfits.
  SVD reduces to 500 dense dimensions (tractable: 500 × 512 = 256k params).
- **alpha=0.01 (L2 reg)**: Stronger than sklearn default 0.0001. Essential on small dataset.
- **early_stopping=True**: Monitors validation loss, stops if no improvement for 15 epochs.
- **sample_weight='balanced'**: MLPClassifier has no class_weight parameter, so balanced
  sample weights are computed and passed to fit() — equivalent to class_weight='balanced'.

### New models
1. **`mlp_svd`**: `TfidfMLPClassifier(n_svd=500, hidden=(512,256), alpha=0.01)`
   - Pure MLP approach — tests whether non-linear boundaries help DS/Grok

2. **`ensemble_mlp`**: `VotingClassifier(soft)` of `[mlp_svd, two_stage_top2]`
   - Orthogonal representations: MLP learns dense SVD features, top2 learns sparse TF-IDF
   - Soft averaging of orthogonal models typically beats either model individually

### Changes from Run 7
1. Added `TfidfMLPClassifier` class to `src/models.py`
   - Pure sklearn (no torch/transformers required)
   - TruncatedSVD for dimensionality reduction
   - MLPClassifier with Adam optimizer + L2 regularization + early stopping
   - compute_sample_weight for class imbalance handling
2. Added `mlp_svd` model to factory
3. Added `ensemble_mlp` model to factory (mlp_svd + two_stage_top2 soft vote)
4. Updated `AVAILABLE_MODELS` and module docstring

### How to run
```
cd d:\hachaton\text-authorship-detection
.env\Scripts\activate
python main_train.py --config configs/config.yaml
python main_infer.py --config configs/config.yaml
```

### Run 8 actual results (OOF CV)
| Model         | CV Macro F1 | OOF F1  | DS→Grok | Grok→DS | DS recall | DS F1 |
|---------------|-------------|---------|---------|---------|-----------|-------|
| ensemble_mlp  | **0.9350**  | 0.9360  | 22      | 12      | 0.69      | 0.75  |
| mlp_svd       | 0.9340      | 0.9354  | 25      | 9       | 0.65      | 0.74  |
| two_stage_top2| 0.9328      | 0.9339  | 21      | 14      | 0.70      | 0.74  |
| ensemble_soft | 0.9272      | 0.9284  | 28      | 10      | 0.61      | 0.70  |

**Kaggle LB result: 0.90978** (down from 0.91089 — MLP overfit to CV distribution).

### Run 8 diagnosis
The MLP DID break the linear CV ceiling (0.9328 → 0.9350), confirming that non-linear
decision boundaries are beneficial. BUT the Kaggle LB dropped slightly (0.91089 → 0.90978).
This means the MLP is learning patterns specific to the training distribution that do not
generalise to the test set. Root causes:
1. **Probability miscalibration**: MLP softmax outputs are often overconfident. The pair
   threshold and global thresholds optimised on these OOF probabilities may mis-fire on test.
2. **SVD instability**: TruncatedSVD fitted inside each CV fold can learn slightly different
   subspaces. The final model's SVD (fitted on all 2400 samples) represents all training
   vocabulary, while fold SVDs miss ~20% of terms. The test set may have features not in
   the fold SVD subspace.
3. **Randomness**: With only 600 test samples, a 5-label difference changes LB by ~0.008.

Key observation: `mlp_svd` (pure MLP) has only 9 Grok→DS errors — much better than
`two_stage_top2` (14). But `mlp_svd` has 25 DS→Grok vs 21 for top2. The two models are
genuinely complementary in where they make errors.

### What to watch in results
- If `mlp_svd` CV > 0.9328 → non-linear boundaries are helping
- If DS recall improves from 0.70 → neural features are learning the DS/Grok interaction
- If `ensemble_mlp` > both individually → orthogonal ensemble working as expected
- Overfit gap: MLP with alpha=0.01 should have train_f1 ≈ 0.96-0.98 (not 1.0)
  If train_f1 = 1.0 → increase alpha (try 0.05 or 0.1)

---

## Run 9 — Stabilise MLP + Calibration + Weighted Ensemble
**Date**: 2026-03-16
**Kaggle public LB**: (not yet submitted — per-model submissions generated)
**Best CV model**: `ensemble_mlp` — OOF 0.9360

### Run 8 diagnosis → Run 9 fixes
Run 8 showed MLP improves CV but hurt LB. Three targeted fixes:
1. **Calibrate MLP probabilities** (GPT fix): raw MLP softmax is overconfident.
   CalibratedClassifierCV (isotonic) reshapes probabilities to match true frequencies.
   This should make the pair threshold and global thresholds work correctly on test data.
2. **Mild DS boost** (GPT fix): instead of global class weight boost, apply 1.5x sample
   weight specifically to DeepSeek in the MLP. More targeted than the 2x global boost
   that created Grok→DS errors.
3. **Weighted ensemble** (GPT + own): give MLP 2x vote vs two_stage_top2 in the ensemble.
   MLP has lower overfit gap (0.059 vs 0.065) suggesting it generalises better.

### Changes from Run 8
1. **GPT fix**: `ensemble_mlp` now reads two-stage settings from config (was hardcoded)
2. **GPT fix**: `mlp_svd` and `ensemble_mlp` now read ALL MLP params from `config.mlp`
3. **`mlp_svd_ds_boost`** (new): TfidfMLPClassifier with DeepSeek sample weight ×1.5
   - Mild boost: forces MLP to pay attention to DS errors without over-correcting
4. **`mlp_svd_calibrated`** (new GPT): CalibratedClassifierCV(mlp_svd, cv=3, isotonic)
   - Fixes overconfident MLP probabilities → better threshold calibration on test set
5. **`ensemble_mlp_weighted`** (new GPT+own): VotingClassifier weights=[2,1] (mlp:top2)
   - MLP has lower overfit gap → giving it 2x vote weight should improve test generalisation
6. **Better verbose**: fold progress bars, running best-model tracker, per-model summaries
7. **Per-model submissions**: each model folder gets its own submission CSV
8. **Added `mlp.verbose` to config**: set `true` to see MLP loss per iteration

### How to run
```
cd d:\hachaton\text-authorship-detection
.env\Scripts\activate
python main_train.py --config configs/config.yaml
python main_infer.py --config configs/config.yaml
```
Submit `artifacts/submissions/submission_latest.csv` for best model.
Per-model: submit each `artifacts/experiments/<run>/*/submission_*.csv` to compare.

### Run 9 Actual Results
| Model                   | CV F1  | OOF F1 | DS→Grok | Grok→DS |
|-------------------------|--------|--------|---------|---------|
| ensemble_mlp            | 0.9350 | 0.9360 | 22      | 12      |
| mlp_svd                 | 0.9340 | 0.9354 | 25      | 9       |
| ensemble_mlp_weighted   | 0.9329 | 0.9342 | —       | —       |
| two_stage_top2          | 0.9328 | 0.9339 | 21      | 14      |
| mlp_svd_calibrated      | 0.9184 | 0.9193 | —       | —       | ← FAILED
| mlp_svd_ds_boost        | 0.9172 | 0.9155 | 15      | 28      | ← FAILED

**Diagnosis:**
- `mlp_svd_calibrated` FAILED: isotonic regression needs 500+ samples per class; DS=80.
  Calibration noise dominated the signal. CV dropped -0.017 vs uncalibrated.
- `mlp_svd_ds_boost` FAILED: DS recall improved (62/80) but Grok→DS exploded to 28.
  Total DS+Grok confusion = 43 (worse than baseline 34).
- `ensemble_mlp_weighted` provided no gain — MLP already dominates at equal weight.
- Error analysis: DS→Grok and Grok→DS confused texts are identical length (~191 chars,
  ~27 words). Semantic deadlock — DS/Grok both produce short encyclopedic factual sentences.

---

## Run 10 — LightGBM + Stacking Meta-Learner
**Date**: 2026-03-16
**Kaggle public LB**: (pending)
**Target**: Exceed CV 0.9360 (ensemble_mlp) via gradient boosting + stacking

### Run 9 diagnosis → Run 10 approach
Three problems to address:
1. **Linear ceiling**: LR/SVC all cap at 0.9328. MLP broke it but with calibration issues.
   → Try LightGBM: gradient boosting over SVD space, different inductive bias from MLP.
2. **Calibration**: MLP isotonic calibration needs more data than available (80 DS samples).
   → Fix with sigmoid calibration (Platt scaling — only 2 params per class).
3. **Ensemble diversity plateau**: soft vote of same-type models gives diminishing returns.
   → Stacking meta-learner: learns WHEN each model is right, not just averages probabilities.

### Changes from Run 9
1. **`lgbm_svd`** (new): `TruncatedSVD(300)` + `LGBMClassifier` with balanced sample weights.
   Gradient boosting handles class imbalance natively; tree splits capture interactions
   that both LR and MLP may miss; well-calibrated probabilities out-of-the-box.
2. **`ensemble_lgbm`** (new): soft vote `lgbm_svd` + `two_stage_top2`.
   Analogous to `ensemble_mlp`, tests whether LGBM complements the linear two-stage model.
3. **`mlp_svd_calibrated_sigmoid`** (new): `CalibratedClassifierCV(mlp_svd, cv=3, sigmoid)`.
   Sigmoid needs only 2 params per class; data-efficient unlike isotonic (which failed Run 9).
4. **`stacking_lgbm`** (new): `StackingClassifier([two_stage_top2, mlp_svd, lgbm_svd] → LR)`.
   Meta-LR learns which base model's predictions to trust per region of the feature space.
   18-dim meta-features (3 models × 6 class proba) → `LogisticRegression(C=0.1)`.
5. **2 new stylometric features** (43→45 total):
   - `starts_with_the`: 1.0 if text begins with "The " — Grok→DS confused texts pattern.
   - `clause_per_sent`: commas per sentence — proxy for compound sentence complexity.
6. **LightGBM installed**: `lightgbm==4.6.0`, `xgboost==3.2.0`.
7. **`lgbm:` config section** added (n_svd_components=300, n_estimators=300, etc).

### How to run
```
cd d:\hachaton\text-authorship-detection
.env\Scripts\activate
python main_train.py --config configs/config.yaml
python main_infer.py --config configs/config.yaml
```
Submit `artifacts/submissions/submission_latest.csv` for best model.
Per-model: submit each `artifacts/experiments/<run>/*/submission_*.csv` to compare.
