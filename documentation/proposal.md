# Evaluating Explanation Consistency in Zero-Shot LLM Toxicity Classification: Integrated Gradients vs. Attention

---

## Abstract

This project uses zero-shot toxicity classification on the Civil Comments dataset as a testbed to investigate whether different interpretability methods provide consistent and trustworthy explanations, particularly in the presence of ambiguity and fairness concerns. Rather than treating explanations as a given, we treat them as objects of evaluation.

We formulate toxicity detection as a prompt-based label selection task, scoring predictions via the log-probability difference between competing labels (*toxic* vs. *non-toxic*). We then compute two fundamentally different explanation signals — Integrated Gradients (IG), which estimates how much each token contributed to the final decision, and attention-based token importance, which reflects how information flows between tokens inside the model — and explicitly compare them at the token level.

The central question is not simply what the model attends to, but whether these two notions of importance tell the same story, and what it means when they do not. To assess fairness, we compute per-group classification metrics and disparity measures across demographic identity columns, including Statistical Parity Difference (SPD) and Equal Opportunity Difference (EOpp), and examine whether explanation disagreements are correlated with model uncertainty or fairness failures.

This end-to-end setup positions interpretability not as a visualization tool, but as a subject of rigorous evaluation, and highlights how explanation reliability itself must be interrogated in responsible LLM audits.

---

## 1. Methodology

### 1.1 Dataset and Task Definition

We use the Civil Comments dataset, which contains user-generated text and toxicity-related annotation fields. The core task is binary toxicity classification:

- **Positive class:** toxic
- **Negative class:** non-toxic

The `text` field is used as model input, while the toxicity target is binarized for evaluation. Identity-related columns are retained to support subgroup fairness analysis.

---

### 1.2 Zero-Shot LLM Classification

#### Prompt-Based Formulation

For each comment, we construct an instruction-style prompt asking the model to choose between two labels (e.g., *"toxic"* vs. *"non-toxic"*). This avoids supervised fine-tuning and demonstrates in-context task transfer.

#### Scoring Rule

For each comment $x$, we compute:

$$s(x) = \log p(\text{toxic} \mid x) - \log p(\text{non-toxic} \mid x)$$

where each label probability is computed autoregressively over its token sequence (teacher forcing). The prediction is obtained by sign:

$$\hat{y} = \begin{cases} 1 & \text{if } s(x) > 0 \\ 0 & \text{otherwise} \end{cases}$$

This creates a calibrated relative preference between competing labels rather than a single raw next-token score.

---

### 1.3 Interpretability: Two Explanation Signals

Rather than treating explanations as inherently valid, we compute two structurally different token-level signals and subject their agreement to analysis. The core premise is that Integrated Gradients and attention measure fundamentally different concepts: IG estimates causal contribution to the output, while attention reflects internal information routing. Divergence between them is treated as a meaningful signal, not a nuisance.

#### Integrated Gradients (IG)

We compute IG token attributions on the scoring function $s(x)$. Positive attribution indicates contribution toward *"toxic,"* and negative attribution indicates contribution toward *"non-toxic."*

**IG setup:**

| Parameter | Value |
|-----------|-------|
| Baseline embedding | Zero embedding |
| Path | Linear interpolation from baseline to actual input embeddings |
| Steps | Fixed number (e.g. 32) |
| Gradient computation | Through `inputs_embeds` to ensure attribution reflects prompt-token contributions |

Attributions are normalized per example and stored as token-level importance vectors.

#### Attention-Based Token Importance

We extract attention weights from one or more transformer layers and aggregate them into a per-token importance score. Unlike IG, attention does not directly measure output sensitivity; it reflects how the model routes information internally. Attention weights are normalized and stored in a format comparable to IG attributions.

#### Cross-Method Comparison

For each example in the comparison subset, we align the IG attribution vector and the attention importance vector over the shared token sequence and analyze:

- **Agreement:** tokens that are highly ranked by both methods
- **Divergence:** tokens emphasized by one method but not the other (e.g. attention highlights identity terms while IG highlights insults)
- **Divergence patterns:** whether disagreements are more frequent in borderline or ambiguous examples, and whether they correlate with fairness-relevant identity terms or elevated model uncertainty

A token-level agreement score (e.g. rank correlation or top-*k* overlap) is computed per example to support quantitative comparison alongside the qualitative heatmap analysis.

---

### 1.4 Fairness Evaluation

Fairness is evaluated across identity columns using group membership indicator $A \in \{0, 1\}$, where $A = 1$ indicates presence of identity signal and $A = 0$ its absence.

#### Per-Group Performance

For each identity and each subgroup ($A = 0$, $A = 1$), we compute:

- Accuracy
- F1
- True Positive Rate (TPR)
- False Positive Rate (FPR)
- Positive prediction rate

A minimum group-size filter is applied to avoid unstable small-sample estimates.

#### Disparity Metrics

**Statistical Parity Difference (SPD):**

$$\text{SPD} = P(\hat{Y} = 1 \mid A = 1) - P(\hat{Y} = 1 \mid A = 0)$$

**Equal Opportunity Difference (EOpp):**

$$\text{EOpp} = \text{TPR}_{A=1} - \text{TPR}_{A=0}$$

#### Worst-Case Summary

We report the following worst-case metrics:

- $\max |\text{SPD}|$ across identities
- $\max |\text{EOpp}|$ across identities
- Worst subgroup accuracy
- Worst subgroup F1

This highlights the most adverse group-level behaviour, rather than only average behaviour.

---

### 1.5 Experimental Workflow

The experimental pipeline proceeds as follows:

1. Load and normalize Civil Comments data.
2. Run zero-shot inference over a capped number of rows.
3. Save prediction outputs (`idx`, `pred`, `score`, log-prob components).
4. For a comparison subset (for efficiency):
   - Compute IG attributions and save token-level vectors.
   - Extract attention weights from target layer(s) and save token-level importance vectors.
   - Compute per-example agreement scores between the two signals.
   - Save side-by-side heatmaps for qualitative inspection.
5. Merge predictions with labels and identity attributes.
6. Compute per-group and per-identity fairness reports.
7. Analyze whether explanation disagreements correlate with borderline predictions, fairness-sensitive identity terms, or elevated model uncertainty.
8. Export:
   - per-group metrics CSV
   - per-identity disparity CSV
   - worst-case summary CSV
   - per-example explanation agreement scores CSV

---

### 1.6 Evaluation and Reporting Plan

#### Quantitative Outputs

- Overall classification metrics (accuracy, F1)
- Identity-wise disparity table (SPD, EOpp)
- Worst-case subgroup metrics
- Per-example explanation agreement scores (IG vs. attention)

#### Qualitative Outputs

Side-by-side IG and attention heatmaps for:

- a correct toxic example with high explanation agreement
- a false positive where the two methods diverge
- a borderline/ambiguous example to illustrate how disagreement relates to uncertainty

#### Interpretation Goals

- Determine whether model decisions rely on toxic lexical cues vs. identity terms, and whether the two explanation methods give the same answer.
- Identify whether attention highlights identity terms in cases where IG highlights insults, and examine what that divergence implies.
- Assess whether explanation disagreement is correlated with fairness disparities or model uncertainty.
- Identify identities with higher disparity risk.
- Discuss mitigation options (prompt redesign, threshold tuning, data balancing, post-hoc calibration), informed by explanation analysis.

---

### 1.7 Limitations

- Zero-shot prompting can be sensitive to wording and model choice.
- Identity columns may be sparse or imbalanced, affecting disparity stability.
- Toxicity labels may include annotation subjectivity.
- Attribution methods explain model sensitivity, not causal truth; neither IG nor attention constitutes a ground-truth explanation.
- Attention is not designed as an explanation mechanism and its interpretability as a token importance signal remains contested in the literature.
- Agreement between explanation methods does not guarantee correctness; both methods can be simultaneously misleading.
- The comparison subset used for explanation analysis may not be representative of the full data distribution.
