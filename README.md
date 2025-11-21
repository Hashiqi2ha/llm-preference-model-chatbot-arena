# LLM Preference Model on Chatbot Arena

Preference modeling project based on the Kaggle **LLM Classification Finetuning** competition.  
Given a prompt and two candidate LLM responses, the goal is to predict which answer humans prefer.

- **Task:** 3-class classification – *A wins / B wins / tie*.
- **Data:** Chatbot Arena conversations with human preference labels.
- **Stack:** Python, pandas, numpy, scikit-learn, SciPy, LightGBM, Jupyter.
- **Focus:** Strong classical baselines, feature engineering, and bias analysis (length, tone, structure).
- **Outcome:** Improved validation log loss and accuracy vs. a plain TF-IDF baseline, with a clear understanding of what drives human preferences.

---

## 1. Project Overview

Large language models are often evaluated by human preference: given two answers, which one feels better?  
This project treats that as a supervised learning problem:

> **Input:** prompt + response A + response B  
> **Output:** which option humans chose (A / B / tie)

Instead of fine-tuning a huge LLM, the goal here is to:

1. Build **transparent classical models** that are easy to explain and reproduce.  
2. Understand **what features** (length, tone, structure, refusals, etc.) correlate with preference.  
3. Provide a clean, end-to-end example of an **LLM reward / preference model pipeline**.

---

## 2. Data & Preprocessing

**Source**

- Kaggle competition: *LLM Classification Finetuning*.
- Train data contains: prompt, two LLM responses, model IDs, and one-hot preference labels (`winner_model_a`, `winner_model_b`, `winner_tie`).

**Preprocessing steps**

- Convert JSON-style text fields into plain strings.
- Combine fields into a single modeling text, e.g.:

  > `[PROMPT] ... [A] ... [B] ...`

- Map the three one-hot label columns into a single integer label:
  - `0` – A wins
  - `1` – B wins
  - `2` – tie
- Stratified train/validation split to keep class balance.

Raw CSV files are not stored in this repo due to GitHub size limits.  
See `data/README.md` for instructions on how to download and place the Kaggle data.

---

## 3. Modeling Approach

The project is structured as several iterations, each adding more information to the model.

### 3.1 Baseline: TF-IDF + Logistic Regression

1. Build a **TF-IDF representation** of the combined text:
   - Unigrams + bigrams (`ngram_range = (1, 2)`).
   - Large vocabulary with max features capped for efficiency.
2. Train a multinomial **Logistic Regression** classifier.
3. Evaluate using:
   - **Log loss** (competition metric).
   - **Accuracy** and error counts.

This gives a strong, simple baseline to improve upon.

---

### 3.2 Feature Engineering & Tuning

To go beyond raw bag-of-words, I engineered a set of **structured features** computed for each response and their differences:

- **Length & word stats**
  - Character length, word counts, length ratios and differences.
- **Sentence & punctuation**
  - Approximate sentence counts, average sentence length.
  - Comma / period density as a proxy for writing style.
- **Structure**
  - Number of code blocks, bullet lists, numbered lists.
  - Newline counts (how “formatted” the answer looks).
- **Tone & attitude**
  - Presence of apologies (“sorry”, “apologize”…).
  - Refusal / limitation phrases (“cannot”, “unable”, “as an AI model…”)  
  - “Don’t know” / uncertainty wording.

These features are concatenated with the TF-IDF matrix using `scipy.sparse.hstack`, giving a hybrid representation:

> **text features (TF-IDF) + statistical features (length, structure, tone)**

On top of this, I tune the logistic model mainly through:

- Adjusting **regularization strength** (`C`).
- Increasing `max_iter` for stability.
- Monitoring validation log loss and overfitting.

**Result:**  
Compared with the baseline, the tuned model with engineered features:

- Lowers validation **log loss**.  
- Improves **accuracy**.  
- Reduces the number of misclassified validation samples.

(Exact numbers are documented in `LLM_Tuning.ipynb` / `LLM_Tuning.md`.)

---

### 3.3 Additional Experiments

These experiments are exploratory and not all are part of the final pipeline, but they show directions for improvement:

- **LightGBM on hybrid features**  
  - Train gradient-boosted trees on the TF-IDF + statistical feature space.  
  - Handle sparse matrices directly to avoid memory blow-ups.  
  - Compare log loss and feature importance vs logistic regression.

- **Sentence-BERT (future work)**  
  - Use `sentence-transformers` (e.g. `all-mpnet-base-v2`) to encode responses into dense embeddings.  
  - Explore combining SBERT embeddings with structured features as inputs to tree-based models.

---

## 4. Bias & Error Analysis

Beyond metrics, the project includes **qualitative analysis** to understand what the model is learning.

### 4.1 Length Bias

- Bucket samples by length difference between A and B:
  - A much longer / slightly longer / similar length / B slightly longer / B much longer.
- For each bucket, measure the empirical distribution of winners.

Observation:  
When one answer is **much longer**, that side is significantly more likely to be labeled as the winner.  
When lengths are similar, the probability of a **tie** increases sharply.  
This suggests a clear **length bias** in human preferences and in the model.

### 4.2 Tone & Refusal Patterns

By inspecting high-confidence misclassifications and feature importances, I see that:

- Apologetic / refusal phrases often hurt preference **unless** the other answer is clearly worse.  
- Answers that **directly address the task** and provide concrete steps are favored.  
- Very similar answers are often labeled as tie, which is hard for a classifier to distinguish.

These insights are useful if this model is later used as part of an RLHF / reward modeling pipeline.
