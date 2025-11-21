# LLM Preference Model on Chatbot Arena

- **Task:** Kaggle *LLM Classification Finetuning* – predict which of two LLM responses humans prefer (A wins / B wins / tie). :contentReference[oaicite:0]{index=0}  
- **Stack:** Python, pandas, numpy, scikit-learn, SciPy, (LightGBM, Sentence-BERT experiments). :contentReference[oaicite:1]{index=1}  
- **NLP techniques:** TF-IDF n-grams, hand-crafted statistical features (length, structure, tone), basic error & bias analysis. :contentReference[oaicite:2]{index=2}  
- **Result:** Validation log loss improved from **1.1063 → 1.0623**, accuracy from **38.0% → 44.6%**, with **758 fewer errors** vs the baseline TF-IDF model. :contentReference[oaicite:3]{index=3}  
- **Insight:** Longer answers and apologetic / refusal tones strongly correlate with preference labels and model decisions. :contentReference[oaicite:4]{index=4}  

---

## 1. Project Overview

This repository contains my experiments for the Kaggle **LLM Classification Finetuning** competition using human preference data from **Chatbot Arena**.  
Given a prompt and two candidate LLM responses (`response_a`, `response_b`), the goal is to predict which option humans prefer:

- class `0` – model A wins  
- class `1` – model B wins  
- class `2` – tie  

The focus is on building a **strong but lightweight classical baseline**, exploring **feature engineering** and **biases** in the data rather than training a huge LLM.
