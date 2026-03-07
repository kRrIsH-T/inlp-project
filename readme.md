# 🧠 Knowledge Unlearning in LLMs using Sparse Autoencoders

[![Course: INLP](https://img.shields.io/badge/Course-Introduction%20to%20NLP-blue.svg)](https://github.com/Gopalkataria/INLP_PROJECT)
[![Project: Terminal Touchers](https://img.shields.io/badge/Team-Terminal%20Touchers-orange.svg)](https://github.com/Gopalkataria/INLP_PROJECT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

## 📖 Overview

This project explores the fascinating intersection of **Mechanistic Interpretability** and **Model Editing**. Specifically, we investigate how to selectively "forget" or unlearn specific domain knowledge—in this case, the **Harry Potter** universe—from a pre-trained language model (**GPT-2 Medium**). 

Instead of traditional fine-tuning or gradient-based editing, we leverage **Sparse Autoencoders (SAE)** to identify and ablate high-level interpretable features associated with the target knowledge. This approach aims for a more surgical and interpretable form of knowledge removal, minimizing collateral damage (entanglement) to the model's general reasoning capabilities.

---

## 🛠️ Project Architecture

### 1. **Data Preprocessing**
*   **Target Corpus**: Preprocessing the Harry Potter books to extract high-density knowledge tokens.
*   **Neutral Corpus**: Utilizing a mix of **WikiText-2** and **TinyStories** (fiction) as a baseline. This broadening prevents the SAE from misidentifying general high-fantasy terms (e.g., "wand") as Harry Potter-specific.

### 2. **Sparse Autoencoders (SAE)**
*   Implementation of **Top-K SAEs** on the residual stream of GPT-2 Medium (Layer 12).
*   Training SAEs to reconstruct activations in a sparse, interpretable basis.

### 3. **Interpretability & Analysis**
*   **Difference-in-Means**: Method to identify features that activate significantly more on the target corpus than on the neutral corpus.
*   **Feature Validation**: Manual and automated inspection of selected "Harry Potter" features.

### 4. **Intervention & Evaluation**
*   **Ablation Hooks**: Using `TransformerLens` to zero out specific SAE features during the forward pass.
*   **Knowledge Metrics**: Evaluating completion accuracy on Harry Potter-specific prompts (inspired by Eldan & Russinovich, 2023).
*   **General Capability**: Measuring Perplexity and MMLU performance post-ablation to ensure the model hasn't "collapsed."

---

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/Gopalkataria/INLP_PROJECT.git
cd INLP_PROJECT

# Install dependencies
pip install torch transformer-lens datasets tqdm einops transformers
```

### Usage Workflow

1.  **Preprocessing**:
    ```bash
    python src/data/preprocess.py
    ```
2.  **Train SAE** (for GPT-2 Medium):
    ```bash
    python src/sae/train.py --model_name gpt2-medium --layer 12 --k 32 --expansion_factor 16 --epochs 20
    ```
3.  **Analyze & Filter Features**:
    Extracts features that are 20x+ more active in HP than in the Neutral Corpus:
    ```bash
    python src/analysis/diff_means.py --model_name gpt2-medium --layer 12 --num_features 100 --min_ratio 50.0 --sort_by ratio
    ```
4.  **Consolidated Statistical Evaluation**:
    Run the full suite with **Top-P Sampling** and **Negative Scaling** for optimal unlearning depth and stability:
    ```bash
    python src/eval/unified_evaluate.py --layer 12 --num_features 100 --ablation_scale -3.0 --freq_penalty 1.0 --top_p 0.9
    ```
5.  **Qualitative LLM-as-a-Judge Evaluation (Manual/Hugging Face)**:
    Requires setting `HF_TOKEN` in a local `.env` file for automated classification using free serverless models (e.g., `Qwen/Qwen2.5-7B-Instruct`).
    ```bash
    python src/eval/evaluate_llm_judge.py --model Qwen/Qwen2.5-7B-Instruct
    ```

---

## 📊 Results Summary (GPT-2 Medium)

By ablating the top 100 Harry Potter-specific features identified in Layer 12, we achieved highly selective unlearning across 300 sampled completion pairs:

| Metric | Baseline (Top-P) | Ablated ($-3.0$ Scale) | Impact |
| :--- | :--- | :--- | :--- |
| **HP Knowledge Recall** | **46.7%** (140/300) | **4.7%** (14/300) | **90.0% Reduction (Manual Judge)** |
| **HP Log-Probability** | $-3.6567$ | $-5.1248$ | **$-1.4681$ Shift (Massive Drop)** |
| **Magic Log-Probability** | $-3.0521$ | $-3.1354$ | **Preserved ($-0.08$)** |
| **Fantasy Log-Probability**| $-6.4321$ | $-6.8215$ | **Minor Collateral ($-0.39$)** |
| **Real World Log-Prob** | $-3.9647$ | $-3.9692$ | **Unchanged (0.00)** |
| **Degeneration Rate** | **0.0%** | **0.0%** | **Perfectly Stable** |
| **WikiText-2 Perplexity** | 43.84 | 43.88 | **0.1% Degradation** |

*Qualitative observation*: When prompted with Harry Potter context, the ablated model often pivots to generic historical or geographical facts (e.g., mentioning "Neil Armstrong" or "The United Kingdom") or stays in the fantasy genre without utilizing specific HP lore.

---

## 📂 Repository Structure

```text
├── 📁 src/
│   ├── 📁 sae/           # SAE model architecture & training
│   ├── 📁 data/          # Tokenization and corpus handling
│   ├── 📁 analysis/      # Feature discovery (Specificity Ratio)
│   ├── 📁 intervention/  # Ablation hooks
│   └── 📁 eval/          # Evaluation tools (unified_evaluate.py, evaluate_llm_judge.py)
├── 📁 results/           # JSON reports and side-by-side completions
├── 📄 readme.md          # Project documentation
└── 📄 task.md            # Internal development log
```

---

## 👥 Team: Terminal Touchers

Developed as a Course Project for **Introduction to Natural Language Processing (INLP)** at **IIIT Hyderabad**.

---

## 📝 Acknowledgments

*   [Who's Harry Potter?](https://arxiv.org/abs/2310.02238) by Ronen Eldan and Mark Russinovich.
*   [TransformerLens](https://github.com/neelnanda-io/TransformerLens) toolkit.
*   Neel Nanda and the AI Safety community for SAE research.
