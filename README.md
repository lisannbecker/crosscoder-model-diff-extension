# Hidden Signatures: Using Crosscoders to Compare Transformer Features Between Model Pairs

**Lisann Becker, Morris de Haan, Krijn Dignum, Aaron Kuin, Gjalt Hoekstra**

📄 [Read the full paper](./paper.pdf)

---

## Abstract

Understanding how knowledge is preserved during model transformation remains challenging. We introduce novel metrics for quantifying feature sharing between transformer models using crosscoders—sparse autoencoders that learn shared dictionaries across multiple models. We extend crosscoders to handle different residual-stream dimensions and develop two metrics: "sharedness" based on weighted entropy of decoder norms, and feature alignment using canonical correlation analysis. Evaluating four model relationship types (identical, instruction-tuned, different scales, and distilled pairs), we find early layers consistently exhibit high feature sharing while middle layers diverge based on the transformation method. These findings reveal how different techniques preserve representations and establish crosscoders as a tool for mechanistic model analysis.

---

## Setup and Usage

### 1. Prepare Data

Each model requires its own tokenized dataset:

```bash
# Example for GPT-2
python _load_dataset_gpt2_1m_tokens.py
````

Tokenizers for Pythia and Gemma are also available in `*_load_dataset_*.py` scripts. These generate `.pt` files of 1M tokens.

---

### 2. Train Crosscoder

To train a crosscoder between two models (e.g., GPT-2 and DistilGPT2), run:

```bash
python train.py --method distillation --layer middle
```

Supported methods:

* `default` / `finetuning`: for identical or instruction-tuned models (e.g., Gemma)
* `scale`: for models of different sizes (e.g., Pythia-160M vs 70M)
* `distillation`: for teacher-student comparisons (e.g., GPT-2 vs DistilGPT2)
* `same`: for sanity checks with identical models

Supported layers: `first`, `middle`, `last`

This creates trained checkpoints and logs training metrics (e.g., sparsity, reconstruction loss) via Weights & Biases.

---

### 3. Analyze Feature Similarity

Once training is complete, you can analyze feature sharing and alignment.

#### Run prebuilt analysis (e.g. for Gemma 2B vs IT):

```python
python analysis.py
```

This loads a trained `CrossCoder`, calculates:

* Relative decoder norm strength
* Cosine similarity of decoder vectors (for shared features)
* Plots histograms using Plotly

You can modify the visualization titles and filters inside `analysis.py`.

#### Compute strength and similarity metrics:

```bash
python feature_strengths.py \
  --config_dir ./checkpoints/gpt2-137m-88m \
  --version last \
  --modelA gpt2 \
  --modelB distilgpt2 \
  --is_multiscale
```

This script:

* Loads both models
* Encodes 1M token activations
* Computes average latent activation strength
* Calculates similarity via Canonical Correlation Analysis (CCA)
* Saves a `.pt` file with `data_strength`, `model_strength`, and `sim_metric`

---

## Experimental Jacobian Distillation Pipeline

This repository also includes an early-stage implementation of a **Jacobian matching distillation pipeline** for GPT-2.
This setup was designed to compare the latent effects of KL-divergence vs Jacobian-based distillation in an aligned setting where:

* Both student and teacher share architecture (e.g., GPT-2 vs DistilGPT2)
* The goal is to quantify differences in internal features using the same crosscoder evaluation tools

Although memory constraints prevented a complete analysis before paper submission, the infrastructure is in place for future research:

* All necessary configuration, training logic, and model hooks are implemented
* Open-source models distilled using Jacobian matching are currently unavailable—this repo can serve as a starting point

---
