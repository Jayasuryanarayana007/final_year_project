# GNBR Drug Repurposing Project — Workflow

## Overview

This project uses **Knowledge Graph Embeddings** powered by **Biomedical Language Models** to predict drug repurposing candidates for **rare diseases**. The core idea: frame drug repurposing as a **triple classification** task on a knowledge graph mined from PubMed literature.

---

## Phase 1: Data Acquisition

**Goal:** Obtain the raw GNBR (Global Network of Biomedical Relationships) data.

GNBR was mined from ~16 million PubMed abstracts using NLP tools (PubTator for NER, CoreNLP for dependency parsing). It contains triples of the form `(entity_one, relation, entity_two)` across 4 categories:

| Source File | Relationship Type | Size |
|---|---|---|
| `formatted/gnbr.chemical-gene.tsv` | Drug ↔ Gene interactions | 471 MB |
| `formatted/gnbr.gene-gene.tsv` | Gene ↔ Gene interactions | 1.3 GB |
| `formatted/gnbr.chemical-disease.tsv` | Drug ↔ Disease interactions | 1.3 GB |
| `formatted/gnbr.gene-disease.tsv` | Gene ↔ Disease interactions | 1.1 GB |

Each row contains: `relation | score | pubmed_id | type_one | id_one | name_one | type_two | id_two | name_two | sentence`

**Status:** ✅ Done — raw files present in `formatted/`

---

## Phase 2: Rare Disease Identification & Filtering

**Goal:** Identify which diseases in GNBR are rare diseases, so the dataset can be focused on them.

### Step 2a — Extract Rare Disease MeSH IDs from Orphanet

**Script:** `filtering/rare_mesh.py`

- **Input:** Orphanet's XML catalog (`en_product1.xml`) — the official international registry of rare diseases
- **Process:** Parses the XML to extract MeSH (Medical Subject Heading) IDs for each rare disease
- **Output:** `filtering/ids.txt` — **1,798 MeSH IDs** of rare diseases

### Step 2b — Match MeSH IDs to Named Diseases via CTD

**Script:** `filtering/diseases.py` (first half)

- **Input:** `filtering/ids.txt` + `filtering/CTD_diseases.csv` (Comparative Toxicogenomics Database — contains ~12K diseases with their MeSH IDs)
- **Process:** Filters CTD diseases to only those whose MeSH ID is in `ids.txt`
- **Output:** `filtering/filtered_diseases.txt` — CTD rare diseases with clean names

### Step 2c — Fuzzy-Match CTD Names to GNBR Entity Names

**Script:** `filtering/diseases.py` (second half)

- **Problem:** GNBR uses inconsistent naming (e.g. `acute_myeloid_leukemia` vs `myeloid leukemia, acute`)
- **Process:** Uses `difflib.get_close_matches()` (Ratcliff-Obershelp algorithm, cutoff=0.3) to fuzzy-match each CTD rare disease name to the closest GNBR entity name
- **Output:** `filtering/disease_mapping.tsv` — **1,279 rare diseases** successfully mapped

Example mappings:
```
diffuse alopecia                        → diffuse_alopecia
severe combined immunodeficiency...     → deficiency_of_adenosine_deaminase
xyy syndrome 47                         → XYY_syndrome
```

**Status:** ✅ Done — all filtering outputs exist

---

## Phase 3: Dataset Construction

**Goal:** Build the final train/val/test splits from GNBR, centered around rare diseases.

**Script:** `preprocess.py`

### Step 3a — Extract & Clean Triples

**Function:** `get_triples()`

- Reads the 4 raw GNBR files from `formatted/`
- Maps short relation codes to full text names (e.g. `A+` → `activates`, `T` → `treatment`, `Sa` → `side_effect`)
- Deduplicates triples
- Outputs both scored and unscored versions to `data/`

Full relation mapping (31 types):
```
A+ → activates       A- → blocks          B → binds
E+ → increases expr  E- → decreases expr  E → affects expr
N  → inhibits        O  → transport       K → metabolism
Z  → enzyme activity T  → treatment       C → inhibits cell growth
Sa → side effect     Pr → suppresses      Pa → alleviates
J  → role in disease Mp → progression     U  → causal mutation
Ud → mutations       D  → drug target     Te → therapeutic effect
Y  → polymorphisms   G  → promotes prog.  Md → diagnostic biomarker
X  → overexpression  L  → improper reg.   W  → enhances
V+ → activates       I  → signals         H  → same complex
Rg → regulates       Q  → production
```

### Step 3b — BFS Filter Around Rare Diseases

**Function:** `filter_triples(max_edges=15, degrees=2)`

Builds the dataset using **Breadth-First Search** starting from the 1,279 rare diseases:

**Iteration 1 (depth=0):**
- Start from the 1,279 rare diseases as seed nodes
- Find all triples in `chemical-disease` and `gene-disease` where `name_two` is a rare disease
- Collect all drugs and genes directly connected to rare diseases
- Checkpoint: `data/checkpoints/filtered_triples_1.tsv`, `filtered_entities_1.tsv`

**Iteration 2 (depth=1):**
- For each entity found in iteration 1, search **all 4 GNBR tables** for their connections
- Cap at **15 edges** per entity per direction (randomly sampled if more). This keeps the dataset computationally feasible
- Checkpoint: `data/checkpoints/filtered_triples_2.tsv`, `filtered_entities_2.tsv`

**Final dataset:** 166,859 triples | 25,411 entities | 31 relations

### Step 3c — Train/Val/Test Split

**Function:** `split(prop=(0.1, 0.2, 0.7))`

Key design: **only `(drug, treatment, rare_disease)` triples are held out** for validation and testing, since the goal is to predict treatments for rare diseases.

```python
# Only treatment triples targeting rare diseases are split
to_split = triples[(relation == 'treatment') & (name_two in rare_diseases)]

# Everything else (all other relation types) goes to training
train = everything_else + 10% of to_split
```

| Split | Content | Positive Triples | With Negatives |
|---|---|---|---|
| `train.tsv` | All relation types + 10% treatment | 165,882 | 331,764 |
| `dev.tsv` | Only (drug, treatment, rare_disease) | 217 | 434 |
| `test.tsv` | Only (drug, treatment, rare_disease) | 760 | 1,520 |

Also generates:
- `data/entity2text.txt` — entity ID → clean text name (25,411 entries)
- `data/relation2text.txt` — relation ID → text description (31 entries)

**Status:** ✅ Done — all data files present in `data/`

---

## Phase 4: Pre-train biobert-tiny

**Goal:** Create a smaller, faster variant of BioBERT for resource-constrained training.

> [!IMPORTANT]
> **We did NOT pre-train biobert-tiny ourselves.** The pre-training was done by the **original paper authors** (Yash Patil) on AWS/Sherlock infrastructure. We use their published pre-trained model downloaded from HuggingFace. The scripts below are part of the original repo but were run by the paper authors, not by us.

### Step 4a — Download PubMed Abstracts

**Script:** `pre_train_downloader.py`
- Connects to NCBI's FTP server (`ftp.ncbi.nlm.nih.gov`)
- Downloads and parses ~20M PubMed abstract XML files
- Extracts raw abstract text into individual files

### Step 4b — Preprocess & Tokenize

**Script:** `pre_train_preprocess.py`
- Trains a custom BPE tokenizer with vocab size 32K on the biomedical corpus
- Tokenizes all abstracts with truncation at 512 tokens
- Pushes processed dataset to HuggingFace Hub (`yashpatil/processed_bio_bert_tiny_dataset`)

### Step 4c — Train with Masked Language Modeling

**Script:** `pre_train_runner.py`
- Base architecture: `prajjwal1/bert-tiny` (2 layers, hidden_size=128, 4.4M params)
- Training: MLM with 15% mask probability, 10 epochs, batch_size=64, lr=1e-3
- Dataset: 1M train + 200K eval samples from the processed PubMed abstracts
- Published model: `yashpatil/biobert-tiny-model` on HuggingFace

### What We Have Locally

We use **3 pre-trained models** downloaded into `cached_model/`:

| Model | Directory | Size | Parameters | Origin |
|---|---|---|---|---|
| BERT-tiny | `cached_model/bert-tiny/` | 17 MB | 4.4M | HuggingFace (`prajjwal1/bert-tiny`) |
| biobert-tiny | `cached_model/biobert-tiny/` | 55 MB | 4.4M (re-trained) | Paper authors (`yashpatil/biobert-tiny-model`) |
| biobert-tiny-squared | `cached_model/biobert-tiny-squared/` | 18 MB | ~4.4M | Paper authors (variant) |

> [!NOTE]
> The full BioBERT (110M params, `dmis-lab/biobert-v1.1`) is NOT cached locally. The `main.py` config points to `./cached_model/biobert/` but that directory does not exist. This means the full BioBERT experiments from the paper were run on different infrastructure (AWS/Sherlock) and are not reproducible from this local setup.

**Status:** ⚠️ Scripts exist but were NOT run by us — we use the pre-trained models downloaded from HuggingFace

---

## Phase 5: Training (LMKE for Triple Classification)

**Goal:** Train the LMKE model to classify triples as valid (1) or invalid (0).

**Scripts:** `main.py` → `dataloader.py` → `model.py` → `trainer.py`

### Step 5a — Data Loading

**Script:** `dataloader.py`

- Reads `train.tsv`, `dev.tsv`, `test.tsv`
- Builds `ent2id` and `rel2id` mappings (entity/relation → integer ID)
- Loads entity and relation text descriptions from `entity2text.txt` and `relation2text.txt`
- Computes degree statistics for each entity (used in degree-aware scoring)
- `DataSampler` generates negative samples by randomly corrupting head or tail of each positive triple

### Step 5b — Model Architecture

**Script:** `model.py` — the `LMKE` class

```
Input: (head_entity, relation, tail_entity)
           ↓
Tokenize:  "[CLS] head [desc_h] relation [desc_r] tail [desc_t] [SEP]"
           ↓
Replace [CLS] positions with learned entity/relation embeddings
           ↓
Pass through BERT (frozen or fine-tuned)
           ↓
Extract embeddings at head, relation, tail positions
           ↓
[TC mode] → Classifier(CLS_embedding) → 2-class prediction
[LP mode] → Entity/Relation classifier → rank all candidates
```

Key components:
- 3 copies of the language model (given, target, classification)
- Learned entity embeddings (`n_ent × hidden_size`)
- Learned relation embeddings (`n_rel × hidden_size`)
- TransE-style embeddings for structural scoring
- Binary classifier for triple classification
- Contrastive matching with degree-aware similarity

### Step 5c — Training Loop

**Script:** `trainer.py`

```
For each epoch:
    ├── Create DataSampler (shuffles data, generates negatives)
    ├── For each batch:
    │     ├── Tokenize triples
    │     ├── Forward pass through LMKE
    │     ├── Compute CrossEntropyLoss (TC) or BCELoss (LP)
    │     └── Backward pass + optimizer step
    ├── Evaluate on validation set
    └── Save checkpoint if validation metric improved
            → params/tc/{identifier}-epc_{epoch}_metric_{metric}.pt
```

### Training Command

```bash
python main.py \
    --task TC \
    --plm biobert \
    --batch_size 64 \
    --epoch 50 \
    --load_descriptions
```

Key flags:
- `--task TC` — triple classification (vs `LP` for link prediction)
- `--plm biobert` — which language model backbone to use
- `--uncertainty` — use support scores as soft labels
- `--contrastive` — use contrastive learning mode
- `--use_structure` — add TransE structural scoring

**Status:** ✅ Done — trained on Kaggle (GPU), checkpoints saved in `params/tc/`

---

## Phase 6: Evaluation

**Goal:** Evaluate the trained model on the test set and compute metrics.

### Step 6a — Generate Predictions

```bash
python main.py \
    --task TC \
    --triple_classification \
    --plm biobert \
    --load_epoch <best_epoch>
```

This runs inference on the test set and saves predictions to `predictions/{identifier}-predictions.csv`.

### Step 6b — Compute Metrics

**Script:** `evaluate.py`

```bash
python evaluate.py predictions/<filename>.csv
```

Computes:
- **Accuracy** — overall correct predictions
- **Precision** — of predicted positives, how many are correct
- **Recall** — of actual positives, how many were found
- **F1 Score** — harmonic mean of precision and recall
- **Confusion Matrix** — saved as PNG heatmap

### Results (from the research paper)

| Method | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| DistMult (baseline) | 0.915 | 0.921 | 0.928 | 0.925 |
| TransE (baseline) | 0.921 | 0.914 | 0.935 | 0.925 |
| LMKE + BERT-tiny | 0.937 | 0.906 | 0.975 | 0.939 |
| LMKE + biobert-tiny | 0.938 | 0.921 | 0.959 | 0.939 |
| LMKE + TinyBioBERT | 0.940 | 0.917 | 0.968 | 0.942 |
| **LMKE + BioBERT** | **0.947** | **0.921** | **0.979** | **0.949** |

**Status:** ✅ Done — evaluation pipeline functional

---

## Script Execution Order (Quick Reference)

| Step | Script | Command |
|---|---|---|
| 1 | `filtering/rare_mesh.py` | `python filtering/rare_mesh.py` |
| 2 | `filtering/diseases.py` | `python filtering/diseases.py` |
| 3 | `preprocess.py` | `python preprocess.py --max_edges 15 --max_depth 2` |
| 4 | *(Not run by us)* | Pre-trained models downloaded from HuggingFace |
| 5 | `main.py` | `python main.py --task TC --plm biobert --epoch 50` |
| 6a | `main.py` | `python main.py --task TC --triple_classification --load_epoch <N>` |
| 6b | `evaluate.py` | `python evaluate.py predictions/<file>.csv` |

---

## Visual Pipeline Summary

```
 ┌─────────────────────────────────────────────────────────────────┐
 │  PHASE 1: Data Acquisition                                     │
 │  GNBR raw data (4 files, ~4GB total from PubMed)               │
 └───────────────────────────┬─────────────────────────────────────┘
                             │
 ┌───────────────────────────▼─────────────────────────────────────┐
 │  PHASE 2: Rare Disease Filtering                                │
 │  Orphanet → 1,798 MeSH IDs                                     │
 │  CTD matching → filtered_diseases.txt                           │
 │  Fuzzy match to GNBR → 1,279 rare diseases (disease_mapping)   │
 └───────────────────────────┬─────────────────────────────────────┘
                             │
 ┌───────────────────────────▼─────────────────────────────────────┐
 │  PHASE 3: Dataset Construction                                  │
 │  BFS from rare diseases (depth=2, max_edges=15)                 │
 │  → 166,859 triples | 25,411 entities | 31 relations             │
 │  Split: train (165K) + val (434) + test (1,520)                 │
 └───────────────────────────┬─────────────────────────────────────┘
                             │
 ┌───────────────────────────▼─────────────────────────────────────┐
 │  PHASE 4: Pre-train biobert-tiny  ⚠️  NOT done by us           │
 │  6M PubMed abstracts → BPE tokenizer → MLM pre-training        │
 │  We downloaded the pre-trained model from HuggingFace           │
 └───────────────────────────┬─────────────────────────────────────┘
                             │
 ┌───────────────────────────▼─────────────────────────────────────┐
 │  PHASE 5: Training                                              │
 │  LMKE model + BioBERT backbone                                  │
 │  Triple Classification on rare disease KG                       │
 │  Trained on Kaggle GPU, checkpoints in params/tc/               │
 └───────────────────────────┬─────────────────────────────────────┘
                             │
 ┌───────────────────────────▼─────────────────────────────────────┐
 │  PHASE 6: Evaluation                                            │
 │  Accuracy: 0.947 | F1: 0.949 (BioBERT, best)                   │
 │  Confusion matrix + classification report                       │
 └─────────────────────────────────────────────────────────────────┘
```
