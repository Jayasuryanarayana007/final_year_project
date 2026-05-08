# Novelty Tracker — GNBR Drug Repurposing Project

> This document tracks all novelty enhancements made to the GNBR Drug Repurposing project.
> Each entry records **what was changed**, **why it's novel**, **which files were modified**, and **results** (once evaluated).

---

## Baseline Performance (Before Any Novelty)

These are the results from the original project — no novelty applied.

| Model | Accuracy | Precision | Recall | F1 | Config |
|---|---|---|---|---|---|
| **biobert-tiny (our run)** | **0.9226** | 0.8852 | 0.9712 | 0.9262 | batch=64, desc=True, uncertainty=False |
| BioBERT (paper) | 0.947 | 0.921 | 0.979 | 0.949 | batch=64, desc=True |
| TinyBioBERT (paper) | 0.940 | 0.917 | 0.968 | 0.942 | batch=64, desc=True |
| TransE baseline (paper) | 0.921 | 0.914 | 0.935 | 0.925 | — |
| DistMult baseline (paper) | 0.915 | 0.921 | 0.928 | 0.925 | — |

**Baseline command:**
```bash
python main.py --task TC --plm biobert-tiny --batch_size 64 --epoch 50 --load_descriptions
```

**Baseline confusion matrix (4,368 test triples):**
```
              Predicted 0    Predicted 1
Actual 0        1909            275
Actual 1          63           2121
```

---

## Novelty #2: Drug Candidate Ranking (Link Prediction for Drug Repurposing)

**Date:** 2026-05-08

### What Is It?

A standalone drug candidate ranking pipeline (`rank_drug_candidates.py`) that repurposes the trained Triple Classification (TC) model to rank unknown drug candidates for any given disease.

### Why Is It Novel?

1. **Stated Future Work**: The original LMKE paper explicitly stated that *"Important future work should focus on link prediction to enable models to suggest top drug candidates for certain diseases."*
2. **Repurposing the TC Model**: Using a trained triple classifier as a drug ranker is a creative application not explored in the original paper. We can reuse the `params/tc/` checkpoints directly without needing to retrain a massive Link Prediction model.
3. **Actionable Clinical Output**: Instead of just outputting `accuracy`, the system now outputs actionable, ranked lists of novel drug candidates. It also automatically splits "known treatments" (from the knowledge graph) from "novel predictions".

### Files Modified

| File | Change | Lines |
|---|---|---|
| `rank_drug_candidates.py` | [NEW] Standalone drug ranking script | +400 lines |

### How To Run

List all available diseases:
```bash
python rank_drug_candidates.py --list_diseases
```

Rank drugs for a specific disease (e.g., malaria):
```bash
python rank_drug_candidates.py --disease malaria --top_k 20 --show_known
```

### Results

Testing on `malaria` yielded 6,076 candidates scored in 15.9s on CPU.
Mean score for known treatments was significantly higher (0.88) compared to novel predictions (0.57).

> **Status:** ✅ Implemented, 💾 Results saved in `predictions/`

---

## Novelty #3: *(Planned — Not Yet Implemented)*

*Reserved for next novelty enhancement.*

---

## Summary Table

| # | Novelty | Status | Accuracy Δ | Commit |
|---|---|---|---|---|
| 1 | Degree-Aware Ensemble (LM + TransE) | ✅ Implemented, ⏳ Results pending | — | `f467b77` |
| 2 | Drug Candidate Ranking Pipeline | ✅ Implemented | — | pending |
| 3 | — | — | — | — |

---

## How To Update This File

After each novelty implementation:
1. Add a new `## Novelty #N` section with the template above
2. Fill in the **Results** table after training completes
3. Update the **Summary Table** at the bottom
4. Commit this file along with the code changes
