# coding=UTF-8
"""
Drug Candidate Ranking Pipeline (Novelty #5)
=============================================
Given a disease, ranks all drug entities in the GNBR knowledge graph
by their predicted treatment score using the trained TC model.

This repurposes the Triple Classification model as a drug ranker:
for each drug entity, it constructs a (drug, treatment, disease) triple,
runs it through the TC classifier, and ranks by P(valid).

Usage:
    python rank_drug_candidates.py --disease malaria --top_k 20
    python rank_drug_candidates.py --disease Fanconi_anemia --top_k 50 --show_known
    python rank_drug_candidates.py --list_diseases
"""

import argparse
import os
import sys
import time
import csv
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from dataloader import DataLoader
from model import LMKE


# Constants
TREATMENT_RELATION = 'treatment'
DEFAULT_CHECKPOINT = './params/tc/biobert-tiny-u=False-b=64-d=True-epc_49_metric_acc.pt'
DEFAULT_PLM = 'biobert-tiny'
DEFAULT_PLM_PATH = './cached_model/biobert-tiny/'
PREDICTIONS_DIR = './predictions/'


def load_model_and_data(checkpoint_path, plm_path, device):
    """Load the trained LMKE model and data loader."""
    print(f"[1/3] Loading tokenizer and PLM from: {plm_path}")
    lm_tokenizer = AutoTokenizer.from_pretrained(plm_path, do_basic_tokenize=False)
    lm_config = AutoConfig.from_pretrained(plm_path)
    lm_model = AutoModel.from_pretrained(plm_path, config=lm_config)

    print("[2/3] Loading dataset and building data loader...")
    in_paths = {
        'dataset': 'rare-disease',
        'train': './data/train.tsv',
        'valid': './data/dev.tsv',
        'test': './data/test.tsv',
        'text': ['./data/entity2text.txt', './data/relation2text.txt']
    }

    data_loader = DataLoader(
        in_paths, lm_tokenizer, batch_size=64, neg_rate=1,
        add_tokens=False, p_tuning=False, rdrop=False,
        model='bert', descriptions=True, uncertainty=False
    )

    lm_model.to(device)

    model = LMKE(
        lm_model,
        n_ent=len(data_loader.ent2id),
        n_rel=len(data_loader.rel2id),
        add_tokens=False,
        contrastive=False
    )

    print(f"[3/3] Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please ensure you have a trained TC model in params/tc/")
        sys.exit(1)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model.to(device)
    model.eval()

    print("[SUCCESS] Model loaded successfully!\n")
    return model, data_loader, lm_tokenizer


def identify_entity_types(data_loader):
    """
    Identify which entities are drugs and which are diseases based on
    their roles in 'treatment' triples in the knowledge graph.

    - Drugs: entities appearing as HEAD of treatment triples
    - Diseases: entities appearing as TAIL of treatment triples
    """
    drug_entities = set()
    disease_entities = set()
    treatment_triples = defaultdict(set)  # disease -> set of known drug treatments

    all_triples = data_loader.train_set + data_loader.valid_set + data_loader.test_set

    for h, r, t in all_triples:
        if r == TREATMENT_RELATION:
            drug_entities.add(h)
            disease_entities.add(t)
            treatment_triples[t].add(h)

    # Also look for 'therapeutic_effect' and 'alleviates' relations
    for h, r, t in all_triples:
        if r in ('therapeutic_effect', 'alleviates'):
            drug_entities.add(h)
            disease_entities.add(t)

    print(f"[INFO] Entity type analysis:")
    print(f"   Drug entities (appear as head in treatment): {len(drug_entities)}")
    print(f"   Disease entities (appear as tail in treatment): {len(disease_entities)}")
    print(f"   Treatment triples in KG: {sum(len(v) for v in treatment_triples.values())}")
    print()

    return drug_entities, disease_entities, treatment_triples


def list_diseases(data_loader):
    """List all disease entities in the knowledge graph."""
    _, disease_entities, treatment_triples = identify_entity_types(data_loader)

    print(f"\n{'='*70}")
    print(f"  All disease entities with known treatments ({len(treatment_triples)} diseases)")
    print(f"{'='*70}\n")

    # Sort by number of known treatments (descending)
    sorted_diseases = sorted(
        treatment_triples.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    print(f"{'#':<5} {'Disease':<45} {'Known Treatments':<15}")
    print(f"{'-'*65}")

    for i, (disease, drugs) in enumerate(sorted_diseases, 1):
        display_name = disease.replace('_', ' ')
        print(f"{i:<5} {display_name:<45} {len(drugs):<15}")

    print(f"\n[TIP] Use any disease name (with underscores) as --disease argument")
    print(f"   Example: python rank_drug_candidates.py --disease {sorted_diseases[0][0]}")


def rank_drugs_for_disease(model, data_loader, disease, drug_entities,
                            known_treatments, device, batch_size=32, top_k=20,
                            show_known=False):
    """
    Rank all drug entities by their predicted treatment score for a given disease.

    For each drug, constructs the triple (drug, treatment, disease),
    runs it through the TC classifier, and collects P(valid).
    """
    rel2id = data_loader.rel2id
    degrees = data_loader.statistics['degrees']

    if TREATMENT_RELATION not in rel2id:
        print(f"ERROR: '{TREATMENT_RELATION}' relation not found in KG")
        return []

    # Build candidate triples
    drug_list = sorted(drug_entities)
    candidate_triples = [(drug, TREATMENT_RELATION, disease) for drug in drug_list]

    print(f"[INFO] Scoring {len(candidate_triples)} drug candidates for disease: {disease.replace('_', ' ')}")
    print(f"   Known treatments in KG: {len(known_treatments)}")
    print()

    # Batch inference
    all_scores = []

    with torch.no_grad():
        for i in tqdm(range(0, len(candidate_triples), batch_size), desc="Ranking drugs"):
            batch_triples = candidate_triples[i:i + batch_size]
            triple_degrees = [[degrees.get(e, 0) for e in triple] for triple in batch_triples]

            inputs, positions = data_loader.batch_tokenize(batch_triples, 'triple_classification')
            inputs.to(device)

            preds = model(inputs, positions, 'triple_classification', triple_degrees)
            # preds shape: (batch_size, 2) - [P(invalid), P(valid)]
            probs = torch.softmax(preds, dim=1)
            valid_probs = probs[:, 1].cpu().numpy()

            all_scores.extend(valid_probs)

    # Build results
    results = []
    for j, drug in enumerate(drug_list):
        is_known = drug in known_treatments
        results.append({
            'drug': drug,
            'drug_display': drug.replace('_', ' '),
            'score': float(all_scores[j]),
            'known': is_known,
            'degree': degrees.get(drug, 0)
        })

    # Sort by score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)

    # Separate known and novel
    novel_results = [r for r in results if not r['known']]
    known_results = [r for r in results if r['known']]

    return results, novel_results, known_results


def print_results(disease, results, novel_results, known_results, top_k, show_known):
    """Print formatted ranking results to console."""
    disease_display = disease.replace('_', ' ')

    print(f"\n{'='*80}")
    print(f"  Drug Candidate Ranking for: {disease_display}")
    print(f"{'='*80}\n")

    # Novel Predictions
    print(f"[NOVEL] Top {min(top_k, len(novel_results))} NOVEL Drug Predictions (not in training data):\n")
    print(f"{'Rank':<6} {'Drug':<40} {'Score':<10} {'Degree':<8}")
    print(f"{'-'*64}")

    for i, r in enumerate(novel_results[:top_k], 1):
        print(f"{i:<6} {r['drug_display']:<40} {r['score']:.4f}    {r['degree']:<8}")

    # Known Treatments (validation)
    if show_known and known_results:
        print(f"\n{'='*80}")
        print(f"[KNOWN] Known Treatments (validation - should rank high):\n")
        print(f"{'Rank':<6} {'Drug':<40} {'Score':<10} {'Overall Rank':<12}")
        print(f"{'-'*68}")

        for r in known_results:
            overall_rank = next(i for i, x in enumerate(results, 1) if x['drug'] == r['drug'])
            print(f"{'-':<6} {r['drug_display']:<40} {r['score']:.4f}    #{overall_rank:<12}")

    # Summary statistics
    all_scores = [r['score'] for r in results]
    novel_scores = [r['score'] for r in novel_results]

    print(f"\n{'='*80}")
    print(f"  Summary Statistics")
    print(f"{'='*80}")
    print(f"  Total candidates scored:    {len(results)}")
    print(f"  Novel predictions:          {len(novel_results)}")
    print(f"  Known treatments:           {len(known_results)}")
    print(f"  Score range:                [{min(all_scores):.4f}, {max(all_scores):.4f}]")
    print(f"  Mean score (all):           {np.mean(all_scores):.4f}")
    if novel_scores:
        print(f"  Mean score (novel):         {np.mean(novel_scores):.4f}")
    if known_results:
        known_scores = [r['score'] for r in known_results]
        print(f"  Mean score (known):         {np.mean(known_scores):.4f}")

        # How many known treatments appear in top-K?
        top_k_drugs = set(r['drug'] for r in results[:top_k])
        known_in_top_k = sum(1 for r in known_results if r['drug'] in top_k_drugs)
        print(f"  Known in top-{top_k}:            {known_in_top_k}/{len(known_results)}")
    print()


def save_results(disease, results, novel_results, known_results, output_dir):
    """Save results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # Full ranking CSV
    full_path = os.path.join(output_dir, f'drug_ranking_{disease}.csv')
    with open(full_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'drug', 'drug_display', 'score', 'known_treatment', 'degree'])
        for i, r in enumerate(results, 1):
            writer.writerow([i, r['drug'], r['drug_display'], f"{r['score']:.6f}",
                           'Yes' if r['known'] else 'No', r['degree']])

    # Novel predictions only
    novel_path = os.path.join(output_dir, f'novel_predictions_{disease}.csv')
    with open(novel_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'drug', 'drug_display', 'score', 'degree'])
        for i, r in enumerate(novel_results, 1):
            writer.writerow([i, r['drug'], r['drug_display'], f"{r['score']:.6f}", r['degree']])

    print(f"[SAVE] Results saved:")
    print(f"   Full ranking:      {full_path}")
    print(f"   Novel predictions: {novel_path}")

    return full_path, novel_path


def main():
    parser = argparse.ArgumentParser(
        description='Drug Candidate Ranking Pipeline - Rank drugs for a given disease '
                    'using a trained Triple Classification model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rank_drug_candidates.py --disease malaria --top_k 20
  python rank_drug_candidates.py --disease Fanconi_anemia --top_k 50 --show_known
  python rank_drug_candidates.py --list_diseases
        """
    )

    parser.add_argument('--disease', type=str, default=None,
                        help='Disease entity name (use underscores for spaces, e.g. Fanconi_anemia)')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Number of top drug candidates to display (default: 20)')
    parser.add_argument('--show_known', default=False, action='store_true',
                        help='Also show known treatments and their rankings (for validation)')
    parser.add_argument('--list_diseases', default=False, action='store_true',
                        help='List all disease entities in the KG and exit')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help=f'Path to trained TC model checkpoint (default: {DEFAULT_CHECKPOINT})')
    parser.add_argument('--plm', type=str, default=DEFAULT_PLM_PATH,
                        help=f'Path to pre-trained language model (default: {DEFAULT_PLM_PATH})')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference (default: 32)')
    parser.add_argument('--output_dir', type=str, default=PREDICTIONS_DIR,
                        help=f'Directory to save output CSVs (default: {PREDICTIONS_DIR})')

    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[DEVICE] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("[DEVICE] Using CPU (inference only - no GPU required)")

    print()

    # Load model and data
    model, data_loader, tokenizer = load_model_and_data(args.checkpoint, args.plm, device)

    # Identify entity types
    drug_entities, disease_entities, treatment_triples = identify_entity_types(data_loader)

    # List diseases mode
    if args.list_diseases:
        list_diseases(data_loader)
        return

    # Validate disease argument
    if args.disease is None:
        print("ERROR: Please specify a disease with --disease <name>")
        print("       Use --list_diseases to see available diseases")
        sys.exit(1)

    if args.disease not in data_loader.ent2id:
        # Try fuzzy matching
        matches = [e for e in data_loader.entity_list if args.disease.lower() in e.lower()]
        if matches:
            print(f"[ERROR] Disease '{args.disease}' not found. Did you mean one of these?")
            for m in matches[:10]:
                print(f"   -> {m}")
        else:
            print(f"[ERROR] Disease '{args.disease}' not found in the knowledge graph.")
            print(f"   Use --list_diseases to see available diseases.")
        sys.exit(1)

    if args.disease not in disease_entities:
        print(f"[WARN] '{args.disease}' exists in the KG but was never the tail of a "
              f"'{TREATMENT_RELATION}' triple. Results may be less meaningful.")
        print()

    # Get known treatments for this disease
    known = treatment_triples.get(args.disease, set())

    # Rank drugs
    time_start = time.time()

    results, novel_results, known_results = rank_drugs_for_disease(
        model, data_loader, args.disease, drug_entities,
        known, device, batch_size=args.batch_size, top_k=args.top_k,
        show_known=args.show_known
    )

    time_elapsed = time.time() - time_start

    # Print results
    print_results(args.disease, results, novel_results, known_results, args.top_k, args.show_known)
    print(f"[INFO] Inference time: {time_elapsed:.1f}s")

    # Save results
    save_results(args.disease, results, novel_results, known_results, args.output_dir)


if __name__ == '__main__':
    main()
