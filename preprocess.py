import copy
import numpy as np
import pandas as pd
import argparse
import os
import tqdm
from sklearn.model_selection import train_test_split
import statistics

relation_lookup = {
    "A+": "activates",
    "A-": "blocks",
    "B": "binds",
    "E+": "increases expression",
    "E-": "decreases expression",
    "E": "affects expression",
    "N": "inhibits",
    "O": "transport,",
    "K": "metabolism",
    "Z": "enzyme activity",
    "T": "treatment",
    "C": "inhibits cell growth",
    "Sa": "side effect",
    "Pr": "suppresses",
    "Pa": "alleviates",
    "J": "role in disease pathogenesis",
    "Mp": "progression biomarker",
    "U": "causal mutation",
    "Ud": "mutations affecting disease",
    "D": "drug target",
    "Te": "therapeutic effect",
    "Y": "polymorphisms alter risk",
    "G": "promotes progression",
    "Md": "diagnostic biomarker",
    "X": "overexpression",
    "L": "improper regulation",
    "W": "enhances",
    "V+": "activates",
    "I": "signals",
    "H": "same complex",
    "Rg": "regulates",
    "Q": "production"
}

datasets = ['gnbr.chemical-gene.tsv', 'gnbr.gene-gene.tsv', 'gnbr.chemical-disease.tsv', 'gnbr.gene-disease.tsv']
columns = ['relation', 'score', 'pubmed_id', 'type_one', 'id_one', 'name_one', 'type_two', 'id_two', 'name_two',
           'sentence']


def get_triples(files, out_dir):
    all_triples = []
    for file in tqdm.tqdm(files):
        try:
            df = pd.read_csv(file, sep='\t', header=None)
        except FileNotFoundError:
            print("File {} not found".format(file))
            continue

        df.columns = columns

        # generate triples with score
        scored = df[['name_one', 'relation', 'name_two', 'score']]
        # apply relation
        scored['relation'] = scored['relation'].apply(lambda x: relation_lookup[x])
        scored['relation'] = scored['relation'].apply(lambda x: x.replace(' ', '_'))
        # drop duplicates
        scored.drop_duplicates(subset=['name_one', 'relation', 'name_two'], inplace=True)
        scored.to_csv(os.path.join(out_dir, 'scored_{}.tsv'.format(os.path.basename(file).split('.')[1])), sep='\t', index=False)
        # generate triples without score
        no_score = scored[['name_one', 'relation', 'name_two']]
        no_score.to_csv(os.path.join(out_dir, '{}.tsv'.format(os.path.basename(file).split('.')[1])), sep='\t', index=False)

        all_triples.append(scored)

    pd.concat(all_triples).to_csv(os.path.join(out_dir, 'all_triples.tsv'), sep='\t', index=False)

    files = [os.path.basename(file) for file in files]
    return dict(zip(files, all_triples))


def filter_triples(triples_set, disease_map, out_dir, max_edges=50, degrees=2):
    checkpoint = os.path.join(out_dir, 'checkpoints')
    if not os.path.exists(checkpoint):
        os.mkdir(checkpoint)

    disease_map = pd.read_csv(disease_map, sep='\t', header=None)
    disease_map.columns = ['reference', 'gnbr_match']
    diseases = disease_map['gnbr_match'].unique().tolist()

    chem_disease = triples_set['gnbr.chemical-disease.tsv']
    gene_disease = triples_set['gnbr.gene-disease.tsv']

    entities = {disease for disease in diseases}
    filtered_triples = set()
    to_search = set()
    next_search = set()

    for i in range(degrees):
        if i == 0:
            for disease in tqdm.tqdm(diseases):
                # get all triples with disease
                disease_triples = pd.concat([gene_disease[gene_disease['name_two'] == disease],
                                             chem_disease[chem_disease['name_two'] == disease]])
                disease_triples = disease_triples.values.tolist()
                disease_triples = [tuple(triple) for triple in disease_triples]
                filtered_triples.update(disease_triples)
                add_search = {triple[0] for triple in disease_triples}
                to_search.update(add_search)

            entities.update(to_search)
            print(f'{len(to_search)} entities to search for iteration {i + 2}')
        else:
            for entity in tqdm.tqdm(to_search):
                for key, df in triples_set.items():
                    for j in range(2):
                        to_add = df[df['name_one'] == entity] if j == 0 else df[df['name_two'] == entity]
                        if len(to_add) > max_edges:
                            to_add = to_add.sample(n=max_edges, random_state=42)
                        to_add = to_add.values.tolist()
                        to_add = [tuple(triple) for triple in to_add]
                        filtered_triples.update(to_add)
                        if j == 0:
                            add_search = [triple[2] for triple in to_add if triple[2] not in entities]
                        else:
                            add_search = [triple[0] for triple in to_add if triple[0] not in entities]
                        next_search.update(add_search)
                        entities.update(add_search)

            to_search = copy.deepcopy(next_search)
            next_search = {}
            print(f'{len(to_search)} entities to search for iteration {i + 2}')

        pd.DataFrame(list(filtered_triples)).to_csv(os.path.join(checkpoint, f'filtered_triples_{i + 1}.tsv'), sep='\t',
                                                    index=False, header=False)
        pd.DataFrame(list(entities)).to_csv(os.path.join(checkpoint, f'filtered_entities_{i + 1}.tsv'), sep='\t',
                                            index=False, header=False)

        print(f'Iteration {i + 1}, found {len(filtered_triples)} triples with {len(entities)} total entities')

    filtered_triples = pd.DataFrame(list(filtered_triples), columns=['name_one', 'relation', 'name_two', 'score'])
    return filtered_triples, list(entities), diseases


def split(triples, out_dir, diseases, prop=(0.1, 0.2, 0.7)):
    triples = triples.astype({'name_one': str, 'relation': str, 'name_two': str, 'score': float})
    train = triples[~((triples['relation'] == 'treatment') & (triples['name_two'].isin(diseases)))]
    to_split = triples[(triples['relation'] == 'treatment') & (triples['name_two'].isin(diseases))]

    test_num = prop[2] * len(to_split)
    valid_num = prop[1] * len(to_split)

    train_treatment, testing = train_test_split(to_split, test_size=1-prop[0], random_state=42)
    valid, test = train_test_split(testing, test_size=test_num/(test_num + valid_num), random_state=42)

    train = pd.concat([train, train_treatment])

    print(len(train), len(valid), len(test))

    train.to_csv(os.path.join(out_dir, 'train_scored.tsv'), sep='\t', index=False, header=False)
    valid.to_csv(os.path.join(out_dir, 'dev_scored.tsv'), sep='\t', index=False, header=False)
    test.to_csv(os.path.join(out_dir, 'test_scored.tsv'), sep='\t', index=False, header=False)

    train[['name_one', 'relation', 'name_two']].to_csv(os.path.join(out_dir, 'train.tsv'), sep='\t', index=False, header=False)
    valid[['name_one', 'relation', 'name_two']].to_csv(os.path.join(out_dir, 'dev.tsv'), sep='\t', index=False, header=False)
    test[['name_one', 'relation', 'name_two']].to_csv(os.path.join(out_dir, 'test.tsv'), sep='\t', index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create final dataset")
    parser.add_argument('--path', default='formatted', type=str, help='path to the data')
    parser.add_argument('--diseases', default='./filtering/disease_mapping.tsv', type=str,
                        help='path to the disease map')
    parser.add_argument('--out', default='./data', type=str, help='path to output directory')
    parser.add_argument('--max_edges', default=15, type=int, help='max number of relations per subset for an entity')
    parser.add_argument('--max_depth', default=2, type=int, help='max number of iterations to search for relations')
    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise FileNotFoundError("Folder {} not found".format(args.path))

    if not os.path.exists(args.diseases):
        raise FileNotFoundError("File {} not found".format(args.diseases))

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    relation2text = dict(zip([s.replace(' ', '_') for s in relation_lookup.values()], relation_lookup.values()))
    pd.DataFrame.from_dict(relation2text, orient='index').to_csv(os.path.join(args.out, 'relation2text.txt'), sep='\t',
                                                                 header=False)

    files = [os.path.join(args.path, dataset) for dataset in datasets]

    try:
        all_triples = {}
        for file in files:
            file = os.path.basename(file)
            all_triples[file] = pd.read_csv(os.path.join(args.out, 'scored_{}.tsv'.format(file.split('.')[1])),
                                            sep='\t')
    except FileNotFoundError as e:
        print('No scored triples found, generating them now')
        all_triples = get_triples(files, out_dir=args.out)

    print('Triples generated, filtering now...')

    dataset_triples, entities, diseases = filter_triples(all_triples,
                                                         args.diseases,
                                                         args.out,
                                                         max_edges=args.max_edges,
                                                         degrees=args.max_depth)
    # dataset_triples = pd.read_csv(os.path.join('./data/checkpoints', 'filtered_triples_2.tsv'), sep='\t', header=None)
    # dataset_triples.columns = ['name_one', 'relation', 'name_two', 'score']
    # entities = pd.read_csv(os.path.join('./data/checkpoints', 'filtered_entities_2.tsv'), sep='\t')
    # entities = [str(entity[0]) for entity in entities.values.tolist()]
    # disease_map = pd.read_csv(args.diseases, sep='\t', header=None)
    # disease_map.columns = ['reference', 'gnbr_match']
    # diseases = disease_map['gnbr_match'].unique().tolist()

    print('Filtering done, generating split now...')
    split(dataset_triples, args.out, diseases)

    entities = [str(entity) for entity in entities]
    entities2text = dict(zip(entities, [entity.replace('_', ' ') for entity in entities]))
    pd.DataFrame.from_dict(entities2text, orient='index').to_csv(os.path.join(args.out, 'entity2text.txt'), sep='\t', header=False)

    print('Split and saved to {}'.format(args.out))
