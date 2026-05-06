import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle

relation_lookup = {
    'T': "treatment",
    'C': "inhibits cell growth",
    'Sa': "side effect",
    'Pr': "prevents",
    'Pa': "alleviates",
    'J': "role in disease pathogenesis",
}

# Part I files
part_i_files = ['part-i-chemical-disease-path-theme-distributions.tsv']
part_ii_files = ['part-ii-dependency-paths-chemical-disease-sorted-with-themes.tsv']

# Process part I files
for file in part_i_files:
    path = Path('./cache/processed_{}.p'.format(file.removesuffix(".txt")))

    if path.exists():
        print("File {} already processed".format(file))
        continue

    master_dict = {}

    with open("./files/{}".format(file), 'r') as f:
        headers = f.readline().strip('\n').split('\t')
        for line in tqdm.tqdm(f):
            split = line.strip('\n').split('\t')
            assert len(headers) == len(split)

            # dependency path
            path = split[0]

            scores = split[1:]
            theme = headers[np.argmax(scores) + 1]

            master_dict[path] = theme

    # Cache so we dont have to do this again
    with open('./cache/processed_{}.p'.format(file.removesuffix(".txt")), 'wb') as fp:
        pickle.dump(master_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

# Process part II files
for file in part_ii_files:
    path = Path('./cache/processed_{}.p'.format(file.removesuffix(".txt")))

    if path.exists():
        print("File {} already processed".format(file))
        continue

    master_array = []

    with open("./files/{}".format(file), 'r') as f:
        headers = ['PubMed ID','Sentence number (0 = title)','First entity name, formatted','First entity name, location (characters from start of abstract)','Second entity name, formatted','Second entity name, location','First entity name, raw string','Second entity name, raw string','First entity name, database ID(s)','Second entity name, database ID(s)','First entity type (Chemical, Gene, Disease)','Second entity type (Chemical, Gene, Disease)','Dependency path','Sentence, tokenized']
        for lineno,line in tqdm.tqdm(enumerate(f)):
            split = line.strip('\n').split('\t')
            assert len(headers) == len(split)

            row = { h:v for h,v in zip(headers,split) }

            text1 = row['First entity name, formatted']
            text2 = row['Second entity name, formatted']
            path = row['Dependency path'].lower()

            master_array.append([text1, text2, path])

    # Cache so we dont have to do this again
    with open('./cache/processed_{}.p'.format(file.removesuffix(".txt")), 'wb') as fp:
        pickle.dump(master_array, fp, protocol=pickle.HIGHEST_PROTOCOL)

# Generate triples
part_i_and_ii_files = [ (file1, file2) for file1, file2 in zip(part_i_files, part_ii_files) ]

for pair in part_i_and_ii_files:
    cached_file_part_i = './cache/processed_{}.p'.format(pair[0].removesuffix(".txt"))
    cached_file_part_ii = './cache/processed_{}.p'.format(pair[1].removesuffix(".txt"))

    with open(cached_file_part_i, 'rb') as f1p, open(cached_file_part_ii, 'rb') as f2p:
        path_to_theme = pickle.load(f1p)
        entities_and_paths = pickle.load(f2p)

        # TODO: Generalize the following two lines when we have more themes
        chemicals = []
        diseases = []
        relations = []

        triples = []

        i = 0

        for entity1, entity2, path in entities_and_paths:
            theme = path_to_theme[path]
            entity1 = entity1.lower()
            entity2 = entity2.lower()

            # Skip these themes for now
            # if theme not in relation_lookup or relation_lookup[theme] != "treatment":
            #     continue

            i += 1
            if i == 10000:
                break

            triple = (entity1, relation_lookup[theme], entity2)
            triples.append(triple)
            chemicals.append(entity1)
            diseases.append(entity2)
            relations.append(relation_lookup[theme])

        triples = list(set(triples))
        chemicals = list(set(chemicals))
        diseases = list(set(diseases))

        # Write array to tsv
        df = pd.DataFrame(list(set(triples)), columns=['entity1', 'theme', 'entity2'])
        df.to_csv('./triples/unique_{}.tsv'.format(pair[1].replace("part-ii-dependency-paths-", "").replace("-sorted-with-themes.txt", "")), sep='\t', index=False)

        # df = pd.DataFrame(triples, columns=['entity1', 'theme', 'entity2'])
        # df.to_csv('./triples/non_unique_{}.tsv'.format(pair[1].replace("part-ii-dependency-paths-", "").replace("-sorted-with-themes.txt", "")), sep='\t', index=False)


        # TODO: Generalize the following lines when we have more themes
        # Write entities to tsv
        df = pd.DataFrame(list(set(chemicals)))
        df.to_csv('./entities/chemicals.tsv', sep='\t', index=False)

        df = pd.DataFrame(list(set(diseases)))
        df.to_csv('./entities/diseases.tsv', sep='\t', index=False)

        # Generate train, valid, test splits
        train_split = 0.6
        valid_split = 0.2
        test_split = 0.2

        train_sample_count = int(len(triples) * train_split)
        valid_sample_count = int(len(triples) * valid_split)
        test_sample_count = int(len(triples) * test_split)

        train_triples = []
        valid_test_eligible_triples = []

        entities_in_train = set()

        random.seed(42)
        random.shuffle(triples)

        remove_indices = []

        for x in tqdm.tqdm(range(len(triples))):
            entity1, relation, entity2 = triples[x]

            if entity1 in entities_in_train and entity2 in entities_in_train:
                continue

            entities_in_train.add(entity1)
            entities_in_train.add(entity2)
            train_triples.append((entity1, relation, entity2))
            remove_indices.append(x)

        # Remove the selected triples
        for index in tqdm.tqdm(sorted(remove_indices, reverse=True)):
            del triples[index]

        valid_triples = triples[:valid_sample_count]
        test_triples = triples[valid_sample_count:valid_sample_count*2]

        train_triples.extend(triples[valid_sample_count*2:])

        # Write the train, valid, test splits to tsv
        df = pd.DataFrame(train_triples)
        df.to_csv('./data/train.tsv', sep='\t', index=False, header=False)

        df = pd.DataFrame(valid_triples)
        df.to_csv('./data/dev.tsv', sep='\t', index=False, header=False)

        df = pd.DataFrame(test_triples)
        df.to_csv('./data/test.tsv', sep='\t', index=False, header=False)
        
        # Write the entities to tsv
        all_entities = list(set(chemicals)) + list(set(diseases))
        entities2text = { (c, c.replace("_", " ")) for c in all_entities }
        df = pd.DataFrame(entities2text)
        df.to_csv('./data/entity2text.txt', sep='\t', index=False, header=False)

        # Write the relations to tsv
        all_relations = list(set(relations))
        relations2text = { (r, r.replace("_", " ")) for r in all_relations }
        df = pd.DataFrame(relations2text)
        df.to_csv('./data/relation2text.txt', sep='\t', index=False, header=False)


        # Write the entities to tsv
        all_entities = list(set(chemicals)) + list(set(diseases))
        entities2text = [(entity, count) for count, entity in enumerate(all_entities)]
        # entities2text.sort(key=lambda x: x[1])
        df = pd.DataFrame(entities2text)
        df.to_csv('./openke/entity2id.txt', sep='\t', index=False, header=False)

        # Write the relations to tsv
        all_relations = list(set(relations))
        relations2text = {(relation, count) for count, relation in enumerate(all_relations)}
        df = pd.DataFrame(relations2text)
        print(df.head())
        df.to_csv('./openke/relation2id.txt', sep='\t', index=False, header=False)

        entity_map = {entity: count  for count, entity in enumerate(all_entities)}
        relation_map = {relation: count for count, relation in enumerate(all_relations)}

        train_triples = [(entity_map[tup[0]], entity_map[tup[2]], relation_map[tup[1]]) for tup in train_triples]
        df = pd.DataFrame(train_triples)
        df.to_csv('./openke/train2id.txt', sep=' ', index=False, header=False)

        valid_triples = [(entity_map[tup[0]], entity_map[tup[2]], relation_map[tup[1]]) for tup in valid_triples]
        df = pd.DataFrame(valid_triples)
        df.to_csv('./openke/valid2id.txt', sep=' ', index=False, header=False)

        test_triples = [(entity_map[tup[0]], entity_map[tup[2]], relation_map[tup[1]]) for tup in test_triples]
        df = pd.DataFrame(test_triples)
        df.to_csv('./openke/test2id.txt', sep=' ', index=False, header=False)