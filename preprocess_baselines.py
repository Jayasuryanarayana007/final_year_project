import csv

entities = {}
relations = {}

i = 0
with open('./data copy/entity2text.txt', 'r') as file:
    for line in file:
        entity_name = line.split()[0]
        entities[entity_name] = i
        i += 1

i = 0
with open('./data copy/relation2text.txt', 'r') as file:
    for line in file:
        relation_name = line.split()[0]
        relations[relation_name] = i
        i += 1

with open('./data copy/entity2id.txt', 'w') as f:
    f.write(str(len(entities)) + "\n")
    for entity in entities:
        f.write(entity + "\t" + str(entities[entity]) + "\n")

with open('./data copy/relation2id.txt', 'w') as f:
    f.write(str(len(relations)) + "\n")
    for relation in relations:
        f.write(relation + "\t" + str(relations[relation]) + "\n")



train = []
valid = []
test = []

with open('./data copy/train.tsv', 'r') as tsv_file:
    tsv_reader = csv.reader(tsv_file, delimiter='\t')
    for row in tsv_reader:
        try:
            train.append((entities[row[0]], entities[row[2]], relations[row[1]]))
        except:
            pass

with open('./data copy/dev.tsv', 'r') as tsv_file:
    tsv_reader = csv.reader(tsv_file, delimiter='\t')
    for row in tsv_reader:
        try:
            valid.append((entities[row[0]], entities[row[2]], relations[row[1]]))
        except:
            pass

with open('./data copy/test.tsv', 'r') as tsv_file:
    tsv_reader = csv.reader(tsv_file, delimiter='\t')
    for row in tsv_reader:
        try:
            test.append((entities[row[0]], entities[row[2]], relations[row[1]]))
        except:
            pass

with open('./data copy/train2id.txt', 'w') as f:
    f.write(str(len(train)) + "\n")
    for tup in train:
        f.write(str(tup[0]) + "\t" + str(tup[1]) + "\t" + str(tup[2]) + "\n")

with open('./data copy/valid2id.txt', 'w') as f:
    f.write(str(len(valid)) + "\n")
    for tup in valid:
        f.write(str(tup[0]) + "\t" + str(tup[1]) + "\t" + str(tup[2]) + "\n")

with open('./data copy/test2id.txt', 'w') as f:
    f.write(str(len(test)) + "\n")
    for tup in test:
        f.write(str(tup[0]) + "\t" + str(tup[1]) + "\t" + str(tup[2]) + "\n")