entities = {}
relations = {}

i = 0
with open('./data/entity2text.txt', 'r') as file:
    for line in file:
        entity_name = line.split()[0]
        entities[i] = entity_name
        i += 1

i = 0
with open('./data/relation2text.txt', 'r') as file:
    for line in file:
        relation_name = line.split()[0]
        relations[i] = relation_name
        i += 1

dist_mult_results = []
transe_results = []

with open('dist_mult_ids.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        head = int(row[0])
        relation = int(row[1])
        tail = int(row[2])
        label = int(row[3])

        dist_mult_results.append((entities[head], relations[relation], entities[tail], label))

with open('dist_mult_ids.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        head = int(row[0])
        relation = int(row[1])
        tail = int(row[2])
        label = int(row[3])

        transe_results.append((entities[head], relations[relation], entities[tail], label))

with open("./distmult_results.tsv", "w", newline="") as tsvfile:
    writer = csv.writer(tsvfile, delimiter="\t")
    writer.writerows(dist_mult_results)

with open("./transe_results.tsv", "w", newline="") as tsvfile:
    writer = csv.writer(tsvfile, delimiter="\t")
    writer.writerows(transe_results)
                                                   