import pandas as pd
import rdflib

def convert_to_ttl(df, output_file):
    # Create a new RDF graph
    graph = rdflib.Graph()

    # Loop through each row in the DataFrame and add it to the graph
    for _, row in df.iterrows():
        subject = rdflib.URIRef(row['name_one'])
        predicate = rdflib.URIRef(row['relation'])
        try:
            object = rdflib.URIRef(row['name_two'])
            graph.add((subject, predicate, object))
        except TypeError:
            print(row['name_two'])
        
    # Serialize the graph to Turtle format and save to a file
    with open(output_file, 'w') as f:
        f.write(graph.serialize(format='turtle'))

if __name__ == '__main__':
# Example usage
    df = pd.read_csv('./data/checkpoints/filtered_triples_2.tsv', sep='\t', header=None, names=['name_one', 'relation', 'name_two', 'score'])


    convert_to_ttl(df, './data/dev.ttl')
