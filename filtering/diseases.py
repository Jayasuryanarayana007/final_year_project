import pandas as pd
import json
import difflib

def get_id(x):
    if 'MESH:' in x and len(x) > 5:
        return x.split(':')[1]
    return 'none'




if __name__ == '__main__':
    df = pd.read_csv('CTD_diseases.csv')
    print(df.columns)
    df = df[['id', 'disease']]
    with open('ids.txt', 'r') as f:
        ids = f.readlines()
        ids = [x.strip() for x in ids]
    df['id'] = df['id'].apply(get_id)
    df = df[df['id'].isin(ids)]
    df.to_csv('filtered_diseases.txt', sep='\t', index=False, header=False)
    df.set_index('id', inplace=True)

    df2 = df['disease'].drop_duplicates()
    df2.to_csv('diseases.txt', sep='\t', index=False, header=False)


    entities_df = pd.read_csv('entities.tsv2', sep='\t')
    print(df.head())
    entities_df.columns = ['type', 'id', 'name']
    entities_df=entities_df.astype({'id': 'str'})
    entities_df = entities_df[entities_df['type'] == 'Disease']
    entities_df['id'] = entities_df['id'].apply(get_id)
    entities_df = entities_df[entities_df['id'].isin(ids)]
    print(entities_df.head())
    entities_df = entities_df.drop_duplicates()
    entities_df.set_index('id', inplace=True)
    entities_df = entities_df.groupby('id').apply(lambda x: x.to_numpy().tolist()).to_dict()
    # for i in range(len(entities_df)):
    #     id = entities_df.iloc[i]['id']
    #     reference = df.loc[id]['disease']
    #     reference = modify(reference)


    with open('gnbr_diseases.txt', 'w') as writer:
        writer.write(json.dumps(entities_df))

    mapping = {}
    for k, v in entities_df.items():
        entities_df[k] = [str(i[1]) for i in v]
        copy = entities_df[k]
        entities_df[k] = [i.replace('-', ' ').replace('_', ' ').lower() for i in entities_df[k]]
        before_filter = dict(zip(entities_df[k], copy))
        print(before_filter)
        try:
            reference = df.loc[k]['disease'].lower()
            if len(reference.split(',')) > 1:
                reference = reference.split(',')[1].strip() + " " + reference.split(',')[0].strip()
                for i in range(2, len(reference.split(','))):
                    reference += " " + reference.split(',')[i].strip()
        except KeyError as e:
            print('key error')
            continue

        closest = difflib.get_close_matches(reference, entities_df[k], n=1, cutoff=0.3)
        if len(closest) > 0:
            print(before_filter[closest[0]])
            mapping[reference] = (before_filter[closest[0]])

    print(len(mapping))
    out_df = pd.DataFrame.from_dict(mapping, orient='index', columns=['match'])
    print(out_df.head())
    out_df.reset_index(inplace=True)
    out_df.to_csv('disease_mapping.tsv', sep='\t', index=False, header=False)