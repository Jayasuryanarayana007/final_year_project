
if __name__ == '__main__':
    mesh = False
    ids = []
    with open('/Users/jwang/Downloads/en_product1.xml', 'r', encoding='ISO-8859-1') as f, open('ids.txt', 'w') as w:
        for line in f:

            if mesh:
                ids.append(line.split('>')[1].split('<')[0] + '\n')
                mesh = False

            if 'MeSH' in line:
                mesh = True
        print(ids)
        print(len(ids))
        w.writelines(ids)
        w.close()


