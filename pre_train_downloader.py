import gzip
import os
import xml.etree.ElementTree as ET
from ftplib import FTP

import tqdm

# Set up the FTP connection
ftp = FTP('ftp.ncbi.nlm.nih.gov', timeout=300)
ftp.login()

i = 7856571

# Change to the target directory
ftp.cwd("/pubmed/baseline/")

# Get a list of all the tar files in the directory
tar_files = [file for file in ftp.nlst() if file.endswith('.xml.gz')]

for file in tqdm.tqdm(tar_files[393:]):
    with open("temp.xml.gz", 'wb') as f:
        ftp.retrbinary('RETR {}'.format(file), f.write)

    with gzip.open("temp.xml.gz", 'rb') as f:
        file_contents = f.read()

    # Write the uncompressed contents to a new file
    with open("temp.xml", 'wb') as f:
        f.write(file_contents)

    tree = ET.parse("temp.xml")
    root = tree.getroot()
    tag_name = 'AbstractText'

    for elem in root.iter(tag_name):
        if elem.text:
            with open(f"./abstracts_v2/{i}.txt", "w") as text_file:
                text_file.write(elem.text)

            i += 1

    os.system('rm temp.xml')
    os.system('rm temp.xml.gz')
