from tqdm import tqdm
from transformers import BertTokenizerFast

# repositor id for saving the tokenizer
tokenizer_id="biobert-tiny"

# create a python generator to dynamically load the data
def batch_iterator(batch_size=10000):
    for i in tqdm(range(0, 20619956, batch_size)):
        temp = []
        for x in range(0, 10000):
            try:
                with open(f"./abstracts_v2/{i+x}.txt", "r") as text_file:
                    temp.append(text_file.read())
            except:
                pass
        yield temp

# create a tokenizer from existing one to re-use special tokens
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

bert_tokenizer = tokenizer.train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=32_000)
bert_tokenizer.save_pretrained("tokenizer")
bert_tokenizer.push_to_hub(tokenizer_id)
