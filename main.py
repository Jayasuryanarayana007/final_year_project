# coding=UTF-8
import argparse
import os
import pickle

# import pdb
import random

import numpy as np
import torch
from tqdm import tqdm
from torch.optim import AdamW
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    BertModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from dataloader import DataLoader
from model import LMKE
from trainer import Trainer

if __name__ == '__main__':
    # argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--bert_lr', type=float, default=1e-5)
    parser.add_argument('--model_lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--weight_decay', type=float, default=1e-7)

    parser.add_argument('--data', type=str, default='rare-disease')
    parser.add_argument('--plm', type=str, default='biobert', choices=['tiny', 'biobert', 'tiny-squared', 'biobert-tiny'])
    parser.add_argument('--description', type=str, default='desc')

    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--load_epoch', type=int, default=-1)
    parser.add_argument('--load_metric', type=str, default='hits1')

    # directly run test
    parser.add_argument('--link_prediction', default=False, action='store_true')
    parser.add_argument('--triple_classification', default=False, action='store_true')

    parser.add_argument('--add_tokens', default=False, action='store_true',
                        help='add entity and relation tokens into the vocabulary')
    parser.add_argument('--p_tuning', default=False, action='store_true', help='add learnable soft prompts')
    parser.add_argument('--prefix_tuning', default=False, action='store_true',
                        help='fix language models and only tune added components')
    parser.add_argument('--rdrop', default=False, action='store_true')
    parser.add_argument('--self_adversarial', default=False, action='store_true',
                        help='self adversarial negative sampling')
    parser.add_argument('--no_use_lm', default=False, action='store_true')
    parser.add_argument('--use_structure', default=False, action='store_true')
    parser.add_argument('--contrastive', default=False, action='store_true')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--load_descriptions', default=False, action='store_true')
    parser.add_argument('--predictions', type=str, default='./predictions/')
    parser.add_argument('--task', default='LP', choices=['LP', 'TC'])
    parser.add_argument('--uncertainty', default=False, action='store_true')

    arg = parser.parse_args()

    if arg.task == 'TC':
        neg_rate = 1
    else:
        neg_rate = 0

    identifier = '{}-u={}-b={}-d={}'.format(arg.plm, arg.uncertainty, arg.batch_size, arg.load_descriptions)
    # identifier = '{}-{}-{}-batch_size={}-descriptions={}'.format(arg.data, arg.plm, arg.description, arg.batch_size,
    #                                                               arg.load_descriptions)

    # identifier = '{}-{}-{}-batch_size={}-prefix_tuning={}'.format(arg.data, arg.plm, arg.description, arg.batch_size,
    #                                                               arg.p_tuning)
    # Set random seed
    random.seed(arg.seed)
    np.random.seed(arg.seed)
    torch.manual_seed(arg.seed)

    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    else:
        print("WARNING: CUDA not available. Running on CPU (this will be slow).")
        device = torch.device('cpu')

    if arg.plm == 'bert':
        plm_name = "bert-base-uncased"
        t_model = 'bert'
    elif arg.plm == 'biobert':
        plm_name = './cached_model/biobert/' # "dmis-lab/biobert-v1.1"
        t_model = 'bert'
    elif arg.plm == 'tiny':
        plm_name = "./cached_model/bert-tiny/"
        t_model = 'bert'
    elif arg.plm == 'biobert-tiny':
        plm_name = './cached_model/biobert-tiny/'
        t_model = 'bert'
    elif arg.plm == 'tiny-squared':
        plm_name = './cached_model/biobert-tiny-squared/'
        t_model = 'bert'

    if not arg.uncertainty:
        if arg.data == 'chemical-gene':
            in_paths = {
                'dataset': arg.data,
                'train': './legacy/data/train.tsv',
                'valid': './legacy/data/dev.tsv',
                'test': './legacy/data/test.tsv',
                'text': ['./legacy/data/entity2text.txt', './legacy/data/relation2text.txt']
            }
        elif arg.data == 'rare-disease':
            in_paths = {
                'dataset': arg.data,
                'train': './data/train.tsv',
                'valid': './data/dev.tsv',
                'test': './data/test.tsv',
                'text': ['./data/entity2text.txt', './data/relation2text.txt']
            }
    else:
        if arg.data == 'rare-disease':
            in_paths = {
                'dataset': arg.data,
                'train': './data/train_scored.tsv',
                'valid': './data/dev_scored.tsv',
                'test': './data/test_scored.tsv',
                'text': ['./data/entity2text.txt', './data/relation2text.txt']
            }
    # elif arg.data == 'umls':
    #     in_paths = {
    #         'dataset': arg.data,
    #         'train': './sample_lmke_data/umls/train.tsv',
    #         'valid': './sample_lmke_data/umls/dev.tsv',
    #         'test': './sample_lmke_data/umls/test.tsv',
    #         'text': ['./sample_lmke_data/umls/entity2textlong.txt', './sample_lmke_data/umls/relation2text.txt']
    #     }

    lm_tokenizer = AutoTokenizer.from_pretrained(plm_name, do_basic_tokenize=False)
    lm_config = AutoConfig.from_pretrained(plm_name)
    lm_model = AutoModel.from_pretrained(plm_name, config=lm_config)

    data_loader = DataLoader(in_paths, lm_tokenizer, batch_size=arg.batch_size, neg_rate=neg_rate,
                             add_tokens=arg.add_tokens, p_tuning=arg.p_tuning, rdrop=arg.rdrop, model=t_model,
                             descriptions=arg.load_descriptions, uncertainty=arg.uncertainty)

    if arg.add_tokens:
        data_loader.adding_tokens()
        lm_model.resize_token_embeddings(len(lm_tokenizer))

    lm_model.to(device)

    model = LMKE(lm_model, n_ent=len(data_loader.ent2id), n_rel=len(data_loader.rel2id), add_tokens=arg.add_tokens,
                 contrastive=arg.contrastive)

    no_decay = ["bias", "LayerNorm.weight"]
    param_group = [
        {'lr': arg.model_lr, 'params': [p for n, p in model.named_parameters()
                                        if ('lm_model' not in n) and
                                        (not any(nd in n for nd in no_decay))],
         'weight_decay': arg.weight_decay},
        {'lr': arg.model_lr, 'params': [p for n, p in model.named_parameters()
                                        if ('lm_model' not in n) and
                                        (any(nd in n for nd in no_decay))],
         'weight_decay': 0.0},
    ]

    if not arg.prefix_tuning:
        param_group += [
            {'lr': arg.bert_lr, 'params': [p for n, p in model.named_parameters()
                                           if ('lm_model' in n) and
                                           (not any(nd in n for nd in no_decay))],  # name中不包含bias和LayerNorm.weight
             'weight_decay': arg.weight_decay},
            {'lr': arg.bert_lr, 'params': [p for n, p in model.named_parameters()
                                           if ('lm_model' in n) and
                                           (any(nd in n for nd in no_decay))],
             'weight_decay': 0.0},
        ]

    optimizer = AdamW(param_group)  # transformer AdamW
    scheduler = None
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10000, num_training_steps=arg.epoch * data_loader.step_per_epc)

    hyperparams = {
        'batch_size': arg.batch_size,
        'epoch': arg.epoch,
        'identifier': identifier,
        'load_path': arg.load_path,
        'evaluate_every': 1,
        'update_every': 1,
        'load_epoch': arg.load_epoch,
        'load_metric': arg.load_metric,
        'prefix_tuning': arg.prefix_tuning,
        'plm': arg.plm,
        'description': arg.description,
        'neg_rate': neg_rate,
        'add_tokens': arg.add_tokens,
        'p_tuning': arg.p_tuning,
        'rdrop': arg.rdrop,
        'use_structure': arg.use_structure,
        'self_adversarial': arg.self_adversarial,
        'no_use_lm': arg.no_use_lm,
        'contrastive': arg.contrastive,
        'task': arg.task,
        'wandb': arg.wandb,
        'predictions': arg.predictions,
        'uncertainty': arg.uncertainty,
        'evaluate_tc': arg.triple_classification
    }

    trainer = Trainer(data_loader, model, lm_tokenizer, optimizer, device, hyperparams)

    if arg.link_prediction:
        trainer.link_prediction(split='test')
    elif arg.triple_classification:
        trainer.triple_classification(split='test')
    else:
        trainer.run()
