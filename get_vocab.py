# from collate import PairDataset
import torch
from torch.utils.data import Dataset, IterableDataset
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
import random
from os.path import exists

from config import params, csv_path, saved_vocab_path


def get_vocab(csv_path): # vocab_transform

    dataset = PairIterableDataset(csv_path)
    
    res = {}
    res['src'] = build_vocab_from_iterator(yield_tokens(dataset, source='src'),
                                            min_freq=1,
                                            specials=params.special_symbols,
                                            special_first=True)
    res['src'].set_default_index(params['UNK_IDX'])
    res['tgt'] = res['src'] # you could technically build from scratch but probs not necessary
    res['tgt'].set_default_index(params['UNK_IDX'])
    
    return res


if not exists(saved_vocab_path):
    vocab = get_vocab(csv_path) 
    torch.save(vocab, saved_vocab_path)
else:
    vocab = torch.load(saved_vocab_path)
