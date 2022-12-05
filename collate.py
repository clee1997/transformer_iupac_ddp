## IMPORT FROM MODULES

from get_vocab import vocab
from data_utils import token_transform, tensor_transform
from config import params

## IMPORT LIBRARIES

import os
import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):

    src_batch, tgt_batch = [], []

    # Apply text_transform(three transforms)
    for src_sample, tgt_sample in batch: 

        # print(f'src_sample = {src_sample}')

        src_sample_tokenized, tgt_sample_tokenized = token_transform(src_sample), token_transform(tgt_sample)

        token_ids_src = vocab['src'](src_sample_tokenized)
        token_ids_tgt = vocab['tgt'](tgt_sample_tokenized)

        src_sample_tensor, tgt_sample_tensor = tensor_transform(token_ids_src), tensor_transform(token_ids_tgt)

        src_batch.append(src_sample_tensor)
        tgt_batch.append(tgt_sample_tensor)

    # pad
    src_batch = pad_sequence(src_batch, padding_value=params['PAD_IDX'])
    tgt_batch = pad_sequence(tgt_batch, padding_value=params['PAD_IDX'])

    return src_batch.cuda(), tgt_batch.cuda()