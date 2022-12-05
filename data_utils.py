import torch

from config import tok_ids

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids):
    return torch.cat((torch.tensor([tok_ids['BOS_IDX']]), 
                      torch.tensor(token_ids), 
                      torch.tensor([tok_ids['EOS_IDX']])))

def token_transform(text):
    try: tokenized_text = list(text) 
    except TypeError: return []
    return tokenized_text

# to be used inside get_vocab()
def yield_tokens(dataset, source = 'src'):# -> List[str]:

    print(f'type is {type(dataset)}')

    for items in dataset:
        if source == 'src':
            # print(f'items[0] = {items[0]}')
            yield token_transform(items[0]) # don't need row idx bc of the loop
        elif source == 'tgt':
            # print(f'items[1] = {items[1]}')
            yield token_transform(items[1])

# device should be specified at another module. 
def generate_square_subsequent_mask(sz, rank):
    
    # device = torch.device(f'cuda:{rank}')

    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, rank):
    
    ## if device=rank doesn't work:
    # DEVICE = torch.device('cuda:1')
    device = torch.device(f'cuda:{rank}')

    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, rank)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == tok_ids['PAD_IDX']).transpose(0, 1)
    tgt_padding_mask = (tgt == tok_ids['PAD_IDX']).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# ## revised ones without device specification. delete the above two copies. 
# def generate_square_subsequent_mask(sz):
#     mask = torch.triu(torch.ones((sz, sz)) == 1).transpose(0, 1)
#     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#     return mask


# def create_mask(src, tgt):
#     src_seq_len = src.shape[0]
#     tgt_seq_len = tgt.shape[0]

#     tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
#     src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

#     src_padding_mask = (src == params['PAD_IDX']).transpose(0, 1)
#     tgt_padding_mask = (tgt == params['PAD_IDX']).transpose(0, 1)
#     return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask