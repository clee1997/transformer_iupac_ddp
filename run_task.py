## this module needs some fixing. 


import os
import torch
# from train import transformer
from transformer import Seq2SeqTransformer
from masks import generate_square_subsequent_mask
from get_vocab import vocab
from collate import token_transform, tensor_transform


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab['src'])
TGT_VOCAB_SIZE = len(vocab['tgt'])
EMB_SIZE = 512
NHEAD = 4 # 8이었음. 
FFN_HID_DIM = 512
BATCH_SIZE = 128 ## i don't fucking get this. 128? srsly?
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
NUM_EPOCHS = 20 ###


model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, 
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

# model = TheModelClass(*args, **kwargs)
# model = transformer
ckpt_path = '/Users/chaeeunlee/Downloads/ckpt_saved/ckpt_epoch5.pt'
model.load_state_dict(torch.load(ckpt_path))
# model.eval()

# function to generate output sequence using greedy algorithm 
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

## change return statement!!
# actual function to translate input sentence into target language
def run_task(model: torch.nn.Module, src_sentence: str):
    model.eval()

    # src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1) # tensor_transform까지. 

    src_tokens = token_transform(src_sentence)
    token_ids_src = vocab['src'](src_tokens)
    src = tensor_transform(token_ids_src).view(-1, 1)


    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()

    return "".join(vocab['tgt'].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")



print(run_task(model, "{(28)-1-[[(4R)-3,4-dihydro-2H-c-hromen-4-ylJamino]-1-oxoprop-an-2-yl]-2-ethoxypyridine-3-carboxylate"))