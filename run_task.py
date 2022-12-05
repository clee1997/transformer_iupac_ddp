## this module needs some fixing. 


import os
import torch
# from train import transformer
from transformer import Seq2SeqTransformer
from data_utils import generate_square_subsequent_mask, token_transform, tensor_transform
from get_vocab import vocab
from config import tok_ids, ckpt_path, params


# is it necessary that these run on two gpus as well?
# i don't think so. especially if this is an api function, we don't want to REQUIRE a gpu availability. 
class ErrorCorrect:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ckpt_path = ckpt_path

        self.num_encoder_layers = params['NUM_ENCODER_LAYERS']
        self.num_decoder_layers = params['NUM_DECODER_LAYERS']
        self.emb_size = params['EMB_SIZE']
        self.nhead = params['NHEAD']
        self.ffn_hid_dim = params['FFN_HID_DIM']
        self.batch_size = params['BATCH_SIZE']
        self.num_epochs = params['NUM_EPOCHS'] 

        self.src_vocab_size = len(vocab['src'])
        self.tgt_vocab_size = len(vocab['tgt'])

    # function to generate output sequence using greedy algorithm 
    def greedy_decode(self, model, src, src_mask, max_len, start_symbol):
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)

        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
        for i in range(max_len-1):
            memory = memory.to(self.device)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0), 0) # rank=0
                        .type(torch.bool)).to(self.device)
            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == tok_ids['EOS_IDX']:
                break
        return ys

    ## change return statement!!
    # actual function to translate input sentence into target language
    @torch.no_grad
    def run_task(self, src_sentence: str):
        model = self.prepare_model()
        model.eval()

        # src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1) # tensor_transform까지. 

        src_tokens = token_transform(src_sentence)
        token_ids_src = vocab['src'](src_tokens)
        src = tensor_transform(token_ids_src).view(-1, 1)


        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(model,  src, src_mask, max_len=num_tokens + 5, start_symbol=tok_ids['BOS_IDX']).flatten()

        return "".join(vocab['tgt'].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
    
    def prepare_model(self):

        model = Seq2SeqTransformer(self.num_encoder_layers, self.num_decoder_layers, self.emb_size, self.nhead, self.src_vocab_size, self.tgt_vocab_size, self.ffn_hid_dim)

        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        model.to(self.device)
        model.load_state_dict(torch.load(self.ckpt_path))

        return model




# print(run_task(model, "{(28)-1-[[(4R)-3,4-dihydro-2H-c-hromen-4-ylJamino]-1-oxoprop-an-2-yl]-2-ethoxypyridine-3-carboxylate"))

src_sentence = "{(28)-1-[[(4R)-3,4-dihydro-2H-c-hromen-4-ylJamino]-1-oxoprop-an-2-yl]-2-ethoxypyridine-3-carboxylate"
error_correct = ErrorCorrect()
corrected = error_correct.run_task(src_sentence)

print(f'ORIGINAL: {src_sentence} \n')
print(f'CORRECTED: {corrected}')