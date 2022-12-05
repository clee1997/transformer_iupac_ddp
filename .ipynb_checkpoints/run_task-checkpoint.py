import os
import torch
from multiprocessing import Process, Queue
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
# from train import transformer
from transformer import Seq2SeqTransformer
from data_utils import generate_square_subsequent_mask, token_transform, tensor_transform
from get_vocab import vocab
from config import tok_ids, ckpt_path, params



class ErrorCorrect:
    def __init__(self):
        
        self.ckpt_path = ckpt_path
        self.n_gpus = torch.cuda.device_count()
        # self.Q = Queue()

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
    def greedy_decode(self, model, src, src_mask, max_len, start_symbol, rank):
        src = src.to(rank)
        src_mask = src_mask.to(rank)
        
        # print(f'dir(model) = {dir(model)}')
        # print(f'dir(model.module) = {dir(model.module)}')

        memory = model.module.encode(src, src_mask) # potential fix
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(rank)
        for i in range(max_len-1):
            memory = memory.to(rank)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0), 0) # rank=0
                        .type(torch.bool)).to(rank)
            out = model.module.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.module.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == tok_ids['EOS_IDX']:
                break
        return ys

    def run_task(self, rank, src_sentence): # to be used in spawn
        
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '22141'

        torch.cuda.set_device(rank) 
        dist.init_process_group(backend="nccl", world_size=self.n_gpus, rank=rank)
        
        model = self.prepare_ddp_model(rank)
        model.eval()

        src_tokens = token_transform(src_sentence)
        token_ids_src = vocab['src'](src_tokens)
        src = tensor_transform(token_ids_src).view(-1, 1)


        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(model,  src, src_mask, max_len=num_tokens + 5, start_symbol=tok_ids['BOS_IDX'], rank=rank).flatten()

        # return "".join(vocab['tgt'].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
        res = "".join(vocab['tgt'].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
        if rank == 0:
            print(f'CORRECTED: {res}')
        return
    
    def prepare_ddp_model(self, rank):

        model = Seq2SeqTransformer(self.num_encoder_layers, self.num_decoder_layers, self.emb_size, self.nhead, self.src_vocab_size, self.tgt_vocab_size, self.ffn_hid_dim)

        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        model = model.to(rank)
        model_ddp = DDP(model, device_ids=[rank])
        model_ddp.load_state_dict( torch.load(self.ckpt_path, map_location=f'cuda:{rank}') )

        return model_ddp




ngpus = torch.cuda.device_count()
assert (ngpus >= 2), 'Less than 2 gpus available'

# src_sentence = "{(28)-1-[[(4R)-3,4-dihydro-2H-c-hromen-4-ylJamino]-1-oxoprop-an-2-yl]-2-ethoxypyridine-3-carboxylate"
# src_sentence = 'N-{{(@R)-2,3-dihydro-1,4-benzo-dioxin-3-ylJmethyl]-6-methyl-4-x0-1-[2-(trifluoromethyl)pheny-IJpyridazine-3-carboxamide'
src_sentence = 'N-{2-(cyclohexen-1  -yl)ethy!]-3-(3,4-dihydro-1H-isoquinolin-2-ylsulfonyl)thiophene-2-carboxa-mide'
error_correct = ErrorCorrect()

if __name__ == '__main__':
    print(f'\nORIGINAL: {src_sentence} \n')
    mp.spawn(error_correct.run_task, args=(src_sentence,), nprocs=ngpus, join=True)
    