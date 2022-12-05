import os
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from timeit import default_timer as timer
from transformer import Seq2SeqTransformer
from torch.utils.data.distributed import DistributedSampler


from get_vocab import vocab
from collate import collate_fn
from masks import create_mask
from config import tok_ids, params, csv_path, saved_path
from pair_datasets import PairDataset

## to-d0
# shouldn't we also put dataset to device? -> 그건 collate에서 해야 하는 듯. 
# setup. some cuda setup. 
# mp. spawn!! 

class Trainer:
    def __init__(self):
        # self.rank = rank
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tok_ids['PAD_IDX'])
        self.n_gpus = torch.cuda.device_count()
        self.csv_path = csv_path

        self.num_encoder_layers = params['NUM_ENCODER_LAYERS']
        self.num_decoder_layers = params['NUM_DECODER_LAYERS']
        self.emb_size = params['EMB_SIZE']
        self.nhead = params['NHEAD']
        self.ffn_hid_dim = params['FFN_HID_DIM']
        self.batch_size = params['BATCH_SIZE']
        self.num_epochs = params['NUM_EPOCHS'] 

        self.src_vocab_size = len(vocab['src'])
        self.tgt_vocab_size = len(vocab['tgt'])


    def train(self, rank): # this should be passed as the first arg to mp.spawn()

        train_dataloader, valid_dataloader = self.ddp_dataloader()

        torch.cuda.set_device(rank) 
        dist.init_process_group(backend="nccl", world_size=self.n_gpus, rank=rank)

        model_ddp = self.prepare_ddp_model(rank)
        optimizer = torch.optim.Adam(model_ddp.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

        for epoch in range(1, self.num_epochs+1):
            start_time = timer()
            train_loss, model, optimizer = self.train_epoch(rank, model_ddp, optimizer, train_dataloader)

            end_time = timer()
            epoch_time = end_time - start_time
            val_loss = self.evaluate(rank, model_ddp, valid_dataloader, batch_size=self.batch_size)
            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(epoch_time):.3f}s"))

            if rank == 0:
                save_path = os.path.join(saved_path, f'ckpt_epoch{epoch}_loss_{train_loss}_vloss_{val_loss}_epoch_time_{epoch_time}.pt') ## revise this. 
                torch.save(model.state_dict(), save_path)
            dist.barrier()

            print('#############################')
            print('########### SAVED ###########')
            print('#############################')

        dist.destroy_process_group()
    
    def prepare_ddp_model(self, rank):

        model = Seq2SeqTransformer(self.num_encoder_layers, self.num_decoder_layers, self.emb_size, self.nhead, self.src_vocab_size, self.tgt_vocab_size, self.ffn_hid_dim)

        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        model = model.to(rank)
        model = DDP(model, device_ids=[rank])

        return model

    def ddp_dataloader(self):

        dataset = PairDataset(self.csv_path)

        train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1])

        train_sampler, valid_sampler = DistributedSampler(train_set), DistributedSampler(val_set)

        # train_loss, model, optimizer = train_epoch(transformer, optimizer, train_set, batch_size=params['BATCH_SIZE'])

        train_dataloader = DataLoader(train_set, batch_size=self.batch_size, sampler=train_sampler, collate_fn=collate_fn)
        valid_dataloader = DataLoader(val_set, batch_size=self.batch_size, sampler=valid_sampler, collate_fn=collate_fn)

        return train_dataloader, valid_dataloader


    # train_iter=을 함수 안에서 정의하는건 좀 아니지 않나?
    def train_epoch(self, rank, model, optimizer, dataloader):
        model.train()
        losses = 0
        # train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE)) ###
        # you fuckin sure this goes in here?
        # nah this should go wayyyy up here this should be given as an arg to train(). 
        train_dataloader = dataloader
        
        print('#############################################')
        print('########### Another epoch entered ###########')
        print('#############################################')

        for src, tgt in train_dataloader:
            src = src.to(rank)
            tgt = tgt.to(rank)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, rank)

            logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

            optimizer.zero_grad()

            tgt_out = tgt[1:, :]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            optimizer.step()
            losses += loss.item()

            train_loss = losses / len(train_dataloader)

        return train_loss, model, optimizer


    # revise this too it supports DDP. 
    def evaluate(self, rank, model, dataloader, batch_size):
        model.eval()
        losses = 0

        # val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE)) ###
        val_dataloader = dataloader

        for src, tgt in val_dataloader:
            src = src.to(rank)
            tgt = tgt.to(rank)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, rank)
            
            logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
            
            tgt_out = tgt[1:, :]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

        return losses / len(val_dataloader)