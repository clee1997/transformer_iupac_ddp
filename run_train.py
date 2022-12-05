import torch
import torch.multiprocessing as mp

from trainer import Trainer



ngpus = torch.cuda.device_count()


assert (ngpus >= 2), 'Less than 2 gpus available'

trainer = Trainer()

# def train(self, rank):
mp.spawn(trainer.train, nprocs=ngpus, join=True)