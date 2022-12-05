import torch
import torch.multiprocessing as mp

from trainer import Trainer



ngpus = torch.cuda.device_count()


assert (ngpus >= 2), 'Less than 2 gpus available'

trainer = Trainer()

# def train(self, rank):
if __name__ == '__main__':
    mp.spawn(trainer.train, nprocs=ngpus, join=True)