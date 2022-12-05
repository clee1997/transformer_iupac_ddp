import random
import pandas as pd
from torch.utils.data import Dataset, IterableDataset

class PairIterableDataset(IterableDataset):
    def __init__(self, csv_path):
        self.data_path = csv_path
    
    def __iter__(self):
        iter_csv = pd.read_csv(self.data_path, usecols = ['iupac', 'iupac_noised'], iterator=True, chunksize=1)
        # iter_csv.reset_index(drop=True, inplace=True)

        for line in iter_csv:
            item = (line['iupac_noised'].item(), line['iupac_noised'].item())
            yield item


class PairDataset(Dataset):
    def __init__(self, csv_path, p=0.1):
        self.p = p # 0.1
        df = pd.read_csv(csv_path, usecols = ['iupac', 'iupac_noised'], skiprows=lambda x: x > 0 and random.random() >= self.p)
        self.df = df 
        self.df.reset_index(drop=True, inplace=True)

 
    def __len__(self):
        # return number of rows
        return len(self.df)
   
    def __getitem__(self, idx):

        item = (self.df['iupac_noised'][idx], self.df['iupac'][idx])
    
        return item # target이 original!
