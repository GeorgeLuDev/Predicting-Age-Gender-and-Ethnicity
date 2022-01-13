from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

# DataLoader
class PeopleDataset(Dataset):

  def __init__(self, df):
    self.df = df
    self.x = df['pixels'].apply(lambda x:  np.array(x.split(), dtype="float32")).apply(lambda x:  x.reshape(1,48,48)).values
    self.y = df.iloc[:, 0:3].values # 0-age, 1-ethnicity, 2-gender
    self.n_samples = df.shape[0]

  def __getitem__(self, index):
    return self.x[index], self.y[index]

  def __len__(self):
    return self.n_samples