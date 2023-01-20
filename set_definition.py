import torch
from train import X_train, X_val, Y_train, Y_val
from test import X_test,Y_test


class Trainset(torch.utils.data.Dataset):
  def __init__(self, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = X_train[ID]
        y = Y_train[ID]
        return X,y

class Testset(torch.utils.data.Dataset):
  def __init__(self, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self,index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = X_test[ID]
        y = Y_test[ID]
        return X,y

class Validationset(torch.utils.data.Dataset):
  def __init__(self, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = X_val[ID]
        y = Y_val[ID]
        return X,y
