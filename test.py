import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from load_data import load_data
from set_definition import Testset
from model import resnet11, load_model
from train_test_definition import test


lr = 0.001
n_epochs = 100
batch_size = 512
weight_decay = 0.00001
gpu_device = torch.device('cuda')
model = resnet11(10)


if torch.cuda.is_available():
    model.to(gpu_device)
    
    
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)

_, _, X_test, _, _, Y_test = load_data()

testset = Testset([i for i in range(len(X_test))])
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
del(testset)

load_model()


loss,accuracy1,accuracy2,accuracy3 = test(gpu_device,batch_size)
