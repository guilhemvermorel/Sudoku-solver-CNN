import torch
import torch.nn as nn

from load_data import load_data, one_hot_matrix_X
from set_definition import Testset
from model import resnet11, load_model
from train_test_definition import test


#hyperparameters
batch_size = 1
gpu_device = torch.device('cuda')
model = resnet11(10)
loss_function = nn.CrossEntropyLoss()


if torch.cuda.is_available():
    model.to(gpu_device)
    

#load data
_, _, X_test, _, _, Y_test = load_data()

testset = Testset([i for i in range(len(X_test))])
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
del(testset)

#load model
load_model()

#test
loss,accuracy1,accuracy2,accuracy3 = test(gpu_device,batch_size,one_hot_matrix_X,model,test_loader,loss_function)
