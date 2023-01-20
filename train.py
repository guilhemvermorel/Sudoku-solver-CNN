import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from load_data import load_data, one_hot_matrix_X
from set_definition import Trainset, Validationset
from model import resnet11, load_model, save_model
from train_test_definition import train, eval



#hyperparameters
lr = 0.001
n_epochs = 100
batch_size = 512
weight_decay = 0.00001
gpu_device = torch.device('cuda')
model = resnet11(10)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)

if torch.cuda.is_available():
    model.to(gpu_device)



#list to plot 
epochs=[i for i in range(n_epochs)]
train_loss = []
valid_loss = []
train_accuracy1 = []
train_accuracy2 = []
train_accuracy3 = []

valid_accuracy1 = []
valid_accuracy2 = []
valid_accuracy3 = []



#load data
X_train, X_val, _ , Y_train, Y_val, _ = load_data()

trainset = Trainset([i for i in range(len(X_train))])
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
del(trainset)

validationset = Validationset([i for i in range(len(X_val))])
validation_loader = torch.utils.data.DataLoader(validationset, batch_size=batch_size, shuffle=True)
del(validationset)


#we can load saved model and lists associated
depoch = 0 
#epochs,train_loss,valid_loss,train_accuracy1,train_accuracy2,train_accuracy3,valid_accuracy1,valid_accuracy2,valid_accuracy3,
#                                                                   depoch,loss_function = load_model(model,optimizer,n_epochs)

for epoch in range(depoch + 1, n_epochs + 1):
   
    #train
    train(epoch, gpu_device,batch_size,model,train_loader,optimizer,loss_function,n_epochs)

    #trainset evaluation model
    loss,accuracy1,accuracy2,accuracy3 = eval(epoch,"train",gpu_device,batch_size,model,one_hot_matrix_X,
                                                train_loader,validation_loader,loss_function)
    train_loss.append(loss)
    train_accuracy1.append(accuracy1)
    train_accuracy2.append(accuracy2)
    train_accuracy3.append(accuracy3)

    #validset evaluation model
    loss,accuracy1,accuracy2,accuracy3 = eval(epoch,"valid",gpu_device,batch_size,model,one_hot_matrix_X,
                                                train_loader,validation_loader,loss_function)
    valid_loss.append(loss)
    valid_accuracy1.append(accuracy1)
    valid_accuracy2.append(accuracy2)
    valid_accuracy3.append(accuracy3)

    #scheduler.step()
    if epoch%5 == 0 : 
      save_model(epoch,model,optimizer,loss_function,train_loss,valid_loss,train_accuracy1,train_accuracy2,train_accuracy3,
                valid_accuracy1,valid_accuracy2,valid_accuracy3)

      
#plot     
plt.figure()
plt.plot(epochs,train_loss,label='Train loss')
plt.plot(epochs,valid_loss,label='Validation loss')
plt.title('Train and Validation loss for lr = 0.001')
plt.legend()
plt.grid()
plt.savefig('./loss.png')


plt.figure()
plt.plot(epochs,train_accuracy1,label='Train accuracy')
plt.plot(epochs,valid_accuracy1,label='Validation accuracy')
plt.title('Train and Validation accuracy for lr = 0.001')
plt.legend()
plt.grid()
plt.savefig('./accuracy1.png')

plt.figure()
plt.plot(epochs,train_accuracy2,label='Train accuracy')
plt.plot(epochs,valid_accuracy2,label='Validation accuracy')
plt.title('Train and Validation accuracy for lr = 0.001')
plt.legend()
plt.grid()
plt.savefig('./accuracy2.png')

plt.figure()
plt.plot(epochs,train_accuracy3,label='Train accuracy')
plt.plot(epochs,valid_accuracy3,label='Validation accuracy')
plt.title('Train and Validation accuracy for lr = 0.001')
plt.legend()
plt.grid()
plt.savefig('./accuracy3.png')


