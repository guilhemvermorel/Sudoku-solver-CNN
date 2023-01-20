import time
import numpy as np
import torch



def train(epoch, gpu_device,batch_size,model,train_loader,optimizer,loss_function,n_epochs):


    start = time.time()
    model.train()

    for index_batch, (sudoku,target) in enumerate(train_loader):
      
      sudoku = torch.Tensor(np.array(sudoku))
      #we get a table containing 9x9 boolean matrices, i_th line of n_th matrix shows the value of the sudoku in position (i,n),
      #value equal to the position of the value 1 on this line plus 1. 
      target = torch.Tensor(np.eye(9)[target])

      if torch.cuda.is_available():
        sudoku = sudoku.to(gpu_device)
        target = target.to(gpu_device)
      
      if len(sudoku) < batch_size :
        continue
      
      #zero the parameter gradients
      optimizer.zero_grad()

      #forward
      outputs = model(sudoku).view(batch_size,9,9,9)

      #backward
      loss = loss_function(outputs, target)
      loss.backward()

      #weight update
      optimizer.step()


    print('Training Epoch: {epoch} \tLoss: {:0.4f}\tLR: {:0.6f}'.format(
          loss.item(),
          optimizer.param_groups[0]['lr'],
          epoch=epoch,
          trained_samples=(index_batch+1)*batch_size*epoch ,
          total_samples=len(train_loader)*batch_size*n_epochs
        ))  
    

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))



def eval(epoch, eval_type,gpu_device,batch_size,model,one_hot_matrix_X,train_loader,validation_loader,loss_function):

    start = time.time()
    model.eval()

    test_loss = 0.0 # cost function error
    correct1 = 0.0
    correct2 = 0.0
    correct3 = 0.0
    n_zeros = 0.0
    M_one_hot_x= np.array([one_hot_matrix_X for i in range(batch_size)])

    #choice of dataloader
    if eval_type == "train":
      loader = train_loader

    else :
      loader = validation_loader


    with torch.no_grad():  # torch.no_grad for TESTING
      for index_batch, (sudoku,target) in enumerate(loader):

          sudoku = torch.Tensor(np.array(sudoku))
          target = torch.Tensor(np.eye(9)[target])


          if torch.cuda.is_available():
              sudoku = sudoku.to(gpu_device)
              target = target.to(gpu_device)

          if len(sudoku) < batch_size :
            continue
          
          #forward
          outputs = model(sudoku).view(batch_size,9,9,9)

          #calcul of loss
          loss = loss_function(outputs, target)
          test_loss += loss.item()

          _,preds = outputs.max(3)
          _,targets = target.max(3)

          #accuracy1 calcul : number of values equal
          correct1 += preds.eq(targets).sum().item()

          #get back the original sudoku shape
          input = np.sum(sudoku.to("cpu").numpy()*M_one_hot_x,axis=1)

          #mask of non-blanked values
          mask = input != np.zeros((batch_size,9,9))
          
          #number of blanked values on the sudokus
          n_zeros += 81*batch_size-mask.sum()

          #accuracy2 calcul : number of initial blanked values equal
          correct2 += np.equal(np.ma.masked_array(preds.to("cpu").numpy(), mask),
                               np.ma.masked_array(targets.to("cpu").numpy(), mask)).sum()
          

          #accuracy3 calcul : number of sudokus entirely equal
          correct3 += (np.sum(np.sum(preds.eq(targets).to("cpu").numpy(),axis=2),axis=1)//81).sum()


    loss = test_loss / (index_batch*batch_size)
    accuracy1 = correct1 / ((index_batch*batch_size)*81)
    accuracy2 = correct2 / n_zeros
    accuracy3 = correct3 / ((index_batch*batch_size))

    finish = time.time()
    
    if eval_type == "train" : 
      print('Evaluating Network.....')
      print('Train set: Epoch: {}, Average loss: {:.4f}, Accuracy1: {:.4f}, Accuracy2: {:.4f}, Accuracy3: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        loss,
        accuracy1,
        accuracy2,
        accuracy3,
        finish - start
    ))


    else : 
      print('Valid set: Epoch: {}, Average loss: {:.4f}, Accuracy1: {:.4f}, Accuracy2: {:.4f}, Accuracy3: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        loss,
        accuracy1,
        accuracy2,
        accuracy3,
        finish - start
    ))
      


    return loss, accuracy1, accuracy2, accuracy3
  
  
  

def test(gpu_device,batch_size,one_hot_matrix_X,model,test_loader,loss_function):

    M_one_hot_x= np.array([one_hot_matrix_X for i in range(batch_size)])
    start = time.time()
    model.eval()

    test_loss = 0.0 # cost function error
    correct1 = 0.0
    correct2 = 0.0
    correct3 = 0.0
    n_zeros = 0.0


    with torch.no_grad():  # torch.no_grad for TESTING

      for index_batch, (sudoku,target) in enumerate(test_loader):

          if index_batch%10 == 0 : print(index_batch)

          input = np.array(sudoku)
          sudoku = torch.Tensor(input)
          target = torch.Tensor(np.eye(9)[target])
          preds = torch.zeros(batch_size,9,9)

        
          if torch.cuda.is_available():
              sudoku = sudoku.to(gpu_device)
              target = target.to(gpu_device)
              preds = preds.to(gpu_device)

          if len(sudoku) < batch_size :
            continue
          


          #while inputs contain zero values
          while sudoku[:,0].sum().item() > 0 :
            
            #forward 
            outputs = model(sudoku).view(batch_size,9,9,9)

            #get back the original sudoku shape
            sudoku_count0 = np.sum(sudoku.to("cpu").numpy()*M_one_hot_x,axis=1)

            #mask of non-blanked values
            mask = (sudoku_count0 != np.zeros((batch_size,9,9))).reshape(batch_size,9*9)

            #mask reshaped along one axe and multiplied by 9
            mask_reshaped = [[mask[j][i//9] for i in range(9*9*9)] for j in range(batch_size)]

            #we get a place in the list of the output maximum value, corresponding to the value whose algorithm is most certain
            best_outputs = np.argmax(np.ma.masked_array(outputs.to("cpu").numpy().reshape((batch_size,9*9*9)),mask_reshaped),axis = 1)

            #we retrieve lignes, columns and values associated
            lignes = best_outputs//81
            columns = best_outputs%81//9
            values = best_outputs%9+1

            #we can change values in the corresponding lignes and columns of the sudoku
            for i in range(len(values)):
              if values[i] > 0 :
                sudoku[i,values[i],lignes[i],columns[i]]= True
                sudoku[i,0,lignes[i],columns[i]]= False


          #final prediction 
          for i in range(batch_size):
            for j in range(0,10):
              preds[i] += sudoku[i,j]*j

          #calcul of loss
          loss = loss_function(outputs, target)
          test_loss += loss.item()

          _,targets = target.max(3)
          targets = torch.add(targets,1)

          #accuracy1 calcul : number of values equal
          correct1 += preds.eq(targets).sum().item()



          #get back the original sudoku shape
          input = np.sum(input*M_one_hot_x,axis=1)

          #mask of non-blanked values
          mask2 = input != np.zeros((batch_size,9,9))
          
          #number of blanked values on the sudokus
          n_zeros += 81*batch_size-mask2.sum()

          #accuracy2 calcul : number of initial blanked values equal
          correct2 += np.equal(np.ma.masked_array(preds.to("cpu").numpy(), mask2),
                               np.ma.masked_array(targets.to("cpu").numpy(), mask2)).sum()


          

          #accuracy3 calcul : number of sudokus entirely equal
          correct3 += (np.sum(np.sum(preds.eq(targets).to("cpu").numpy(),axis=2),axis=1)//81).sum()
        

    loss = test_loss / (index_batch*batch_size)
    accuracy1 = correct1 / ((index_batch*batch_size)*81)
    accuracy2 = correct2 / n_zeros
    accuracy3 = correct3 / ((index_batch*batch_size))

    finish = time.time()

    
    
    print('Testing Network.....')
    print('Test set: Average loss: {:.4f}, Accuracy1: {:.4f}, Accuracy2: {:.4f}, Accuracy3: {:.4f}, Time consumed:{:.2f}s'.format(
      loss,
      accuracy1,
      accuracy2,
      accuracy3,
      finish - start
  ))




    return loss, accuracy1, accuracy2, accuracy3
  
