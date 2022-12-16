import time
import numpy as np
import torch
import torch.nn as nn



def train(epoch, gpu_device,batch_size):

    start = time.time()
    model.train()

    for index_batch, (sudoku,target) in enumerate(train_loader):

      sudoku = torch.Tensor(np.array(sudoku))
      target = torch.Tensor(np.eye(9)[target])

      if torch.cuda.is_available():
        sudoku = sudoku.to(gpu_device)
        target = target.to(gpu_device)
      
      if len(sudoku) < batch_size :
        continue

      optimizer.zero_grad()
      outputs = model(sudoku).view(batch_size,9,9,9)

      loss = loss_function(outputs, target)
      loss.backward()
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



def eval(epoch, eval_type,gpu_device,batch_size):

    start = time.time()
    model.eval()

    test_loss = 0.0 # cost function error
    correct1 = 0.0
    correct2 = 0.0
    correct3 = 0.0
    n_zeros = 0.0
    M_one_hot_x= np.array([one_hot_matrix_X for i in range(batch_size)])

    if eval_type == "train":
      loader = train_loader

    else :
      loader = validation_loader


    with torch.no_grad():  # torch.no_grad for TESTING
      for index_batch, (sudoku,target) in enumerate(loader):
          sudoku = torch.Tensor(np.array(sudoku))
          target = torch.Tensor(np.eye(9)[target])
          # Prend un GPU par default
          if torch.cuda.is_available():
              sudoku = sudoku.to(gpu_device)
              target = target.to(gpu_device)

          if len(sudoku) < batch_size :
            continue

          outputs = model(sudoku).view(batch_size,9,9,9)
          loss = loss_function(outputs, target)

          test_loss += loss.item()
          _,preds = outputs.max(3)
          _,targets = target.max(3)


          correct1 += preds.eq(targets).sum().item()

          #input = ((sudoku.to("cpu").numpy()+0.5)*9).astype(int)
          input = np.sum(sudoku.to("cpu").numpy()*M_one_hot_x,axis=1)
          mask = input != np.zeros((batch_size,9,9))
          
          n_zeros += 81*batch_size-mask.sum()


          correct2 += np.equal(np.ma.masked_array(preds.to("cpu").numpy(), mask),
                               np.ma.masked_array(targets.to("cpu").numpy(), mask)).sum()
          

        
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
  
  
  

def test(gpu_device,batch_size):

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

          # Prend un GPU par default
          if torch.cuda.is_available():
              sudoku = sudoku.to(gpu_device)
              target = target.to(gpu_device)
              preds = preds.to(gpu_device)

          if len(sudoku) < batch_size :
            continue
          


          #while les inputs contiennent des 0 
          while sudoku[:,0].sum().item() > 0 :
            
            #on sort les outputs, 
            outputs = model(sudoku).view(batch_size,9,9,9)

            sudoku_count0 = np.sum(sudoku.to("cpu").numpy()*M_one_hot_x,axis=1)
            mask = (sudoku_count0 != np.zeros((batch_size,9,9))).reshape(batch_size,9*9)

            mask_reshaped = [[mask[j][i//9] for i in range(9*9*9)] for j in range(batch_size)]
            #on change les outputs avec les valeurs différentes de 0 dans l'inputs maské
            best_outputs = np.argmax(np.ma.masked_array(outputs.to("cpu").numpy().reshape((batch_size,9*9*9)),mask_reshaped),axis = 1)


            lignes = best_outputs//81
            colonnes = best_outputs%81//9
            valeurs = best_outputs%9+1

            #changer les inputs au emplacement de ligne voulu et colonne voulu 
            for i in range(len(valeurs)):
              if valeurs[i] > 0 :
                sudoku[i,valeurs[i],lignes[i],colonnes[i]]= True
                sudoku[i,0,lignes[i],colonnes[i]]= False


          for i in range(batch_size):
            for j in range(0,10):
              preds[i] += sudoku[i,j]*j


          loss = loss_function(outputs, target)

          test_loss += loss.item()

          _,targets = target.max(3)
          targets = torch.add(targets,1)


          correct1 += preds.eq(targets).sum().item()



          #input = ((sudoku.to("cpu").numpy()+0.5)*9).astype(int)
          input = np.sum(input*M_one_hot_x,axis=1)
          mask2 = input != np.zeros((batch_size,9,9))
          
          
          n_zeros += 81*batch_size-mask2.sum()

          correct2 += np.equal(np.ma.masked_array(preds.to("cpu").numpy(), mask2),
                               np.ma.masked_array(targets.to("cpu").numpy(), mask2)).sum()


          

        
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
  
