#libraries initialization 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def load_data() : 

  #open data
  df_sudoku = pd.read_csv("/sudoku-9m.csv")

  #Data type changing and reshaping, we keep 2.5 M data only 
  S_blanked_raw = df_sudoku['puzzle'].values[1:7800001:3]
  S_solved_raw = df_sudoku['solution'].values[1:7800001:3]


  #Delete Dataframe to avoid ram overexploitation 
  #del(df_sudoku)

  S_blanked = []
  S_solved = []

  #a table containing 9x9 matrices only composed of values n, n being the matrix position in the table
  one_hot_matrix_X = np.array([int(i/81) for i in range(81*10)]).reshape(10,9,9)

  for i in range(len(S_blanked_raw)):
    
    #we get a table containing 9x9 boolean matrices, n_th matrix shows placements of values n in the original sudoku
    x = np.array([int(j) for j in S_blanked_raw[i]]).reshape((9,9))
    x = np.equal(x,one_hot_matrix_X)

    #we make a minus 1, it will be usefull during output between target comparaison 
    y = np.array([int(j) for j in S_solved_raw[i]]).reshape((9,9))-1

    S_blanked.append(x)
    S_solved.append(y)

    if i%10000==0:
      print(i)

  #To avoid ram overexploitation 
  del(S_blanked_raw,S_solved_raw)
  
  #Train/validation/test splitting 
  X_train, X_test, Y_train, Y_test = train_test_split(S_blanked, S_solved, test_size=0.25, shuffle = False)
  del(S_blanked,S_solved)
  X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15, shuffle = False)
 
 
  return X_train, X_val, X_test, Y_train, Y_val, Y_test

  

