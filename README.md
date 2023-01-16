# Résolution de sudokus avec un ResNet11-SPD : Modélisation des grilles comme des images à faible résolution

## Contexte
Le sudoku est un jeu de grille où le but est de remplir l’ensemble des cases avec une série de chiffres, tous différents, qui ne doivent jamais se trouver plus d’une
fois sur la même ligne, la même colonne ou la même zone. L’idée de ce rapport est de tester une approche de résolution des sudokus différente de celle qu'on a l'habitude de voir dans les papiers, en le modélisant comme une image 9x9 à faible résolution et d’utiliser une architecture de réseau de neurones à convolution adéquate :
un ResNet en remplaçant les blocs de pooling par des couches SPD-Conv. Une couche SPD-Conv est composée d’un bloc space-to-death suivi d’une convolution non-strided. En comparant les résultats obtenus avec ceux présents dans les papiers, il s’avère que les accuracys obtenus sont meilleurs pour une complexité plus faible.

Pour mieux comprendre le projet, je vous invite à lire [le rapport associé](\ResNet11-SPD_sudoku_solver.pdf).

Parler des métriques

## Versions 
`python` - `3.11.1` \
`numpy` - `1.24.1` \
`pandas` - `1.5.2` \
`pytorch` - `1.13.1` 

## Dataset
Le dataset utilisé contient 9 millions de sudokus avec leur solutions, et chaque sudoku ne contient qu'une unique solution. Il a été généré sur kaggle [ici](https://www.kaggle.com/datasets/rohanrao/sudoku).

## Description du modèle 
Le modèle utilisé est fortement inspiré du modèle [ResNet50-SPD](https://github.com/LabSAINT/SPD-Conv), lui même inspiré par le modèle du ResNet50 mais adapté aux images de faibles dimensions. Il a alors été modifié pour correspondre à des images de dimension impair, comme c'est le cas pour les sudokus, et sa complexité a été réduite pour un temps de calcul plus rapide. On obtient un modèle de ResNet11-SPD. 

## Description des fichiers
`load_data.py` est une fonction qui ouvre le fichier de données, les transforme pour qu'elles soient utilisable par le modèle et les renvoie partagé en données d'entraînement, d'évaluation et de test. \
`set_definition.py` défini les ensembles d'entraînement, d'évaluation et de test pour que les données soient itérables dans la boucle d'entraînement et de test. \
`model.py` contient le modèle ResNet11-SPD. \
`train_test_definition.py` défini les foctions d'entraînement, d'évaluation et de test, ainsi que les 3 métriques utilisées pour évaluer le modèle. \
`train.py` est utilisé pour l'entraînement et l'évaluation durant l'entraînement. \
`test.py` est utilisé pour le test avec l'approche humaine (cf rapport projet). 

## Entraînement 
Etape 1 : télécharger le dataset [ici](https://www.kaggle.com/datasets/rohanrao/sudoku).
Etape 2 : lancer l'entraînement du fichier `train.py` ou télécharger directement [le fichier entraîné](https://www.dropbox.com/scl/fo/y6xfwpa7zgkjrb5xgejnd/h?dl=0&rlkey=nosi1f0r57hbp09vjb86oopqr).

## Tests
Lancer directement le fichier `test.py`

## Métriques d'évaluation
Il existe 3 métriques différentes pour l'évaluation du modèle : 
- `accuracy1` : qui correspond au nombre de cases (initialement vides ou pleines) qui ont été correctement identifié par le modèle divisé par le nombre de cases totale. 
- `accuracy2` : qui correspond au nombre de cases vides qui ont été correctement identifié par le modèle divisé par le nombre de cases vides totale. 
- `accuracy3` : qui correspond au nombre de sudokus entièrement résolus sur le nombre de sudokus 

## Résultats
![accuracy1](\accuracy1_sudoku_9m.png) \
![accuracy2](\accuracy2_sudoku_9m.png) \
![accuracy3](\accuracy3_sudoku_9m.png) \
Pour le test, on utilise une approche différente que pour l'évaluation, une approche plus humaine, on fait tourner le réseau sur une seule case vide à la fois : 
- `accuracy1 = 0.9545` 
- `accuracy2 = 0.9226` 
- `accuracy3 = 0.0885`

## Références 
Mon projet est beaucoup inspiré des travaux de Raja Sunkara et Tie Luo sur le sujet [No More Strided Convolutions or Pooling: A New CNN Building Block for Low-Resolution Images and Small Objects](https://arxiv.org/pdf/2208.03641v1.pdf)

 
