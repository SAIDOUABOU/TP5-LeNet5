#module toolsmethods

#importation
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors

from sklearn.metrics import (accuracy_score , precision_score , recall_score , f1_score , confusion_matrix , ConfusionMatrixDisplay ,
classification_report)
from sklearn.model_selection import  StratifiedKFold

import os
from typing import Any
import pickle    # sert à Sauvegarder un modèle et le redéployer après

"""
    cette classe offre des fonctions sous forme de méthodes de classe necessaires à faire une variétés de tâches
"""


class ToolsMethods:

        @classmethod
        def crossValWithStratifiedBestParams(self , layer_sizes: list[int] , learning_rates: list[float] ,  X: np.ndarray , y: np.ndarray , _lambdas: list[float] , n_splits: int=5 , shuffle: bool= True , random_state: int= None) -> tuple[dict , dict , dict]:
            #elle entraîne le modèle su k-1 folds et l'évaluer sur le fold restant à chaque itération pour chaque combinaison des hyper-paramètres
            # [learning rate et coefficient de régularisation] et retourne les accuracies de train, de validation et les best paramètres
            #validation des entrées 
            assert ( 
                       isinstance(layer_sizes , list) and all(isinstance(element , int) and (element > 0) for element in layer_sizes) and 
                       len( layer_sizes ) >= 2 
                   ), " layer_sizes 'arg1' doit être une liste de moins deux entiers strictement positifs !"
                  
            assert (
                    isinstance( learning_rates , list) and np.all(isinstance(rate , (float , np.integer , int)) and 
                                                                  rate > 0 for rate in  learning_rates)
                     ) , " Learning rates 'arg2' doit être une liste de nombres positifs !"
            assert (
                    isinstance( _lambdas , list) and np.all(isinstance(coef , (float , np.integer , int)) and 
                                                                  coef > 0 for coef in  _lambdas)
                     ) , " coefs. de régularisation 'arg5' doit être une liste de nombres positifs !"
            assert ( 
                       isinstance(X , np.ndarray) and all(isinstance(row, np.ndarray) for row in X) and 
                       all(isinstance(number, float)  for row in X for number in row ) 
                   ), "l'arg3 doit être une matrice de réels !"
                  
            assert ( 
                       isinstance(y , np.ndarray) and np.all(isinstance( number , (float , int , np.integer)) for number in y) 
                    ), "l'arg4 doit être un tableau d'entiers ou de réels !"
                  
            assert isinstance(n_splits , int) and n_splits > 0 , "le nombre de folds 'arg6' 'n_splits' doit être un entier postif non nul !"
            assert isinstance(shuffle , bool) , "arg7 doit être un booleeen (il est par défaut True) !"
            assert isinstance(random_state , int) , "l'agr8 'random_state' doit être un entier (il est par défaut None) !"
            
            #instanciation de 'StratifiedKFold'
            cv = StratifiedKFold(n_splits= n_splits , shuffle= shuffle , random_state= random_state)
            #scores de la validation croisée
            scores_train = {}  
            scores_val = {} 
        
            #les meilleurs hyperparamètres: [learning rate , coefficient de régularisation] 
            best_params = {}
            
            #initialisation de la meilleurs accuracy de validation qui sert à la comparaison
            best_accuracy_val = 0
        
            for rate in learning_rates:
                for _lambda in _lambdas:
                    #validation croisée
                    for i , (train_indices , val_indices) in enumerate(cv.split(X , y) , 1):
                        X_train = X[train_indices] 
                        X_val   = X[val_indices]
                        y_train = y[train_indices]
                        y_val   = y[val_indices]
                    
                        #création du modèle 
                        model = NeuralNetwork(layer_sizes= layer_sizes , learning_rate= rate)
                        #Entraînement du modèle avec SGD (Batch= 32 , epochs= 32)
                        model.train(X_train , y_train , X_val, y_val , epochs= 100 , batch_size= 32 , _lambda= _lambda)
                        #prédiction sur le K-fold de validation les k-1 de train  calcul de l'accuracy
                        y_pred_train = model.predict(X_train)
                        y_pred_val = model.predict(X_val)
                
                        accuracy_train = model.compute_accuracy(y_train , y_pred_train)
                        accuracy_val   = model.compute_accuracy(y_val , y_pred_val)
        
                        #capturer les meilleurs hyper-paramètres
                        if accuracy_val > best_accuracy_val:
                            best_accuracy_val = accuracy_val.copy()
                            best_params['alpha'] = rate
                            best_params['lambda'] = _lambda
                            
                        #ajout de score 'accuracy' à la liste scores
                        scores_train.setdefault(f'Alpha{rate}', {}).setdefault(f'lamda{_lambda}', {})[f"Train folds{i}"] = accuracy_train
                        scores_val.setdefault(f'Alpha{rate}', {}).setdefault(f'lamda{_lambda}', {})[f"Validation fold{i}"] = accuracy_val
                    
            #validation de résultat à retourner
            assert (
                isinstance(scores_val , dict)  
                    ) , "dictionnaire scores_val contient des valeurs non réelles !"
            assert (
                isinstance(scores_train , dict) 
                    ) , "dictionnaire scores_train contient des valeurs non réelles !"
        
            assert (
                isinstance(best_params , dict) 
            ) , "Le dictionnaire des best_params ne contient pas des valeurs réelles !" 
            
            return scores_train , scores_val , best_params

        @classmethod
        def crossValidationWithStratifiedKFold(self , layer_sizes: list[int] , learning_rate: float ,  X: np.ndarray , y: np.ndarray , _lambda: float= 0.01 ,
                                               n_splits: int=5 , shuffle: bool= True , random_state: int= None) -> tuple[dict , dict]:
            #elle entraîne le modèle su k-1 folds et l'évaluer sur le fold restant à chaque itération et retourne les accuracies des validations
            #validation des entrées 
            assert ( 
                       isinstance(layer_sizes , list) and all(isinstance(element , int) and (element > 0) for element in layer_sizes) and 
                       len( layer_sizes ) >= 2 
                   ), " layer_sizes 'arg1' doit être une liste de moins deux entiers strictement positifs !"
                  
            assert isinstance( learning_rate , (int , float )) and learning_rate > 0, " Learning rate 'arg2' doit être un nombre positif !"
            assert isinstance( _lambda , (int , float , np.integer )) and _lambda >= 0, " Coef. de régularisation L2 'arg5' doit être un nombre positif !"
            assert ( 
                       isinstance(X , np.ndarray) and all(isinstance(row, np.ndarray) for row in X) and 
                       all(isinstance(number, float)  for row in X for number in row ) 
                   ), "l'arg3 doit être une matrice de réels !"
                  
            assert ( 
                       isinstance(y , np.ndarray) and np.all(isinstance( number , (float , int , np.integer)) for number in y) 
                    ), "l'arg4 doit être un tableau d'entiers ou de réels !"
                  
            assert isinstance(n_splits , int) and n_splits > 0 , "le nombre de folds 'arg6' 'n_splits' doit être un entier postif non nul !"
            assert isinstance(shuffle , bool) , "arg7 doit être un booleeen (il est par défaut True) !"
            assert isinstance(random_state , int) , "l'agr8 'random_state' doit être un entier (il est par défaut None) !"
            
            #instanciation de 'StratifiedKFold'
            cv = StratifiedKFold(n_splits= 5 , shuffle= True , random_state= 35)
            #scores de la validation croisée
            scores_train = {}  
            scores_val = {} 
        
            
            #validation croisée
            for i , (train_indices , val_indices) in enumerate(cv.split(X , y) , 1):
                X_train = X[train_indices] 
                X_val   = X[val_indices]
                y_train = y[train_indices]
                y_val   = y[val_indices]
            
                #création du modèle 
                model = NeuralNetwork(layer_sizes= layer_sizes , learning_rate= learning_rate)
                #Entraînement du modèle avec SGD (Batch= 32 , epochs= 32)
                model.train(X_train , y_train , X_val, y_val , epochs= 100 , batch_size= 32 , _lambda= _lambda)
                #prédiction sur le K-fold de validation les k-1 de train  calcul de l'accuracy
                y_pred_train = model.predict(X_train)
                y_pred_val = model.predict(X_val)
        
                accuracy_train = model.compute_accuracy(y_train , y_pred_train)
                accuracy_val   = model.compute_accuracy(y_val , y_pred_val)
        
                #ajout de score 'accuracy' à la liste scores
                scores_train[f"Train folds{i}"] = accuracy_train
                scores_val[f"Validation fold{i}"] = accuracy_val
            #validation de résultat à retourner
            assert (
                isinstance(scores_val , dict) and all(isinstance(number , float) for number in scores_val.values())  
                    ) , "dictionnaire scores_val contient des valeurs non réelles !"
            assert (
                isinstance(scores_train , dict) and all(isinstance(number , float) for number in scores_train.values())  
                    ) , "dictionnaire scores_train contient des valeurs non réelles !"
            
            return scores_train , scores_val
    
        @classmethod
        def multiCurvesGraph(self , courbeList: list[list , ...] , labelsList: list[str , ...] , titre: str , colorslist: list[str , ...]= ['red' , 'blue']):
            # affichage de n courbes dans un seul plot 
            # validation des entrées
            assert isinstance(courbeList , list) and all(isinstance(curve , list) for curve in courbeList) , "Arg1 doit être une liste de liste!"
            assert isinstance(labelsList, list) and all(isinstance(label, str) for label in labelsList) , "Arg2 doit être une liste de string !"
            assert isinstance(titre , str) , "Arg3 doit être une string (titre du graphe)!"
            assert isinstance(colorslist , list) and all(isinstance(color , str) and color in mcolors.CSS4_COLORS.keys() for color in colorslist) , (
               "Arg4 doit être une liste de coleurs valides!"
            )
            assert len(courbeList) == len(labelsList) == len(colorslist) , (
                "Il faut un label et une couleurs à chaque courbe: longueur(arg1) est différent de longueur(arg2)!"
            )
            
            plt.figure(figsize=(15, 4))
            
            for curve , label , color in  zip(courbeList , labelsList , colorslist ):
                plt.plot( curve , color= color , label= label)
                
            plt.title(titre , fontsize=12)
            plt.xlabel("epochs", fontsize=10)
            
            if "loss" in labelsList[0].lower():
                plt.ylabel("Loss", fontsize=10)
            else:
                plt.ylabel("Accuracy", fontsize=10)
                
            plt.legend(fontsize=10)
            plt.grid(which= 'major' , alpha= 0.7 , linestyle= '--')
            # plt.savefig("courbes_Adam.png" , dpi=300 , bbox_inches='tight')
            plt.show()
            
    
        @classmethod
        def multiConfusionMatrices(self , listOfConfusionMatrix: list[np.ndarray , ...] , listOfClassLabels: list[list , ...] , listOftitle= list[str , ...]):
            #cette fonction affiche une matrice de confusion ou plusieurs sur la même figure
            #validation des entrées
            assert len(listOfConfusionMatrix)  == len(listOftitle) , "Les args 1 et 3 doivent avoir la même taille!"
            assert( isinstance(listOfConfusionMatrix, list) and all(isinstance(subList, np.ndarray) for subList in listOfConfusionMatrix)
                             and all(isinstance(number, (int , np.integer))  for subList in listOfConfusionMatrix for row in subList for number in row)
                  ) , "l'arg1 doit être une liste de matrice contenant uniquement  des entiers  !"  
            assert( isinstance(listOfClassLabels, list) and all(isinstance(subList, list) for subList in listOfClassLabels)
                            and all(isinstance(name , (str , int , np.integer)) for subList in listOfClassLabels for name in subList) 
                  ) , "l'arg2 doit être une liste de listes contenant uniquement labels de classes !"
            assert( isinstance(listOftitle, list) and all(isinstance(subList, str) for subList in listOftitle) 
                  ) , "l'arg3 doit être une liste de String (titres des matrices de confusion) !"
        
            #graphe dynamique
            numberOfColumns= 2
            numberOfRows = (len(listOfConfusionMatrix) + 1) // numberOfColumns
        
            plt.figure(figsize=(15*numberOfRows , 15*numberOfRows))
        
            for i , (matrix , labels , title) in enumerate(zip(listOfConfusionMatrix , listOfClassLabels , listOftitle) , 1):
                    
                plt.subplot( numberOfRows , numberOfColumns , i) 
        
                confusionMatrix = ConfusionMatrixDisplay(confusion_matrix= matrix , display_labels= labels)
                confusionMatrix.plot(cmap='Blues', values_format='d' , ax= plt.gca() , colorbar= False)
                
                plt.title(title , fontsize=14 , fontweight= 'bold')
            
            plt.tight_layout()  
            # plt.savefig("matrices_Adam.png", dpi=300, bbox_inches='tight')
            plt.show()
            

        @classmethod
        def save_model(self , model: object , fileName: str= "model.pkl") -> None:
            """
            Elle sauvegarde un modèle CNN entraînné: les poids et les biais ainsi que ça structure sous forme d'un fichier .pkl.
            """
            #validation des entrées
            assert isinstance(model , object) , "L'argument 1 'model' est de type 'Object' !"
            assert isinstance(fileName , str) and fileName.endswith('.pkl') , \
                                                    " L'argument 2 'fileName' doit être un nom de fichier de type .pkl sous forme d'une String !"
            
            #variables de stockage des paramètres du modèle sous forme d'une liste de dictionnaires et de sa structure 
            parametres = []
            structure  = []
        
            #si le modèle est de type PersonalizedLeNet5
            if type(model).__name__ == 'PersonalizedLeNet5':
                for layer in model.layers:
                    if layer.trainable:
                        parametres.append({'weights': layer.weights , 'biases': layer.biases })
                        structure.append(type(layer).__name__ )
                    else:
                        parametres.append(None)
                        structure.append(type(layer).__name__ )
        
            """
            Code de sauvegarde d'autres type de modèle à ajouter ici
            
            """
            
            #création de fichier sauvegardant le modèle            
            with open(fileName , 'wb') as f:
                pickle.dump({'structure' : structure , 'parametres' : parametres} , f)
                
            # le chemin vers le fichier sauvegadant le modèle
            path = os.path.abspath(fileName)
            print(f" Modèle sauvegardé avec succès dans le fichier '{path}'.")
        
        
        @classmethod
        def load_model(self , model: object , fileName: str) -> Any:
            """
            Elle recharge le modèle (poids , biais , et sa structure) sauvegardé dans le fichier .pkl dans le canevas 'model' passé en argument
            et retourne le modèle récupéré.
            """
            #validation des entrées
            assert isinstance(model , object) , "L'argument 1 'model' est de type object !"
            assert isinstance(fileName , str) and fileName.endswith('.pkl') , \
                                                    " L'argument 2 'fileName' doit être un nom de fichier de type .pkl sous forme d'une String !"
            #vérification de l'existence du fichier
            assert os.path.exists(fileName) , f"Le fichier {filename} est introuvable !"
        
            #chargement du contenu du fichier .pkl
            with open(fileName , 'rb') as f:
                content = pickle.load(f)
            
            structure  = content['structure']
            parametres = content['parametres']
            
            #vérification de compatibilité entre le modèle voulu et celui récupéré (nombre de couches)
            assert len(model.layers) == len(structure) , "La structure du modèle incompatible avec les paramètres chargés !"
            
            #rechargement des poids et biais selon le type du modèle
            if type(model).__name__ == 'PersonalizedLeNet5':    #si le modèle est de type PersonalizedLeNet5
                for i , (layer , structure_name , param) in enumerate(zip(model.layers , structure , parametres)):
                    if getattr(layer , 'trainable', False):
                        #vérification la correspndance entre type attendu et récupéré de la couche
                        assert type(layer).__name__ == struct_name , \
                                                f"Incohérence de structure à la couche {i}: attendu {structure_name}, obtenu {type(layer).__name__}"
                        layer.weights = parametres['weights']
                        layer.biases  = parametres['biases']
        
            """
            Code de traîtement des autres type de modèles à ajouter ici
            
            """
            
            #chemin du fichier de récupération 
            path = os.path.abspath(fileName)
            print(f"Le modèle est chargé avec succès depuis : {path}")

            
            return model
        
