#module neuralnetwork_models

"""
    Ce module sert à englober les modèles implémentés from scratch du Deep Learning. 
    Cette version 1.0  contient deux modèles: 
                <> PersonalizedLeNet5: CNN paramétrable en s'appuyant sur un autres module nommé 'cnn_layers_mini_batch'
                <> MultiClassNeuralNetwork: MLP paramétrable à l'étape d'instanciation.
"""

#importations
from tqdm import tqdm
import numpy as np 
import pandas as pd

#importations des modules pour leNet5
import importlib
import datapreprocessing
import optimisationmodels
import cnn_layers_mini_batch
import toolsmethods

# Rechargement les modules pour prendre en compte les modifications sans redémarrer le kernel
importlib.reload(datapreprocessing)
importlib.reload(optimisationmodels)
importlib.reload(cnn_layers_mini_batch)
importlib.reload(toolsmethods)

from datapreprocessing import ImagesDataPreprocessingForCNN
from optimisationmodels import AdamOptimizer , SGDOptimiser , MomentumOptimizer
from toolsmethods import ToolsMethods
from cnn_layers_mini_batch import ConvLayer, ActLayer, PoolLayer, Flatten, DenseLayer




######################################################## Classe  PersonalizedLeNet5  #########################################################################

class PersonalizedLeNet5:
    
    def __init__ (self , layers: list , optimizer: str= 'SGD' , learning_rate: float= 0.01 , _lambda: float= 0.0):
        """
        #Le constructeur d'Objet 'PersonalizedLeNet5' qui initialise le réseau de neurones CNN en définissant les ses différentes couches
         et le learning rate.
        """
        #validation des entrées
        objects = (ConvLayer, ActLayer, PoolLayer, Flatten, DenseLayer)

        print(f"Nombre de couches = {len(layers)}")
        for i, layer in enumerate(layers):
            print(f"[{i}] -> type: {type(layer)}, isinstance: {isinstance(layer, (ConvLayer, ActLayer, PoolLayer, Flatten, DenseLayer))}")



        
        assert isinstance(layers , list) and all(isinstance(layer , objects)  for layer in layers) and len(layers) >= 5 , (
            " layers doit être une liste d'au moins 5 objets de type [ConvLayer , ActLayer , PoolLayer , Flatten , DenseLayer] !"
        )
        assert isinstance (optimizer , str) and optimizer in ['SGD','Adam','Momentum'] , \
                                                       "L'optimiseur doit être une chaîne de caratères parmi ['SGD','Adam','Momentum'] !"
        assert isinstance (learning_rate , (int , float )) and learning_rate > 0, " Learning rate doit être un nombre positif!"

        #initialisation des attributs par les paramètres passés au constructeur et les autres par  des listes vides
        self.layers         = layers
        self.optimizer_type = optimizer
        self.optimizers     = []
        self.learning_rate  = learning_rate

        # Initialisation des optimiseurs pour les couches entraînables
        for layer in self.layers:
            if layer.trainable:
                if self.optimizer_type == "SGD":
                    self.optimizers.append(SGDOptimiser(learning_rate= learning_rate  , _lambda = _lambda))
                    
                elif self.optimizer_type == "Adam":
                    self.optimizers.append(AdamOptimizer(learning_rate= learning_rate))
                    
                else:
                    self.optimizers.append(MomentumOptimizer(learning_rate= learning_rate))
            else:
                 self.optimizers.append(None)


    def forward (self , X: np.ndarray) -> np.ndarray:
        """
        Le pipeline de forward: chaque couche reçoit la sortie de la couche d'avant et transmet sa sortie à la couche de devant pour 
        enfin avoir les prédictions (X -> Conv1 -> activation1 -> pool1 -> ... -> flatten -> dense1 -> ... -> sortie -> prédictions).
        """
        #vérification des entrées
        assert isinstance (X, np. ndarray ) and X.ndim == 4 , "L'argument 1 doit être de type 'ndarray' 4D du 'numpy' !"
        assert X.shape[3] == self.layers[0].numberOfChannels , \
                                         f"Le nombre de canaux ({X.shape[3]}) doit être égale à ({self.layers[0].numberOfChannels})!"
        
        #création et initialisation des sorties des couches
        self.output_layers = [X]

        out_put = X.copy()

        #propagation en avant:
        for layer in self.layers:
            out_put = layer.forward(out_put)
            #sauvegarde de la sortie des couches pour traçer les traîtements subis par l'image
            self.output_layers.append(out_put)
            
        #validation de la sortie finale
        assert isinstance(self.output_layers[-1] , np.ndarray) and self.output_layers[-1].ndim == 2 , \
                                                                               "La sortie de forward doit être de type 'ndarray' de 2D !"
        assert self.output_layers[-1].shape[0] == X.shape[0] , \
                                   f"Le nombre de prédictions ({self.output_layers[-1].shape[0]}) doit être égal au nombre d'images ({X.shape[0]})!"
        assert np.all(np.isfinite(self.output_layers[-1])) , "La sortie ne doit pas contenir des valeurs 'Nan' ou 'inf' !"

        return self.output_layers[-1]


    def compute_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
            Calcule la cross-entropy loss entre les vraies classes et les probabilités softmax.
            Compatible avec y en one-hot (2D) ou y comme vecteur (1D)
        """
        #validation des entrées
        assert isinstance(y, np.ndarray) and isinstance(y_pred, np.ndarray), "Entrées doivent être des numpy ndarray !"
        assert y_pred.shape[0] == y.shape[0], "Nombre d'éléments incohérent entre y_true et y_pred !"
    
        m = y_pred.shape[0]
        epsilon = 1e-15  # pour éviter log(0)

        # éviter les division par zéros
        y_pred = np.clip(y_pred , epsilon , 1.0 - epsilon) 
    
        if y.ndim == 1:
            # y est un vecteur dont les valeurs sont les numéros de classes
            log_probabilities = - np.log(y_pred[np.arange(m) , y])
        else:
            # y est un vecteur one-hot comme y_pred
            log_probabilities = - np.sum(y * np.log(y_pred) , axis= 1)
    
        loss = np.mean(log_probabilities)
        
        assert np.isfinite(loss), "Loss invalide (inf ou NaN)"
        
        return loss


    def compute_accuracy(self , y: np.ndarray , y_pred: np.ndarray) -> float:
        """
            calcule  la précision : proportion de prédictions correctes
        """
        #validation des entrées
        assert isinstance(y , np.ndarray) and isinstance(y_pred , np.ndarray), "Les entrées de 'compute_accuracy' doivent être des tableaux numpy!"
        assert y.shape[0] == y_pred.shape[0] , "Les deux arguments 'y' et 'y_pred' doivent avoir le même nombre de lignes !"
 
        #détermine la classe selon la probabilité maximum pour chaque ligne (la condition pour vérifier le codage one-Hot de y_pred)
        predictions = np.argmax(y_pred , axis=1) if y_pred.ndim > 1 else y_pred

        #transformer les valeurs vraies y en vecteur d'entiers s'il est sous forme de matrice
        trueLabels = np.argmax(y , axis=1) if y.ndim > 1 else y
    
        #calcul de la accuracy
        accuracy = np.mean(predictions == trueLabels)
        
        assert 0 <= accuracy <= 1 , "Accuracy doit être comprise entre 0 et 1 !"
        
        return accuracy

    def backward(self , y: np.ndarray , y_pred: np.ndarray):
        """
        Le pipeline de backward des couches ou chaque couche calcule le gradient de loss par rapport à ses poids et ses biais si elle en dispose
        à son entrée qu'elle transmet à la couche d'avant.
        la couche de sortie 'softmax' a besoin de y et y_pred pour calculer son gradient: y_pred - y
        """
        #validation des entrées
        assert isinstance(y , np.ndarray) , "L'argument 1 'y_true' doit être de type 'ndarray' de 'numpy'!"
        assert isinstance(y_pred , np.ndarray) , "L'argument 2 'y_pred' doit être de type 'ndarray' de 'numpy'!"
        
        assert y.shape == y_pred.shape, "y et y_pred doivent avoir les mêmes dimensions !"
        
        #self.output_layers = [X]

        input_gradient = y_pred - y  #gradient à passer à backward de la couche de sortie 'softmax'

        #rétropropagation de gradient:
        for layer in reversed(self.layers):
            input_gradient = layer.backward(input_gradient)


    def update(self):
        """
        Elle mise à jour les poids et biais des couches entraînables (convolution et dense) avec l'optimiseur (SGD , Adam , Momentum) choisi 
        pendant l'instanciation du CNN.
        """
        #la mise à jour des poids et biais
        for i , layer in enumerate(self.layers , 0):
            if layer.trainable:
                layer.weights , layer.biases = self.optimizers[i].updateForOneLayer(layer.weights , layer.biases , layer.d_Weights , layer.d_Biases)
        
    
            

    def train(self , X: np.ndarray , y: np.ndarray , X_val: np.ndarray , y_val: np.ndarray , epochs: int= 100 , batch_size: int= 32) \
    -> tuple[list , list , list , list]:
        """
            l'entraînement de réseau CNN  en utilisant un mini-batch SGD avec validation et un optimiseur (SGD , Adam , Momentum)
            et retourne des listes de loss et d'accuracy pour train et validation.
        """
        #validation des entrées
        assert isinstance(X , np.ndarray) and isinstance(y , np.ndarray), "Les arguments 1 et 2 doivent être de type 'ndarray'!"
        assert isinstance(X_val , np.ndarray) and isinstance(y_val , np.ndarray), "Les arguments 3 et 4 doivent être de type 'ndarray'!"

        assert isinstance(epochs , int) and epochs > 0, "Epochs doit être un entier positive!"
        assert isinstance(batch_size , int) and batch_size > 0, "Batch size doit être un entier positive!"
        
    
        train_losses = []         #traçage des pertes pour l'ensemble de train
        val_losses = []           #traçage des pertes pour les ensembles de validation
        train_accuracies = []     #traçage des précisions pour l'ensemble de train
        val_accuracies = []       #traçage des précisions pour les ensembles de validation

        for epoch in tqdm(range(epochs) , desc= "Entraînement du modèle:"):
            start_time = time.time()
            #mélanger les données avant leurs division en mini-batch
            indices = np.random.permutation(X.shape[0])    
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0  # le loss de chaque epoch
    
            for i in range(0 , X.shape[0] , batch_size):
                t1 = time.time()                 #time de début
                
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
    
                y_pred = self.forward(X_batch)
                
                t2 = time.time()                   #time de forward
                
                epoch_loss += self.compute_loss(y_batch , y_pred)
                self.backward(y_batch , y_pred)

                t3 = time.time()           #time de backward
                self.update()

                t4 = time.time()         #time de l'optimiseur

    
            # Calcul des pertes et précisions sur les ensembles de train et de validation
            train_pred = self.forward(X)
            val_pred   = self.forward(X_val)
    
            train_loss = self.compute_loss(y , train_pred)
            val_loss   = self.compute_loss(y_val , val_pred)
    
            train_accuracy = self.compute_accuracy(y , train_pred)
            val_accuracy   = self.compute_accuracy(y_val , val_pred)

            #ajout des  pertes et des précisions aux listes de traçages correspondantes
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append (val_accuracy)

            #Affichage d'un message tout les 10 épochs
            if epoch % 2 == 0:
                tqdm.write(
                    f" Epoch {epoch} ---> Train Loss: {train_loss :.4f} | Validation Loss : {val_loss :.4f} |" 
                    f"Train Accuracy: {train_accuracy:.4f} | Validation Accuracy: {val_accuracy:.4f}"
                )

            # print(f"Epoch {epoch+1} time: {time.time() - start_time:.2f}s")

        assert len(train_losses) > 0  and len(val_losses) > 0 and len(train_accuracies) > 0 and len(val_accuracies)  , (
                "Une au moins des listes de pertes et de précisions à retourner est vide!"
            )

        return train_losses , val_losses , train_accuracies , val_accuracies

            

    def predict(self , X: np.ndarray) -> np.ndarray:
        """
            prédit les étiquettes des données
        """
        #validation des entrées 
        assert isinstance(X , np.ndarray), "L'argument doit être de type 'ndarray' de 'numpy'!"
    
        y_pred = self.forward(X)  # propagation avant pour obtenir les probabilités
        predictions = np.argmax(y_pred , axis= 1)  
    
        assert predictions.shape == (X.shape[0] , ), "Les dimensions de predictions sont incorrecte!"
        
        return predictions


##################################################### Classe NeuralNetwork  ###############################################################################


class MultiClassNeuralNetwork:
    def __init__ (self , layer_sizes: list[int]= [8 , 16 , 8 , 1] , learning_rate: float= 0.01):
        #Initialise le réseau de neurones par les tailles des couches 'layer_sizes' [input_size , hidden1_size , ... , hiddenk_size , output_size]
        #et le learning rate.
        #vérification des entrées
        assert isinstance(layer_sizes , list) and all(isinstance(element , int) and (element > 0) for element in layer_sizes) and len( layer_sizes ) >= 2, (
            " layer_sizes doit être une liste de moins deux entiers strictement positifs!"
        )
        assert isinstance ( learning_rate , (int , float )) and learning_rate > 0, " Learning rate doit être un nombre positif!"

        #initialisation des attributs par les paramètres passés au constructeur et les autres par  des listes vides
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        # Initialisation des poids et biais
        np. random . seed (42)
        for i in range (len ( layer_sizes ) - 1):
            w = np.random.randn(layer_sizes[i] , layer_sizes[i+1]) * 0.01
            b = np.zeros((1 , layer_sizes[i+1]))
            #vérification de la compabilité des dimensions entre la couche actuelle est la couche d'avant
            assert w.shape == ( layer_sizes [i], layer_sizes[i +1]) , (
                f"La matrice des poids de la couche {i +1} a des dimensions incompatible avec la couche {i}!"
            )
            assert b.shape == (1 , layer_sizes[i +1]) , f"Le vecteur colonne des biais de la couche {i +1} a des dimmensions incorrêctes!"
            #ajout des poids et des biais aux listes qui leurs correspondent
            self . weights . append (w)
            self . biases . append (b)
        print(f"Vous venez de créer un RN de {len(self.layer_sizes)} couches dont {len(self.layer_sizes)-2} couches cachées")
        print(f"les tailles des couches de votre réseau: {self.layer_sizes}")

    def forward (self , X: np.ndarray) -> np.ndarray:
        #calcule des combinaisons Z et leurs activations des différentes couches de réseau
        #vérification des entrées
        assert isinstance (X, np. ndarray ), "L'argument 1 doit être de type 'ndarray' du 'numpy'!"
        assert X. shape[1] == self.layer_sizes [0] , f" La dimension d'entrée ({X.shape[1]}) doit être égale à ({self. layer_sizes [0]})!"
        
        #création et initialisation de deux attributs de  l'instance: listes des combinaisons (Z_list) et listes des activations (A_list)
        self.Z_list = []
        self.A_list = [X]
        for i in range (len(self.weights) - 1):
            #calcul de les combinaisons z pour chaque couche à l'exception de la couche d'entrée
            z = np.dot(self.A_list[i] , self.weights[i]) + self.biases[i]
            #vérification des dimensions de z avant de l'ajouter à la liste Z_list 
            assert z.shape == (X. shape [0] , self.layer_sizes[i+1]) , f" Les dimensions de Z^{[i +1]} sont incorrêtes!"
            self.Z_list.append(z)

            #les même étapes pour les activations de chaque couche à l'exception de la couche d'entrée
            # activation par ReLu pour les couches cachées
            a = self.relu(self.Z_list[i] , i) # activation par ReLu pour les couches cachées
            #vérification des dimensions de a avant de l'ajouter à la liste A_list 
            assert a.shape == (X. shape [0] , self.layer_sizes[i+1]) , f" Les dimensions de A^{[i +1]} sont incorrêtes!"
            self.A_list.append(a)

        #les combinaisons et les activations de la couche de sortie 
        z = np.dot(self.A_list[-1] , self.weights[-1]) + self.biases[-1]
        #vérification des dimension de output z
        assert z.shape == (X. shape[0] , self.layer_sizes [-1]) , "Z de la couche de sortie n'a pas de bonnes dimensions!"
        #ajout de z à la liste 'A_list'
        self.Z_list.append(z)
        #calcule de activations (prédictions) de la couche de sortie
        A_output = self.softmax(z) # activation par softmax pour la couche de sortie
        #vérification des dimensions de A de sortie
        assert A_output.shape == (X.shape[0] , self.layer_sizes[-1]) , "La couche de sortie n'a des dimensions incorrêtes!"
        #ajouter l'activation 'A_output' à la la liste 'A_list' 
        self.A_list.append(A_output)    
    
        return self.A_list[-1]


    def compute_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        # Calcule la cross-entropy loss entre les vraies classes et les probabilités softmax.
        # Compatible avec y en one-hot ou vecteur d'entiers.
        #validation des entrées
        assert isinstance(y, np.ndarray) and isinstance(y_pred, np.ndarray), "Entrées doivent être des np.ndarray"
        assert y_pred.shape[0] == y.shape[0], "Nombre d'échantillons incohérent"
    
        m = y.shape[0]
        epsilon = 1e-15  # pour éviter log(0)
    
        # Transformer y en indices si c'est du one-hot
        if y.ndim > 1 and y.shape[1] > 1:
            y = np.argmax(y, axis=1)
    
        # Extraire la probabilité de la bonne classe pour chaque exemple
        probas = y_pred[np.arange(m), y]
        log_probas = np.log(probas + epsilon)
    
        loss = -np.mean(log_probas)
        assert np.isfinite(loss), "Loss invalide (inf ou NaN)"
        
        return loss


    def compute_accuracy(self , y: np.ndarray , y_pred: np.ndarray) -> float:
        #calcule  la précision : proportion de prédictions correctes
        #vérification des entrées
        assert isinstance(y , np.ndarray) and isinstance(y_pred , np.ndarray), "Les entrées de 'compute_accuracy' doivent être des tableaux numpy!"
        assert y.shape[0] == y_pred.shape[0] , "Les deux arguments 'y' et 'y_pred' doivent avoir le même nombre de lignes !"

        #y_pred est une matrice de probabilités (softmax), on prend la classe prédite c'est à dire la classe avec 
        #détermine la classe selon la probabilité maximum pour chaque ligne
        predictedClass = np.argmax(y_pred , axis=1) if y_pred.ndim > 1 else y_pred

        # transformer les valeurs vraies y en vecteur d'entiers s'il est sous forme de matrice
        trueLabels = np.argmax(y , axis=1) if y.ndim > 1 else y
    
        # Calcul de la accuracy
        accuracy = np.mean(predictedClass == trueLabels)
        
        assert 0 <= accuracy <= 1, "Accuracy doit être comprise entre 0 et 1 !"
        
        return accuracy

    def backward(self, X: np.ndarray , y: np.ndarray , y_pred: np.ndarray , _lambda: float):
        # calcule les gradients dW et db pour chaque couche excepte la couche d'entrée
        #vérification des entrées
        assert isinstance(X , np.ndarray) and isinstance(y , np.ndarray) and isinstance(y_pred , np.ndarray) , (
            "Les trois arguments doivent être de type 'ndarray' de 'numpy'!"
        )
        assert X.shape[1] == self.layer_sizes[0] , f"La dimension de l'entrée du NN({X.shape[1]}) doit être égale à ({self.layer_sizes[0]})!"
        assert y.shape == y_pred.shape, "y et y_pred doivent avoir les mêmes dimensions"
        assert isinstance(_lambda , (float , np.integer , int)) and _lambda >= 0 , "Le terme de régularisation 'lambda'doit être un nombre positif!"
    
        m = X.shape[0]
        #création et initialisation par des zéros de deux attributs (listes des matrices des poids et biais) de l'instance 
        self.d_weights = [np.zeros_like(w) for w in self.weights]
        self.d_biases = [np.zeros_like(b) for b in self.biases]
    
        # les gradients de la couche de sortie
        dZ = y_pred - y 
        assert dZ.shape == y_pred.shape , "les gradients 'dZ' de la couche de sortie a des dimensions incorrêctes!"
        self.d_weights[-1] = (self.A_list[-2].T @ dZ) / m
        self.d_weights[-1] += _lambda * self.weights[-1] / m         #ajout de régularisation aux poids
        self.d_biases[-1] = np.sum(dZ, axis=0, keepdims=True) / m
    
        # Les gradients des couches cachées
        for i in range(len(self.weights) - 2 , -1 , -1):
            dZ = np.dot(dZ , self.weights[i+1].T) * self.relu_derivative(self.Z_list[i] , i)
            self.d_weights[i] = np.dot(self.A_list[i].T , dZ) / m 
            self.d_weights[i] += _lambda * self.weights[i] / m         #ajout de régularisation aux poids
            self.d_biases[i] = np.sum(dZ, axis=0, keepdims=True) / m

        #la mise à jour des poids et biais
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate*self.d_weights[i]
            self.biases[i]  -= self.learning_rate*self.d_biases[i] 

    def train(self , X: np.ndarray , y: np.ndarray , X_val: np.ndarray , y_val: np.ndarray , epochs: int= 100 , batch_size: int= 32 , _lambda: float= 0.01):
        #l'entraînement de réseau se fait en utilisant un mini-batch SGD avec validation
        #vérification des entrées
        assert isinstance(X , np.ndarray) and isinstance(y , np.ndarray), "Les arguments 1 et 2 doivent être de type 'ndarray'!"
        assert isinstance(X_val , np.ndarray) and isinstance(y_val , np.ndarray), "Les arguments 3 et 4 doivent être de type 'ndarray'!"
        assert X.shape[1] == self.layer_sizes[0], (
            f"La dimension d'entrée ({X.shape[1]}) doit correspondre à la taille de la couche d'entrée ({self.layer_sizes[0]})!"
        )
        assert y.shape[1] == self.layer_sizes[-1], (
            f"La dimension de sortie ({y.shape[1]}) doit correspondre à la taille de la couche de sortie ({self.layer_sizes[-1]})!"
        )
        assert X_val.shape[1] == self.layer_sizes[0], (
            f"La dimension d'entrée de la validation ({X_val.shape[1]}) doit correspondre à la taille de la couche d'entrée ({self.layer_sizes[0]})!"
        )
        assert y_val.shape[1] == self.layer_sizes[-1], (
        f"La dimension de sortie de la validation ({y_val.shape[1]}) doit correspondre à la taille de la couche de sortie ({self.layer_sizes[-1]})!"
        )
        assert isinstance(epochs , int) and epochs > 0, "Epochs doit être un entier positive!"
        assert isinstance(batch_size , int) and batch_size > 0, "Batch size doit être un entier positive!"
        assert isinstance(_lambda , (float , np.integer , int)) and _lambda >= 0 , "Le terme de régularisation 'lambda'doit être un nombre positif!"
    
        train_losses = []         #traçage des pertes pour l'ensemble de train
        val_losses = []           #traçage des pertes pour les ensembles de validation
        train_accuracies = []     #traçage des précisions pour l'ensemble de train
        val_accuracies = []       #traçage des précisions pour les ensembles de validation

        for epoch in tqdm(range(epochs) , desc= "Entraînement du modèle:"):
            #mélanger les données avant leurs division en mini-batch
            indices = np.random.permutation(X.shape[0])    
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0  # le loss de chaque epoch
    
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
    
                y_pred = self.forward(X_batch)
                epoch_loss += self.compute_loss(y_batch , y_pred)
                self.backward(X_batch , y_batch , y_pred , _lambda)

    
            # Calcul des pertes et précisions sur les ensembles de train et de validation
            train_pred = self.forward(X)
            val_pred = self.forward(X_val)
    
            train_loss = self.compute_loss(y , train_pred)
            val_loss   = self.compute_loss(y_val , val_pred)
    
            train_accuracy = self.compute_accuracy(y , train_pred)
            val_accuracy   = self.compute_accuracy(y_val , val_pred)

            #ajout des  pertes et des précisions aux listes de traçages correspondantes
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append (val_accuracy)

            #Affichage d'un message tout les 10 épochs
            if epoch % 10 == 0:
                tqdm.write(
                    f" Epoch {epoch} ---> Train Loss: {train_loss :.4f} | Validation Loss : {val_loss :.4f} |" 
                    f"Train Accuracy: {train_accuracy:.4f} | Validation Accuracy: {val_accuracy:.4f}"
                )

            assert len(train_losses) > 0  and len(val_losses) > 0 and len(train_accuracies) > 0 and len(val_accuracies)  , (
                "Une au moins des listes de pertes et de précisions à retourner est vide!"
            )

        return train_losses , val_losses , train_accuracies , val_accuracies

            

    def predict(self , X: np.ndarray) -> np.ndarray:
        #prédit les étiquettes des données
        #vérification des entrées 
        assert isinstance(X , np.ndarray), "L'argument doit être de type 'ndarray' de 'numpy'!"
        assert X.shape[1] == self.layer_sizes[0], f"la dimension deux de l'entrée ({X.shape[1]}) doit être égale à ({self.layer_sizes[0]})!"
    
        y_pred = self.forward(X)  # propagation avant pour obtenir les probabilités
        predictions = np.argmax(y_pred , axis= 1)  
    
        assert predictions.shape == (X.shape[0] , ), "Les dimensions de predictions sont incorrecte!"
        
        return predictions


              #-------------- Les fonctions d'activation et leurs dérivées sont ajoutées ici comme méthodes de la classe NeuralNetwork ------------

    def relu(self , X: np.ndarray , i: int) -> np.ndarray:
        #traite les valeurs de la matrice d'entrée et modifie les valeurs de la matrice d'activation A qui est une attribut de l'objet Layer
        #Validation des arguments
        assert isinstance(i , int) , "Le compteur i doit être un entier!"
        assert isinstance(X , np.ndarray) , "L'argument doit être de type ndarray de numpy!"
        assert X.shape[1] == self.layer_sizes[i+1] , "la deuxième dimension de Z est incorrecte!"

        #calcule des activations
        resultat = np.maximum( 0 , X )
        #validation de résultat
        assert np.all( resultat >= 0 ) , "l'activation selon ReLU ne doit pas contenir des valeurs strictement négatives!"

        return resultat

    def relu_derivative(self , X: np.ndarray , i: int) -> np.ndarray:
        # calcule les dérivées de la fonction Relu et renvoie un tableau contenant des valeurs 0 et/ou 1
        #Validation des arguments
        assert isinstance(i , int) , "Le compteur i doit être un entier!"
        assert isinstance(X , np.ndarray) , "L'argument n'est pas de type ndarray de numpy"
        assert X.shape[1] == self.layer_sizes[i+1] , "la deuxième dimension de Z est incorrecte"
        
        #calcule des activations
        resultat = (X > 0).astype(float)
        #validation de résultat
        assert np.all((resultat == 0) | (resultat == 1)) , "la dérivée de ReLU égale à 0 ou à 1!"

        return resultat

    def softmax(self , X: np.ndarray) -> np.ndarray :
        #calcule les probabilités d'appartenir à une classe (dans notre cas 33 classe chacune représente un alphabet de Tifinagh)
        #validation de l'entrée
        assert isinstance(X , np.ndarray), "L'argument doit être de type 'ndarray' de 'numpy' !"
        
        # pour éviter les valeurs 'inf' <==> exp de grandes valeurs tends vers l'infini
        maxLine = np.max(X , axis=1 , keepdims=True)
        expOfX  = np.exp(X - maxLine)

        #calcule de la sortie en utilisant la formule de softmax (les rapport exp(xi)/ sum(xj) , j= les valeurs de chaque ligne)
        resultat = expOfX / np.sum(expOfX, axis=1, keepdims=True)

        #validation de la sortie
        assert isinstance(resultat , np.ndarray) and resultat.shape == X.shape , "La sortie doit être de même type et a la même shape que l'entrée !"
        assert np.all((resultat >= 0) & (resultat <= 1)), "Les valeurs de la sortie doivent appartenir à [0, 1] !"
        assert np.allclose(np.sum(resultat, axis=1), 1), "Softmax doit retourner des valeurs dont la somme vaut 1 pour chaque enregistrement !"
        
        return resultat


