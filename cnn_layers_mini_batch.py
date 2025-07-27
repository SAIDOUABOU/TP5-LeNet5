#module cnn_layers_mini_batch

"""
    Ce module contient des classe servant à instancier différentes couches constituant un CNN:
        - Classe 'ConvLayer' : pour instancier des couches de convolution.
        - Classe 'ActLayer'  : pour instancier des couches d'activation qui suivent souvent des convolutions.
        - Classe 'PoolLayer' : pour instancier des couches de pooling.
        - Classe 'DenseLayer': pour construire des couches Fully connected et les couches de sortie.
"""

#importations
import pandas as pd
import numpy as np
from scipy.signal import correlate2d
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.model_selection import train_test_split
import os
import cv2
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import convolve2d, correlate2d



##############################################Classe Convolution Layer  ###################################################################################
                                             # (teste de la classe a été faite avec succes )
class ConvLayer:
    """
        Cete classe sert à intancier des couches de convolution parametrables par des arguments qu'y sont passées par le constructeur:
         - numberOfChannels: le nombre de canaux, un pour des images en gris et 3 pour les images RGB
         - numberOfFilters: le nombre de filtres à appliquer.
         - sizeOfFilters: la tailles des filtres à appliquer.
         - stride: le pas de déplacement du filtre sur l'image (Par défaut égale à 1).
         - padding: les marges à ajouter à l'image (Par défaut n'y apas d'ajout).
    """
    def __init__(self , numberOfChannels: int= 1 , numberOfFilters: int= 6 , sizeOfFilters: int= 5 , stride: int= 1 , padding: int= 0):
        """
            Inctancier l'objet classe de convolution avec les parametres spécifiés
        """
        #validation des entrées
        assert (
            isinstance(numberOfChannels , (int , np.integer)) and  numberOfChannels > 0 
        ) , "L'argument 1 'numberOfChannels'doit être un entier strictement positif (1 pour les images en gris et 3 pour les images en RGB pour la  première couche de conv.) !"
        
        assert isinstance(numberOfFilters , (int , np.integer)) and numberOfFilters > 0 , \
                                                          "L'argument 2 'numberOfFilters' doit être un entier strictement positif !"
        assert isinstance(sizeOfFilters , (int , np.integer)) and sizeOfFilters > 0 , "L'argument 3 'sizeOfFilter' doit être un entier strictement positif !"
        assert isinstance(stride , (int , np.integer)) and stride > 0 , "L'argument 4 'stride' doit être un entier strictement positif !"
        assert isinstance(padding , (int , np.integer)) and padding >= 0 , "L'argument 5 'padding' doit être un entier positif !"

        #initialisation des attributs de l'objet 'couche de convolution'
        self.numberOfChannels = numberOfChannels
        self.numberOfFilters = numberOfFilters
        self.sizeOfFilters = sizeOfFilters
        self.stride = stride
        self.padding = padding

        self.trainable = True

        # Initialisation  des filtres par des petites valeurs et les biais des filtres par des zéros
        stddev = np.sqrt(2 / (self.numberOfChannels * self.sizeOfFilters**2))
        self.weights = np.random.randn(self.numberOfFilters, self.numberOfChannels, self.sizeOfFilters, self.sizeOfFilters) * stddev
        self.biases = np.zeros(numberOfFilters)  


    def imageToColumn(self , X: np.ndarray , KH: int , KW: int , stride: int= 1) -> np.ndarray:
        """
        Transforme les cartes (images) en matrice 2D qui'elle retourne
        
        """
        #validation des entrées
        assert isinstance(X , np.ndarray) and X.ndim == 4 , "L'arg1 doit être de type ndarray 4D !"
        assert isinstance(KH , (int , np.integer))  and KH > 0 , "L'arg2 'height of filters' doit être un entier > 0 !"
        assert isinstance(KW , (int , np.integer))  and KW > 0 , "L'arg3 'width of filters' doit être un entier > 0 !"
        assert isinstance(stride , (int , np.integer))  and stride > 0 , "L'arg3 'stride of filters' doit être un entier > 0 !"
    
        #initialisation des des dimensions de la sortie
        n_cards , height , width , channels = X.shape
        out_height = (height - KH) // stride + 1
        out_width  = (width  - KW) // stride + 1
        
        ##glissage des filtres avec stride sur les cartes (axes height et width)
        matrixOfCards = sliding_window_view(X , window_shape= (KH , KW) , axis= (1, 2)) 
        matrixOfCards = matrixOfCards[:, ::stride, ::stride, :, :, :] 
        
        #transformation des données en tableau 2D
        matrixOfCards = matrixOfCards.reshape(n_cards*out_height*out_width , KH*KW*channels)
        
    
        #validation de la sortie
        required_shape = (n_cards*out_height*out_width , KH*KW*channels)
        assert matrixOfCards.shape == required_shape , f"Le shape de data vectorisé est {matrixOfCards.shape} alors que l'attendu est {required_shape} !"
        assert np.all(np.isfinite(matrixOfCards)) , "Le data vectorisé contient des valeurs 'Nan' ou 'inf' !"
        
        return matrixOfCards 

    def columnToImage(self , X_column: np.ndarray , output_shape: tuple , KH: int , KW: int , stride: int) -> np.ndarray:
        """
        Elle reconvertit les blocs vectorisés (cols) en images 4D  et retourne un tableau des images 4D (N , H , W , C)
    
        """
        #validation des entrées
        assert isinstance(X_column , np.ndarray) and X_column.ndim == 3 , "L'arg1 doit être un tableau 'ndarray' 3D !"
        assert (
            isinstance(output_shape , tuple) and all(isinstance(element , (int , np.integer)) for element in output_shape) and 
            len(output_shape) == 4 
        ) , "L'arg2 doit être un tuple de 4 entiers !"
        assert isinstance(KH , (int , np.integer))  and KH > 0 , "L'arg3 'width of filters' doit être un entier > 0 !"
        assert isinstance(KW , (int , np.integer))  and KW > 0 , "L'arg4 'width of filters' doit être un entier > 0 !"
        assert isinstance(stride , (int , np.integer))  and stride > 0 , "L'arg5 'stride of filters' doit être un entier > 0 !"
    
        #initialisation des variables locales
        n_cards , height , width , channels = output_shape
        # height_column = X_column.shape[1]                          #(H_out*W_out)
        
        #calcul de height_out et width_out
        height_out = (height - KH) // stride + 1
        width_out  = (width  - KW) // stride + 1
    
        #initialisation de la sortie et du masque de superposition
        images  = np.zeros((n_cards , height , width , channels) , dtype= X_column.dtype)   
        count    = np.zeros_like(images)
    
        #remplissage de la sortie
        column_indix = 0
        
        for i in range(0 , height - KH + 1 , stride):
            for j in range(0 , width - KW + 1 , stride):
                patch = X_column[: , column_indix , :].reshape(n_cards , KH , KW , channels)
                images[: , i:i+KH , j:j+KW , :] += patch
                count[: , i:i+KH , j:j+KW , :]  += 1
                column_indix += 1
    
        #éviter la surcompensation due aux recouvrements
        images /= np.maximum(count , 1)
    
        #validation de la sortie
        assert images.shape == output_shape , f"Le shape de la sortie {images.shape} ne correspond au shape attendu {output_shape} !"
        assert np.all(np.isfinite(images)) , "La sortie contient des valeurs 'Nan' ou 'inf' !"
        
        return images



    def convolution2d(self , X: np.ndarray , filters: np.ndarray , biases: np.ndarray = None , stride: int= 1) -> np.ndarray :
        """
        Applique une convolution vectorisée sur des entrées X de forme (n_images , height , width , channels) avec des filtres 
        de forme (n_filters , KH , KW , channels) et retourne : (n_images , out_h , out_w , n_filters)
        
        """
        #validation des entrées
        assert isinstance(X , np.ndarray) and X.ndim == 4 , "L'arg1 'Data d'entrée' doit être de type ndarray 4D !"
        assert isinstance(filters , np.ndarray) and filters.ndim == 4 , "L'arg2 'filters' doit être de type ndarray 4D !"
        assert (biases is None) or (isinstance(biases, np.ndarray) and biases.shape == (self.numberOfFilters ,)) , \
                                                         "L'arg3 'biases' doit être un ndarray de shape (n_filters,) ou None"
        assert isinstance(stride , (int , np.integer))  and stride > 0 , "L'arg4 'stride of filters' doit être un entier > 0 !"
        
        #initialisation des variables locales
        n_cards , height , width , channels = X.shape
        n_filters , Channels_fliters ,  KH , KW  = filters.shape
        
        #Vectorisation des entrées (images ou cartes résultantes des filtres précédents)
        X_vectorized = self.imageToColumn(X , KH , KW , stride)
    
        #Vectorisation des filtres  (n_filters , KH*KW*Channels_fliters)
        filters_vectorized = filters.reshape(n_filters , -1) 

        #Application de convolution par le produit matriciel (n_cards , out_height*out_width , n_filters)
        output = X_vectorized @ filters_vectorized.T              
    
        #Ajout des biais des filtres
        if biases is not None:
            #Ajuster le shape des biais pour faire de broadcast
            output += biases  
    
        #Redimensionnement sous forme de (n_cards , out_height , out_width , n_filters)
        out_height  = (height - KH) // stride + 1
        out_width   = (width   - KW) // stride + 1
        
        #vérification des dimension de sortie
        assert out_height > 0 and out_width > 0, (
            f"Taille de sortie invalide : (height={height}, width={width}), "f"filtre={self.sizeOfFilters}, stride={self.stride}, padding={self.padding} "
            f"→ (out_height={out_height}, out_width={out_width})"
        )
        
        output      = output.reshape(n_cards , out_height , out_width , n_filters)
    
        #validation de la sortie
        required_shape = (n_cards , out_height , out_width , n_filters)
        assert output.shape == required_shape , f"Le shape de data vectorisé est {output.shape} alors que l'attendu est {required_shape} !"
        assert np.all(np.isfinite(output)) , "Le résultat de convolution 'output' contient des valeurs 'Nan' ou 'inf' !"
    
        return output



    def forward(self , images: np.ndarray) -> np.ndarray:
        """
            calucl de la convolution sur l'objet 'image' passé comme argument sous forme de ndarray
        """
        #validation des entrées
        assert isinstance(images , np.ndarray) and images.ndim == 4 , "L'argument 1 'image' doit être de type 'ndarray' de dimension 4 !"

        #ajout de padding sur les images 
        self.input_X_NoPadded = images
        
        if self.padding > 0:
            # padding (constantes zéros) sur hauteur et largeur des images (dim1 et dim2)
            padded_images = np.pad(images , ((0 , 0) , (self.padding , self.padding), (self.padding , self.padding) , (0 , 0)) ,
                                  mode='constant' , constant_values= 0) 
        else:
            padded_images = images
            
        #sauvegarde de l'entrée pour en servir lors de backward
        self.input_X = padded_images 

        #calcul de la convolution
        output = self.convolution2d(self.input_X , self.weights , self.biases , self.stride)

        #validation de la sortie
        assert np.all(np.isfinite(output)) , "la sortie contient des valeurs 'Nan' ou 'inf' !"                           

        return output
        

    def backward(self , inputGradients: np.ndarray) -> np.ndarray:
        """
            Elle recoit le gradient passé par la couche qui devant 'Activation' calcule le gradient de la fonction de coût par rapport aux poids et
            biais des filtres et aussi le gradient par rapport à l'entrée de la couche qu'il passe à la couche précédente.
        """
        #validation de l'entrée
        assert isinstance(inputGradients , np.ndarray) and inputGradients.ndim == 4 , "L'entrée doit être un tableau de type 'ndarray' 4D !"
    
        #initialisations des variables locales
        N , H_in , W_in , F = inputGradients.shape
        F , C , KH , KW     = self.weights.shape
        stride  = self.stride
        padding = self.padding
    
        #vectorisation de gradient de sortie 'inputGradients' sous forme 3D (N , H_in*W_in , F)
        inputGradients_reshaped = inputGradients.reshape(N , -1 , F)  
    
        #vectorisation de l'entrée de la couche pendant le forward  (N, H_out*W_out, KH*KW*C)
        input_X_vectorized = self.imageToColumn(self.input_X , KH , KW , stride) 


        #redimensionnement en 2D : (N * H_out * W_out, F)  et (N * H_out * W_out, KH*KW*C)
        inputGradients_flat = inputGradients_reshaped.reshape(-1 , F)
        input_X_flat        = input_X_vectorized.reshape(-1 , KH * KW * C)
    
        #calcul de gradients des poids (dL/dW) 
        dW = inputGradients_flat.T @ input_X_flat   # (F, KH*KW*C)
        output_dW = dW.reshape(F, C , KH , KW)
    
        #calcul de gradients des poids (dL/db) qui est sous forme (F,)
        output_db = np.sum(inputGradients , axis=(0, 1, 2)) 
    
        #calcul du gradient par rapport à l'entrée avec padding (∂L/∂X_padded) sous forme (N , H_in*W_in , KH*KW*C)
        W_flat = self.weights.reshape(F , -1)  
        output_dX_column_padded = inputGradients_reshaped @ W_flat  
    
        #redimensionnement du gradient de sortie paddée en forme (N , height_pad , width_pad , C)
        height_pad = self.input_X.shape[1]
        width_pad  = self.input_X.shape[2]
        output_dX_padded = self.columnToImage(output_dX_column_padded , (N , height_pad , width_pad , C) , KH , KW , stride)
    
        #enlevement de padding s'il est été appliqué pendant le forward
        if padding > 0:
            output_dX = output_dX_padded[: , padding:-padding , padding:-padding , :]
        else:
            output_dX = output_dX_padded
    
        
        #validation des résultats
        assert output_dW.shape == self.weights.shape , \
                             f"le shape des gradients des poids est {output_dW.shape} alors que le shape attendu est {self.weights.shape} !"
        assert output_db.shape == self.biases.shape , \
                              f"le shape des gradients des biais est {output_db.shape} alors que le shape attendu est {self.biases.shape} !"
        assert output_dX.shape == self.input_X_NoPadded.shape , \
                     f"le shape des gradients d'entrée est {output_dX.shape} alors que le shape attendu est {self.input_X_NoPadded.shape} !"

        assert np.all(np.isfinite(output_dW)) , "Les gradients des poids ne doivent pas contenir des valeurs indéfinies (Nan , inf) !"
        assert np.all(np.isfinite(output_db)) , "Les gradients des biais ne doivent pas contenir des valeurs indéfinies (Nan , inf) !"
        assert np.all(np.isfinite(output_dX)) , "Les gradients de l'entrée ne doivent pas contenir des valeurs indéfinies (Nan , inf) !"
        
        #sauvegarde des gradients dans des attributs pour la mise à jour des poids 
        self.d_Weights = output_dW
        self.d_Biases  = output_db
    
        return output_dX
        
################################################## Classe ActivationLayer #############################################################################
                                            #(   la classe a été testée avec succes  )
class ActLayer:
    """
    Elle applique l'activation souhaitée (passée par le constructeur de la classe sous forme d'une chaîne parmi ['relu' , 'tanh']
    sur la sortie de la couche de convolution qui la précède et pour servire la couche de Pooling qui la suive.
    """
    def __init__(self , activationFunction: str= 'relu'):
        """
            Le constructeur de la couche d'activation avec la fonction passée par argument 
        """
        #validation de l'entrée
        assert isinstance(activationFunction , str) and activationFunction in ['relu' , 'tanh'] , \
                                              "La fonction d'activation doit être une parmi ['relu' , 'tanh'] !"

        #création et initialisation de l'attribut activationFonction
        self.activationFunction = activationFunction

        self.trainable = False
        

    def forward(self , inputMaps: np.ndarray) -> np.ndarray:
        """
            Apllique la fonction d'activation sur l'ensemble de cartes reçues de la couche de convolution et retourne le même nombre de cartes 
            activées.
        """
        #validation de l'entrée
        assert isinstance(inputMaps , np.ndarray) , "L'entrée doit être de type 'ndarray' de 'numpy' !"

        #sauvegarde de l'entrée pour en servir lors de la backward
        self.input_Z = inputMaps

        #activation des maps avec l'activation choisie
        if self.activationFunction == 'relu':
            outputMaps = self.relu(inputMaps)
                
        else:
            outputMaps = self.tanh(inputMaps)

        #validation de la sortie
        assert isinstance(outputMaps , np.ndarray) , "La sortie doit être de type 'ndarray' de 'numpy' !"
        assert inputMaps.shape == outputMaps.shape , \
                                    f"La sortie a comme shape {outputMaps.shape} ce qui est incoérent avec le shape de l'entrée {inputMaps.shape} !"
        
        if self.activationFunction == 'tanh':
            assert np.all((outputMaps >= -1) & (outputMaps <= 1)), \
                                               "Les valeurs des cartes activées par 'tanh' doivent être compris entre -1 et 1 !"
        else:
            assert np.all(outputMaps >= 0) , \
                                               "Les valeurs des cartes activées par 'relu' doivent être supérieures ou égales à 0 !"

        return outputMaps


    def backward(self , inputGradients: np.ndarray) -> np.ndarray:
        """
            Elle calcule le gradient de la fonction de coût par rapport aux cartes d'activations et le propage vers la couche de convolution
            qui est juste derrière la couche d'activation.
        """
        #validation de l'entrée
        assert isinstance(inputGradients , np.ndarray) , "L'entrée doit être de type 'ndarray' de 'numpy' !"

        #calcul des gradients
        if self.activationFunction == 'relu':
            output_dZ = inputGradients * self.relu_derivative(self.input_Z)  
        else:
            output_dZ = inputGradients * self.tanh_derivative(self.input_Z)

        #validation de la sortie
        assert isinstance(output_dZ , np.ndarray) , "La sortie doit être de type 'ndarray' de 'numpy' !"
        assert output_dZ.shape == self.input_Z.shape , \
                    f"La sortie a comme shape {output_dZ.shape} ce qui est incohérent avec le shape de l'entrée de la couche {self.input_Z.shape} !"

        return output_dZ
        

    def relu(self , X: np.ndarray) -> np.ndarray:
        """
          traite les valeurs des cartes d'entrée et les modifie selon la logique : max(0 , x) 
        """
        #Validation des arguments
        assert isinstance(X , np.ndarray) , "L'argument doit être de type 'ndarray' de numpy!"

        #calcule des activations
        result = np.maximum( 0 , X )
        #validation de résultat
        assert np.all( result >= 0 ) , "l'activation selon ReLU ne doit pas contenir des valeurs strictement négatives!"

        return result

    def tanh(self , X: np.ndarray) -> np.ndarray:
        """
          traite les valeurs des cartes d'entrée et les modifie selon comme suit: np.tanh(x) 
        """
        #Validation des arguments
        assert isinstance(X , np.ndarray) , "L'argument doit être de type 'ndarray' de numpy!"

        #calcule des activations
        result = np.tanh(X)
        
        #validation de résultat
        assert np.all((result >= -1) & (result <= 1)) , "l'activation selon 'tanh' ne doit pas contenir des valeurs < -1 ou > 1 !"

        return result

    def relu_derivative(self , X: np.ndarray) -> np.ndarray:
        """
          Calcule les valeurs de la dérivée de ReLU sur les éléments de la matrice d'entrée: Relu'(x) = 1 si x>0 , 0 sinon
        """
        #Validation de l'entrée
        assert isinstance(X , np.ndarray) , "L'argument doit être de type 'ndarray' de numpy!"

        #calcule des activations
        result = (X > 0).astype(float)
        
        #validation de résultat
        assert np.all((result == 0) | (result == 1)) , \
                      "La dérivée de l'activation selon ReLU ne doit pas retouner un résultat contenant des valeurs < 0 ou > 1 !"

        return result

    def tanh_derivative(self , X: np.ndarray) -> np.ndarray:
        """
          Calcule les valeurs de la dérivée de Tanh sur les éléments de la matrice d'entrée : tanh'(x) = 1 - tanh²(x)
        """
        #Validation de l'entrée
        assert isinstance(X , np.ndarray) , "L'argument doit être de type 'ndarray' de numpy!"

        #calcule des activations
        result = 1 - np.square(self.tanh(X))
        
        #validation de résultat
        assert np.all((result >= 0) & (result <= 1)) , \
                      "La dérivée de l'activation selon Tanh ne doit pas retouner un résultat contenant des valeurs < 0 ou > 1 !"

        return result
    

##################################################  Classe Pooling Layer  #############################################################################
                                          #la classe a été bien testée avec succes
class PoolLayer:
    """
        Classe de Pooling servant à créer des couches de pooling dans un réseau CNN tout en spécifiant les paramètres suivant:
        - pooling_type: la stratégie adopté pour faire de pooling ['avg' , 'max'] (par défaut: 'avg').
        - window_size: la taille de la fenêtre de pooling (par défaut: (2 , 2)).
        - stride: le pas de déplacement de la fenêtre de pooling (par défaut: 2)
    """
    def __init__(self , pooling_type: str= 'avg' , window_size: tuple[int , int]= (2 , 2) , stride: int= 2):
        """
            Le constructeur de la classe qui sert à instancier des couches de pooling pour un CNN.
        """
        #validation des entrées
        assert isinstance(pooling_type , str) and pooling_type in ['avg' , 'max'] , \
                        "L'argument 1 'pooling_type' doit être une string parmi ces deux 'avg' ou 'max' !"
        assert isinstance(window_size , tuple) and np.all(isinstance(number , (int , np.integer)) for number in window_size) , \
                                            "L'arguement 2 'window_size' doit être un tuple de deux entiers >= 2 !"
        assert isinstance(stride , (int , np.integer)) , "L'arguement 3 'stride' doit être un entier > 0 !"

        #création et initialisation des attributs de l'objet 'PoolLayer'
        self.pooling_type = pooling_type
        self.window_size  = window_size
        self.stride       = stride

        self.trainable = False


    def forward(self, imageMaps: np.ndarray) -> np.ndarray:
        """
        Applique un pooling sur plusieurs cartes d'activation 4D en utilisant une approche vectorisée et retourne un tableau 4D des cartes 
        réduites (N, H_out, W_out, C)
        """
        # validation de l'entrée
        assert isinstance(imageMaps, np.ndarray) and imageMaps.ndim == 4, "L'argument 'imageMaps' doit être un tableau ndarray à 4 dimensions !"

        #sauvegarde de l'entrée pour le backward
        self.inputMaps = imageMaps

        #initialisation des variable locales
        n_maps , height , width , channels = imageMaps.shape
        K_height , K_width = self.window_size
        stride            = self.stride
    
        #vérification des dimensions de la sortie
        height_out = (height - K_height) // stride + 1
        width_out  = (width  - K_width)  // stride + 1
        assert height_out > 0 and width_out > 0 , f"Les dimensions de sortie sont invalides : height_out={height_out}, width_out={width_out}"
    
        #extraction des fenêtres de pooling (N , H_out , W_out , KH , KW , C)
            #transposition de l'imageMaps pour qu'elle devienne (N , C , h , W)
        imageMaps_NCHW = np.transpose(imageMaps , (0 , 3 , 1 , 2))
        
        windows = sliding_window_view(imageMaps_NCHW , (K_height , K_width) , axis= (2 , 3))
        windows = windows[: , : , ::stride , ::stride , : , :]

        #transposition des axes de windows pour obtenir (N, H_out, W_out, KH, KW, C)
        windows = np.transpose(windows, (0, 2, 3, 4, 5, 1))
    
        #application du pooling (maxPooling ou averagePooling)
        if self.pooling_type == "max":
            pooled = np.max(windows , axis= (3 , 4))   
        else:
            pooled = np.mean(windows , axis= (3 , 4)) 

        #validation de la sortie
        required_shape = (n_maps , height_out , width_out , channels)
        assert pooled.shape == required_shape , f"Le shape de la sortie est {pooled.shape} alors que le shape attendu est {required_shape} !"
        assert np.all(np.isfinite(pooled)) , "La sortie contient des valeurs 'NaN' ou 'inf' !"
    
        return pooled

    
    def backward(self , inputGradient: np.ndarray) -> np.ndarray:
        """
        Elle reçoit le gradient de la couche de devant 'inputGradient' et calcule le gradient selon le type de pooling 'average' ou 'max'
        qu'elle retourne à la couche d'avant sous forme un tableau nD de même shape que celui de l'entrée de la couche pendant le forward.
            <> inputGradient de forme ( N , (H-k1)//s + 1 , (W-k2)//s + 1 , C)
            <> outputGradient de forme (N , H , W , C)
        """
        #validation de l'entrée
        assert isinstance(inputGradient , np.ndarray) and inputGradient.ndim == 4, \
            "L'entrée 'inputGradient' doit être un tableau 4D de type 'ndarray' !"
    
        #récupération des dimensions de l'entrée de la couche pendant le forward 
        n_maps , height , width , channels = self.inputMaps.shape
        K_height , K_width = self.window_size
        stride             = self.stride
    
        #vérification de shape de gradient reçu 
        H_in = (height - K_height) // stride + 1
        W_in = (width  - K_width) // stride + 1
        assert inputGradient.shape == (n_maps , H_in , W_in , channels), \
                                  f"Le shape du gradient attendu est {(n_maps , H_in , W_in , channels)} mais reçu {inputGradient.shape} !"
    
        #initialisation du gradient de sortie
        outputGradient = np.zeros_like(self.inputMaps)
    
        if self.pooling_type == "max":
            #construction du masque (position du max pendant le forward)
            imageMaps_NCHW = np.transpose(self.inputMaps , (0 , 3 , 1 , 2))                          # (N, C, H, W)
            windows = sliding_window_view(imageMaps_NCHW , (K_height , K_width) , axis= (2 , 3))
            windows = windows[: , : , ::stride , ::stride , : , :]                                    # (N, C, H_out, W_out, KH, KW)
            windows = np.transpose(windows , (0 , 2 , 3 , 4 , 5 , 1))                                  # (N, H_out, W_out, KH, KW, C)
    
            #masque des maximums avec le même shape 
            max_mask = (windows == np.max(windows , axis= (3 , 4) , keepdims= True))  
    
            #expansion du gradient vers chaque position max
            distributed_gradient = inputGradient[ : , : , : , np.newaxis , np.newaxis , :] * max_mask
            
            #reconstruction de gradient complet par rapport à l’entrée
            for i in range(H_in):
                for j in range(W_in):
                    outputGradient[: , i*stride:i*stride+K_height , j*stride:j*stride+K_width , :] += distributed_gradient[: , i , j , : , : , :]
  
        #cas de l'average Pooling
        else:  
            #partage équitable de gradient entre les cases de la fenêtre
            distributed_gradient = inputGradient / (K_height * K_width)
            
            #reconstruction de gradient complet par rapport à l’entrée
            for i in range(H_in):
                for j in range(W_in):
                    outputGradient[ : , i*stride:i*stride+K_height , j*stride:j*stride+K_width , :] += \
                                                                    distributed_gradient[: , i , j , :][: , np.newaxis , np.newaxis , :]

        #validation de sortie
        assert outputGradient.shape == self.inputMaps.shape , f"Le shape de sortie est {outputGradient.shape}, attendu : {self.inputMaps.shape} !"
        assert np.all(np.isfinite(outputGradient)), "Le gradient contient des valeurs NaN ou inf !"
    
        return outputGradient
    

##################################################  Classe Dense Layer  #############################################################################
                                             #(la classe a été testée avec succes)
class DenseLayer:
    """
        Cette classe permet de créer des couches 'Fully Connected' de réseaux de neurones en signalant lors de l'instanciation: 
            <> numberOfNeurons: le nombre de neurones dans la couche.
            <> activationFonction: la fonction d'activation de la couche['sigmoid' , 'relu' , 'softmax'] (par défaut: 'ReLU').
            <> numberOfInputs: le nombre d'entrées  de la couche: c-à-d le nombre de neronnes de la couche précédente ou 
               le nombre de features de dataset
            NB: le softmax s'utilise en couche de sortie pour prédir des classes (il besoin de lui passer un gradient qui utilise y_true codé en One-Hot)
    """

    def __init__(self , numberOfNeurons: int , numberOfInputs: int , activationFunction: str= 'relu'):

        """
            Le constructeur des objets de classe 'DenseLayer' qui prend trois arguments et retourne un objet de type DenseLayer.

        """
        #validation des entrées
        assert isinstance(numberOfNeurons , (int , np.integer)) and numberOfNeurons > 0 , \
                                                                           "Le nombre de neurones doit être un entier strictement positif !"
        assert isinstance(activationFunction , str) and activationFunction in ['sigmoid' , 'relu' , 'softmax' , 'tanh'] , \
                                                               "La fonction d'activation doit être soit 'sigmoid' ou 'relu'!"
        assert isinstance(numberOfInputs , (int , np.integer)) , "Le nombre d'entrées pour la couche doit être un entier positive !"

        # Initialisation des poids et des biais et d'autres attributs de la couche
        self.numberOfNeurons    = numberOfNeurons
        self.activationFunction = activationFunction
        self.numberOfInputs     = numberOfInputs
        self.Z = None
        self.A = None

        self.trainable = True

        # on fixe le random state par le seed
        np.random.seed(42)
        
        fan_in = self.numberOfInputs
        self.weights = np.random.randn(fan_in, self.numberOfNeurons) * np.sqrt(2. / fan_in)
        self.biases  = np.zeros((1 , self.numberOfNeurons))

    def forward(self , inputMatrix: np.ndarray) -> tuple:
        """
            Calcule de Z et A pour la couche en exploitant l'entrée qui peut être une activation ou le dataset selon la position de la couche
        """
        #Validation de l'argument
        assert isinstance(inputMatrix , np.ndarray) , "L'argument n'est pas de type ndarray de numpy"
        assert inputMatrix.ndim == 2 , "L'entrée doit être un tableau 2D !"
        assert inputMatrix.shape[1] == self.numberOfInputs , "la deuxième dimension de la rentrée pour cette couche est incorrecte!"

        #sauvegarde de l'entrée pour en servir lors de backward
        self.X = inputMatrix

        #Calcul de Z
        self.Z = np.dot(inputMatrix , self.weights) + self.biases
        #validation de Z
        assert (self.Z.shape[0] == inputMatrix.shape[0]) | (self.Z.shape[1] == self.numberOfNeurons)  , "la deuxième dimension de Z  est incorrecte!"
        
        #Calcule de A selon la fonction passée pendant la création de la couche
        if self.activationFunction == "relu":
            self.A = self.relu(self.Z)
        elif self.activationFunction == "sigmoid":
            self.A = self.sigmoid(self.Z)
        elif self.activationFunction == "softmax":
            self.A = self.softmax(self.Z)
        else:
            self.A = self.tanh(self.Z)
            
        #Validation de la sortie
        assert self.Z.shape == self.A.shape , "A et Z doivent avoir le même shape !"
        assert np.all(np.isfinite(self.A)) ,"La sortie ne doit pas contenir des valeurs indéfinies 'NaN' ou 'inf' !"

        return self.A
        

    def backward(self , inputGradient: np.ndarray) -> np.ndarray:
        """
            Elle reçoit le gradient de la couche de devant et calcule les gradients par rapport à ses prpores poids et biais et le gradient
            par rapport à son entrée qu'il passe à la couche d'avant.
            NB: si on a choisie Softmax comme activation on doit fournir les y_true.
        """
        #validation de l'entrée
        assert isinstance(inputGradient , np.ndarray) , "L'entrée doit être de type 'ndarray' de 'numpy' !"
        assert inputGradient.ndim == 2 , "L'entrée doit être un tableau 2D !"
        
        #Calcule de dZ selon la fonction d'activation
        if self.activationFunction == "relu":
            dZ = inputGradient * self.relu_derivative(self.Z)
            
        elif self.activationFunction == "sigmoid":
            dZ = inputGradient * (self.A * (1 - self.A))                           #self.sigmoid_derivative(self.Z)
            
        elif self.activationFunction == "softmax":
            dZ = inputGradient                #softmax avec cross-Entropy dans la couche de sortie: inputGradient = y_pred - y_true
            
        else:
            dZ = inputGradient * self.tanh_derivative(self.Z)

        #calcul des gradients d_Weights , d_biases et d_X
        self.d_Weights = np.dot(self.X.T , dZ)
        self.d_Biases  = np.sum(dZ , axis= 0 , keepdims= True)
        self.d_X       = np.dot(dZ , self.weights.T)

        #validation des résultats
        assert self.d_Weights.shape == self.weights.shape , \
                 f"Les gradients des poids a comme shape {self.d_Weights.shape} ce qui est incohérent avec le shape des poids {self.weights.shape} !"
        assert self.d_Biases.shape == self.biases.shape , \
                    f"Les gradients des biais a comme shape {self.d_Biases.shape} ce qui est incohérent avec le shape des biais {self.biases.shape} !"
        assert self.d_X.shape == self.X.shape , \
                    f"Le gradient à propager a comme shape {self.d_X.shape} ce qui est incohérent avec le shape de l'entrée {self.X.shape} !"

        assert np.all(np.isfinite(self.d_Weights)) , "Les gradients des poids contiennent des valeurs 'Nan' ou 'inf' !"
        assert np.all(np.isfinite(self.d_Biases)) , "Les gradients des biais contiennent des valeurs 'Nan' ou 'inf' !"
        assert np.all(np.isfinite(self.d_X)) , "Le gradient à propager contient des valeurs 'Nan' ou 'inf' !"

        return self.d_X

    #les fonctions d'activation et leurs dérivées

    def relu(self , X: np.ndarray) -> np.ndarray:
        """
          traite les valeurs des cartes d'entrée et les modifie selon la logique : max(0 , x) 
        """
        #Validation des arguments
        assert isinstance(X , np.ndarray) , "L'argument doit être de type 'ndarray' de numpy!"

        #calcule des activations
        result = np.maximum( 0 , X )
        #validation de résultat
        assert np.all( result >= 0 ) , "l'activation selon ReLU ne doit pas contenir des valeurs strictement négatives!"

        return result

    def tanh(self , X: np.ndarray) -> np.ndarray:
        """
          traite les valeurs des cartes d'entrée et les modifie selon comme suit: np.tanh(x) 
        """
        #Validation des arguments
        assert isinstance(X , np.ndarray) , "L'argument doit être de type 'ndarray' de numpy!"

        #calcule des activations
        result = np.tanh(X)
        
        #validation de résultat
        assert np.all((result >= -1) & (result <= 1)) , "l'activation selon 'tanh' ne doit pas contenir des valeurs < -1 ou > 1 !"

        return result

    def sigmoid(self , Z: np.ndarray) -> np.ndarray:
        """
         Calcule les sorties des neurones sous forme de probabilité: elle s'applique à chaque élément de tableau d'entrée
        """
        #validation de l'entrée
        assert isinstance( Z , np.ndarray) , " l'entrée doit être de type ndarray de numpy"
        assert Z.size > 0 , "le tableau d'entrée ne doit pas être vide"
    
        result = 1/( 1 + np.exp( - Z ))
        
        # Les vérifications des  conditions: σ(z) est dans l'intervalle [0; 1], σ'(z) ≥ 0 et σ'(0) = 0.25
        assert np.all(result >= 0) and np.all(result <= 1) , "toutes les valeurs de activate_result doivent être comprises en 0 et 1"
        assert np.all(result * (1 - result) >= 0  ) , "La dérivée de la sigmoide contient des valeurs négatives"
    
        test = 1 / (1 + np.exp(-np.zeros((1,1))))
        assert np.isclose(test*(1 - test) , 0.25) , "La dérivée de la sigmoide de 0 vaut toujours 0.25"
                   
        return result

    def softmax(self , X: np.ndarray) -> np.ndarray :
        """
            calcule les probabilités d'appartenir à une classe (dans notre cas 33 classe chacune représente un alphabet de Tifinagh)
        """
        #validation de l'entrée
        assert isinstance(X , np.ndarray), "L'argument doit être de type 'ndarray' de 'numpy' !"
        
        # pour éviter les valeurs 'inf' <==> exp de grandes valeurs tends vers l'infini
        maxLine = np.max(X , axis=1 , keepdims=True)
        expOfX  = np.exp(X - maxLine)

        #calcule de la sortie en utilisant la formule de softmax (les rapport exp(xi)/ sum(xj) , j= les valeurs de chaque ligne)
        result = expOfX / np.sum(expOfX, axis=1, keepdims=True)

        #validation de la sortie
        assert isinstance(result , np.ndarray) and result.shape == X.shape , "La sortie doit être de même type et a la même shape que l'entrée !"
        assert np.all((result >= 0) & (result <= 1)), "Les valeurs de la sortie doivent appartenir à [0, 1] !"
        assert np.allclose(np.sum(result , axis=1), 1), "Softmax doit retourner des valeurs dont la somme vaut 1 pour chaque enregistrement !"
        
        return result



    def relu_derivative(self , X: np.ndarray) -> np.ndarray:
        """
          Calcule les valeurs de la dérivée de ReLU sur les éléments de la matrice d'entrée: Relu'(x) = 1 si x>0 , 0 sinon
        """
        #Validation de l'entrée
        assert isinstance(X , np.ndarray) , "L'argument doit être de type 'ndarray' de numpy!"

        #calcule des activations
        result = (X > 0).astype(float)
        
        #validation de résultat
        assert np.all((result == 0) | (result == 1)) , \
                      "La dérivée de l'activation selon ReLU ne doit pas retouner un résultat contenant des valeurs < 0 ou > 1 !"

        return result

    def tanh_derivative(self , X: np.ndarray) -> np.ndarray:
        """
          Calcule les valeurs de la dérivée de Tanh sur les éléments de la matrice d'entrée : tanh'(x) = 1 - tanh²(x)
        """
        #Validation de l'entrée
        assert isinstance(X , np.ndarray) , "L'argument doit être de type 'ndarray' de numpy!"

        #calcule des activations
        result = 1 - np.square(self.tanh(X))
        
        #validation de résultat
        assert np.all((result >= 0) & (result <= 1)) , \
                      "La dérivée de l'activation selon Tanh ne doit pas retouner un résultat contenant des valeurs < 0 ou > 1 !"

        return result

##################################################  Classe Flatten  ##################################################################################
                                           #(la classe à été testée avec succes)
class Flatten:
    """
         joue un rôle de passerelle entre les couches de convolution et les couches dense en aplatissant les cartes produites par 
         les couches de convolution en vecteur pour servir les couches fully connected et fait l'inverse pendant la rétropropagation
    """ 
    def __init__(self):
        """
         Le constructeur de la classe Flatten
        """
        #attribut d'objet pou garder le shape de l'entrée
        self.input_shape = None

        self.trainable = False
        
    def forward(self , inputMaps: np.ndarray) -> np.ndarray:
        """
            aplatit l'entré 'inputMaps' en vecteur afin de le transmettre à la première couche dense de réseau Fully connected.
        """
        #validation de l'entrée
        assert isinstance(inputMaps , np.ndarray) and inputMaps.ndim == 4 , "L'entrée doit être de type 'ndarray' nD !"
        
        #sauvegarde de shape de l'entrée dans un attribut pour en servir lors de backpropagation
        self.input_shape = inputMaps.shape
        numberOfImages , n_filters , height , width  = self.input_shape
        m = n_filters*height*width
        #aplatissement de l'entrée
        output = inputMaps.reshape(numberOfImages , m)

        #validation de la sortie
        assert output.shape == (numberOfImages , m) , f"La sortie a comme shape {output.shape}, alors que le shape attendu est ({numberOfImages},{m} ) !"
        
        return output
        

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
            fransforme une vecteur gradient en cartes pour servir les couches de convolution-pooling pendant la rétropropagation
        """
        #validation de l'entrée
        assert isinstance(gradient , np.ndarray) , "L'entrée doit être de type 'ndarray' de 'numpy' !"

        #transformation de l'entrée en cartes pour reproduire le shape de départ
        output = gradient.reshape(self.input_shape)

        #validation de la sortie
        assert output.shape == self.input_shape , f"Le shape de la sortie est {output.shape} alors qu'on attend {self.input_shape} !"
        
        return output