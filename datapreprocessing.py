#module datapreprocessing

"""
    Ce module contient deux classes:
      <>ImagesDataPreprocessingForMLP -> fournit des ensembles X_train, y_train, X_val, y_val, X_test, y_test prêts à utilisés dans un MLP.
      <>ImagesDataPreprocessingForCNN -> fournit des ensembles X_train, y_train, X_val, y_val, X_test, y_test prêts à utilisés dans un CNN.
      <>PreprocessingData             -> fournit des ensembles X_train, y_train, X_val, y_val, X_test, y_test à partir des données tabulaires prêts à utilisés
                                         dans des MLP et des modèles de machine learning.
    NB: mode d'emploi de ces classe:
                                     ___________________________________________________________________________________________________________
                                     |PreprocessingData           | ImagesDataPreprocessingForCNN  |     ImagesDataPreprocessingForMLP         |
                                     |_________________________________________________________________________________________________________|
                                     |instanciation d'un objet    |     méthodes de classe         |     méthodes de classe                    |
                                     |Objet.fit_transform(...)    | Class.fit_transform(...)       |     Class.fit_transform                   |
                                     |____________________________|________________________________|___________________________________________|
"""

#importations
from tqdm import tqdm #sert à afficher les barre de progression d'exécution.

import pandas as pd
import numpy as np
from scipy.signal import correlate2d
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.model_selection import train_test_split
import os
import cv2



################################################### classe imagesDataPreprocessing ##########################################################################

class ImagesDataPreprocessingForMLP:
    """
    la classe rassembles des méthodes de prétraitement des données images et le sur-échantillonage. ces méthodes sont des méthodes de la classe donc 
    elles vont être appelées sans l'instanciation de la classe.
    la méthode fit_transforme() exécute le pipeline de prétaitement et fournit comme sortie les ensembles de train, validation et de test avec les target
    codés en One-hot pour être compatible avec la fonction d'activation softmax et la fonction de perte 'cross-entropy'
    """
    def __init__(self):
        pass
    
    @classmethod
    def loadData_csvFile(self , path: str , file_namesAndLabelsOfImages: str) -> pd.DataFrame:
        #le chargement du fichier contenant les chemins vers les images et les labels en les mettant dans un 
        #DataFrame qui fera l'objet de return
        #validation de l'entrée
        assert isinstance(path , str) , "L'argument 1 passée à cette méthode doit être un chemin vers un dossier 'string' !"
        assert isinstance(file_namesAndLabelsOfImages , str) , "L'arguement 2 doit être un nom d'un fichier csv 'String' !"

        #définition du chemin du dossier contenant les images et le fichier csv (noms des images + labels)
        pathOfDataDirectory = os.path.join(os.getcwd() , path)

        #chargement du fichier csv contenant à la fois les noms des images et leurs étiquettes
        # Charger le fichier CSV contenant les tiquettes
        try:
            labelsAndPahs = pd.read_csv(os.path.join(pathOfDataDirectory , file_namesAndLabelsOfImages) , header= None)
            #ajouter les entêtes aux DataFrame
            labelsAndPahs.columns = ['image_path', 'label'] 
            
            """
             Supprimer le préfixe "images-data-64/" du chemin si présent et cela concerne seulement notre tp3 car le fichier csv contient 
            un répertoire qui ne figure dans le vrai chemin et en plus des slash à la place des anti-slash
            """
            labelsAndPahs['image_path'] = labelsAndPahs['image_path'].apply(lambda p: p[len("./images-data-64/"):]
                                                                            if p.startswith("./images-data-64/") else p)

            #vérification de l'existance des noms des images et de leurs étiquettes dans le fichier csv
            assert (
                'image_path' in labelsAndPahs.columns and 'label' in labelsAndPahs.columns 
            ), "Le fichier doit contenir les noms des fichiers images et le étiquettes de chacune d'elles !"

            #ajout du répertoire racine aux chemins de toutes les images
            labelsAndPahs['image_path'] = labelsAndPahs['image_path'].apply(lambda x: 
                                                                            os.path.normpath(os.path.join(pathOfDataDirectory, x.lstrip("/\\")))) 
            
            print(f"Téléchargement de {len(labelsAndPahs)} échantillons avec {labelsAndPahs['label'].nunique()} classes")
            
            return labelsAndPahs
            
        except FileNotFoundError:
            print(f"Le fichier {file_namesAndLabelsOfImages} est introuvable !")
            return pd.DataFrame()

    @classmethod
    def loadDataFromDirectory(self , path: str) -> pd.DataFrame:
        # Consruction du DataFrame à partir du répertoire contenant des images
        #validation de l'entrée
        assert isinstance(path , str) , "L'argument doit être le chemin vers le répertoire des images sous forme de string !"
        imagePaths = []   #liste des chemins complets des images
        labels = []       #liste des étiquettes des images
        #définition du chemin du dossier contenant les images et le fichier csv (noms des images + labels)
        pathOfDataDirectory = os.path.join(os.getcwd() , path)
        
        for subDirName in os.listdir(pathOfDataDirectory):
            subDirPath = os.path.join(pathOfDataDirectory , subDirName)
            if os.path.isdir(subDirPath):
                for imageName in os.listdir(subDirPath):
                    imagePaths.append(os.path.join(subDirPath , imageName))
                    labels.append(subDirPath)
        
        # Création du DataFrame
        data_labels_imagesPaths = pd.DataFrame({'image_path': imagePaths , 'label': labels})
        
        # Vérification du DataFrame avant de le renvoyer
        assert not data_labels_imagesPaths.empty, "Le data n'est pas téléchargé, vérifier les chemins et le dossier source !"
        
        print(f"Téléchargement de {len(data_labels_imagesPaths)} échantillons avec {data_labels_imagesPaths['label'].nunique()} classes")

        return data_labels_imagesPaths

    @classmethod
    def labelsEncoder(self , data: pd.DataFrame , labels: str ) -> pd.DataFrame:
        #Encodage  des labels des images
        #validation des entrées
        assert isinstance(data , pd.DataFrame) , "L'argument 1 doit être une DataFrame !"
        assert isinstance(labels , str) and labels in data.columns , "L'argument 2 doit être un nom d'une colonne de DataFrame 'arg1' !"
        
        encoder = LabelEncoder()
        data[labels] = encoder.fit_transform (data[labels])
        
        num_classes = len(encoder.classes_)
        #validation de l'opération de l'encodage
        assert data[labels].nunique() == num_classes , "L'opération de l'encodage n'est pas bien aboutie !"

        return data

    @classmethod
    def loadAndProcessImage(self , imagePath: str , sizeOfImage: tuple[int , int]) -> np.ndarray:
        #charge l'image et la redimensionner avant de la rendre sous forme d'une matrice
        #validatio des entrées
        assert isinstance(imagePath , str) , "L'argument 1 est un chemin vers une image sous forme de 'String' !"
        assert os.path.exists (imagePath) , f"L'image est introuvable: {imagePath } !"
        assert (
            isinstance(sizeOfImage , tuple) and np.all(isinstance(number , int)  and  number > 0 for number in sizeOfImage) 
        ) , "La taille de l'image doit être un tuple de deux entiers strictement positifs !"

        #mettre l'image en gris
        image = cv2.imread(imagePath , cv2.IMREAD_GRAYSCALE)
        assert image is not None , f"L'image est introuvable: {imagePath } !"
        #redimensionner l'image
        image = cv2.resize (image , sizeOfImage )
        #normaliser l'image (x - min)/(max-min)
        image = image.astype (np.float32 )/255.0 

        #validation de la sortie
        assert isinstance(image , np.ndarray) and np.all((image >= 0) & (image <= 1)) , "l'image n'était pas traîtée correctement !"
        assert image.shape == sizeOfImage , "Le redimensionnement de l'image n'était pas fait correctement !"
        
        return image.flatten()

    @classmethod
    def processAllImages(self , labelsAndPaths: pd.DataFrame , sizeOfImage: tuple[int , int]) -> np.ndarray:
        #transforme toutes les images en matrice de données et la target en vecteur
        #validation des entrées
        assert isinstance(labelsAndPaths , pd.DataFrame) , "L'argument 1 doit être de type 'DataFrame' de pandas !"
        assert (
                'image_path' in labelsAndPaths.columns and 'label' in labelsAndPaths.columns 
                ), "Le Dataframe 'argument 1' doit contenir les chemins vers les fichiers images et le étiquettes de chacune d'elles !"
        assert (
                  isinstance(sizeOfImage , tuple) and np.all(isinstance(number , int)  and  number > 0 for number in sizeOfImage) 
                ) , "La taille de l'image doit être un tuple de deux entiers strictement positifs !"
        
        #prétaitement des images
        X = np.array([self.loadAndProcessImage(path , sizeOfImage) for path in labelsAndPaths['image_path']])
        y = self.labelsEncoder(labelsAndPaths , 'label' )['label'].values

        # vérification des  dimensions
        assert X.shape[0] == y.shape[0] , "Dimension incohérent entre les données X et target y !"
        assert (
                  X.shape[1] == sizeOfImage[0] * sizeOfImage[1]
        ), f"le nombre de features dans X {X.shape[1]} doit correspondre à {sizeOfImage[0] * sizeOfImage[1]} !"

        return X , y


    @classmethod
    def splitData(self , X: np.ndarray , y: np.ndarray , test_size: float , stratification: bool= True , 
                  sameResultOfSplit: bool= True)  -> tuple[np.ndarray , ...]:
        #divise les données et le traget en sous ensembles de train, de validation et de test
        #validation des entrées
        assert isinstance(X , np.ndarray) , "l'argument1 doit être de type ndarray de numpy !"
        assert isinstance(y , np.ndarray) , "l'argument 2 doit être de type ndarray de numpy !"
        assert X.shape[0] == y.shape[0] , "des données et le target doivent avoir le même nombre d'enregistrements !"
        assert isinstance(test_size , float) and  1 > test_size > 0 , "la taille de test doit être un réel entre 0 et 1 !"
        assert isinstance(stratification , bool) , "L'argument 3 'stratification' doit prendre soit 'True' soit 'False' comme valeur!"
        assert isinstance(sameResultOfSplit , bool) , "L'argument 4 'sameResultOfSplit' doit prendre soit 'True' soit 'False' comme valeur!"
        

        #division en train et test selon les paramètres 'booleen' passés à la méthode
        if stratification:
            if sameResultOfSplit:
                X_train , X_test , y_train , y_test = train_test_split(X , y , test_size= test_size  , stratify= y , random_state= 42)
            else:
                X_train , X_test , y_train , y_test = train_test_split(X , y , test_size= test_size , stratify= y)
        else:
            if sameResultOfSplit:
                X_train , X_test , y_train , y_test = train_test_split(X , y , test_size= test_size , random_state= 42)
            else:
                X_train , X_test , y_train , y_test = train_test_split(X , y , test_size= test_size)

        #validation des résultat avant de les retourner
        assert X_train.shape[0] == y_train.shape[0] , "X_train et y_train doivent avoir le même nombre d'échantillons!"
        assert X_train.shape[0] == y_train.shape[0] , "X_test et y_test doivent avoir le même nombre d'échantillons!"
        assert X_train.shape[1] == X_test.shape[1] , "X_train et X_test doivent avoir le même nombre de variables!"

        return X_train , X_test , y_train , y_test

    @classmethod
    def oneHotEncoder(self , y: np.ndarray) -> np.ndarray:
        #encode les labels en one-hot: sous forme de veteur contenant des zéros et un
        #validation de l'entrée
        assert isinstance(y , np.ndarray) and y.ndim == 1 , "L'argument doit être de type ndarray d'une seule dimension !"

        y_one_hot = np.array(OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1)))

        #validation de la sortie
        assert (
            isinstance(y_one_hot , np.ndarray) and y_one_hot.shape == (y.shape[0] , len(np.unique(y)))
        ) , "Le résultat de oneHotEncoder n'est un ndarray ou ses dimensions sont incorrectes !"

        return y_one_hot

    @classmethod
    def fit_transform(self , path: str , file_namesAndLabelsOfImages: str= None , sizeOfImage: tuple[int , int]= (32 , 32) , val_size: float= 0.2 ,
                      test_size: float= 0.2 ,  stratification: bool= True , sameResultOfSplit: bool= True) -> tuple[np.ndarray, ...]:
        #cette méthode est un pipeline qui fait appel à les autres méthode pour charger, prétraiter et diviser le data et retourne en fin 
        #retourner les ensembles de train, de validation et de test.
        #validation des l'entrées
        assert isinstance(path , str) , "L'argument 1 passée à cette méthode doit être un chemin vers un dossier 'string' !"
        assert isinstance(file_namesAndLabelsOfImages , (type(None) , str)) , "L'arguement 2 doit être un nom d'un fichier csv 'String' ou None !"
        assert (
            isinstance(sizeOfImage , tuple) and np.all(isinstance(number , int)  and  number > 0 for number in sizeOfImage) 
        ) , "La taille de l'image doit être un tuple de deux entiers strictement positifs !"
        assert isinstance(test_size , float) and  1 > test_size > 0 , "la taille de test doit être un réel entre 0 et 1 !"
        assert isinstance(val_size , float) and  1 > val_size > 0 , "la taille de validation doit être un réel entre 0 et 1 !"
        assert isinstance(stratification , bool) , "L'argument 5 'stratification' doit prendre soit 'True' soit 'False' comme valeur!"
        assert isinstance(sameResultOfSplit , bool) , "L'argument 6 'sameResultOfSplit' doit prendre soit 'True' soit 'False' comme valeur!"

        #chargement du data selon le deux cas: à partir d'un fichier csv ou à partir d'un répertoire
        if file_namesAndLabelsOfImages == None:
            data_dataFrame = self.loadDataFromDirectory(path)
        else:
            data_dataFrame = self.loadData_csvFile(path , file_namesAndLabelsOfImages)

        #encodage des labels
        data_dataFrame = self.labelsEncoder(data_dataFrame , 'label')
        #transformation des images en gris, les redimensionner et les transformer en matrices et la normalisation
        X , y = self.processAllImages(data_dataFrame , sizeOfImage)
        #division des données en train, validation et test
        X_temp , X_test , y_temp , y_test = self.splitData(X , y , test_size , True , True)
        X_train , X_val , y_train , y_val = self.splitData(X_temp , y_temp , val_size*1.25 , True , True)

        #Encodage One-Hot des targets
        y_train = self.oneHotEncoder(y_train)
        y_val = self.oneHotEncoder(y_val)
        y_test = self.oneHotEncoder(y_test)

        #validation des sorties
        assert X_train.shape[0] == y_train.shape[0] , "les premières dimensions de X_train et y_train doit être égales !"
        assert X_val.shape[0] == y_val.shape[0] , "les premières dimensions de X_val et y_val doit être égales !"
        assert X_test.shape[0] == y_test.shape[0] , "les premières dimensions de X_test et y_test doivent être égales !"
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1] , "X_train, X_val, X_test doivent avoir le même nombre de features !"
        assert y_train.shape[1] == y_val.shape[1] == y_test.shape[1] , "y_train, y_val, y_test doivent avoir la même deuxième dimension !"

        return X_train , X_val , X_test , y_train , y_val , y_test



########################################### Classe prePorcessingDataForCNN  #################################################################################

class ImagesDataPreprocessingForCNN:
    """
    Cette Classe fournit des données images prêtes à utiliser dans un modèle CNN.
    Elle rassemble des méthodes de prétraitement des données images et le sur-échantillonage. ces méthodes sont des méthodes de la classe donc 
    elles vont être appelées sans l'instanciation de la classe.
    la méthode fit_transforme exécute le pipeline de prétaitement et fournit comme sortie les ensembles de train, validation et de test avec les target
    codés en One-hot pour être compatible avec la fonction d'activation softmax et la fonction de perte 'cross-entropy'
    """
    def __init__(self):
        pass
        
    @classmethod
    def loadData_csvFile(self , path: str , file_namesAndLabelsOfImages: str) -> pd.DataFrame:
        """
         le chargement du fichier contenant les chemins vers les images et les labels en les mettant dans un DataFrame qui sera l'objet de return
         
        """
        #validation de l'entrée
        assert isinstance(path , str) , "L'argument 1 passée à cette méthode doit être un chemin vers un dossier 'string' !"
        assert isinstance(file_namesAndLabelsOfImages , str) , "L'arguement 2 doit être un nom d'un fichier csv 'String' !"

        #définition du chemin du dossier contenant les images et le fichier csv (noms des images + labels)
        pathOfDataDirectory = os.path.join(os.getcwd() , path)

        #chargement du fichier csv contenant à la fois les noms des images et leurs étiquettes
        try:
            labelsAndPahs = pd.read_csv(os.path.join(pathOfDataDirectory , file_namesAndLabelsOfImages) , header= None)
            #ajouter les entêtes aux DataFrame
            labelsAndPahs.columns = ['image_path', 'label'] 
            
            """
             Supprimer le préfixe "images-data-64/" du chemin si présent et cela concerne seulement notre tp3 car le fichier csv contient 
            un répertoire qui ne figure pas dans le vrai chemin et en plus des slash à la place des anti-slash
            """
            labelsAndPahs['image_path'] = labelsAndPahs['image_path'].apply(lambda p: p[len("./images-data-64/"):]
                                                                            if p.startswith("./images-data-64/") else p)

            #vérification de l'existance des noms des images et de leurs étiquettes dans le fichier csv
            assert (
                'image_path' in labelsAndPahs.columns and 'label' in labelsAndPahs.columns 
            ), "Le fichier doit contenir les noms des fichiers images et le étiquettes de chacune d'elles !"

            #ajout du répertoire racine aux chemins de toutes les images
            labelsAndPahs['image_path'] = labelsAndPahs['image_path'].apply(lambda x: 
                                                                            os.path.normpath(os.path.join(pathOfDataDirectory, x.lstrip("/\\")))) 
            
            print(f"Téléchargement de {len(labelsAndPahs)} échantillons avec {labelsAndPahs['label'].nunique()} classes")
            
            return labelsAndPahs
            
        except FileNotFoundError:
            print(f"Le fichier {file_namesAndLabelsOfImages} est introuvable !")
            return pd.DataFrame()

    @classmethod
    def loadDataFromDirectory(self , path: str) -> pd.DataFrame:
        # Consruction du DataFrame à partir du répertoire contenant des images
        #validation de l'entrée
        assert isinstance(path , str) , "L'argument doit être le chemin vers le répertoire des images sous forme de string !"
        imagePaths = []   #liste des chemins complets des images
        labels = []       #liste des étiquettes des images
        #définition du chemin du dossier contenant les images et le fichier csv (noms des images + labels)
        pathOfDataDirectory = os.path.join(os.getcwd() , path)
        
        for subDirName in os.listdir(pathOfDataDirectory):
            subDirPath = os.path.join(pathOfDataDirectory , subDirName)
            if os.path.isdir(subDirPath):
                for imageName in os.listdir(subDirPath):
                    imagePaths.append(os.path.join(subDirPath , imageName))
                    labels.append(subDirPath)
        
        # Création du DataFrame
        data_labels_imagesPaths = pd.DataFrame({'image_path': imagePaths , 'label': labels})
        
        # Vérification du DataFrame avant de le renvoyer
        assert not data_labels_imagesPaths.empty, "Le data n'est pas téléchargé, vérifier les chemins et le dossier source !"
        
        print(f"Téléchargement de {len(data_labels_imagesPaths)} échantillons avec {data_labels_imagesPaths['label'].nunique()} classes")

        return data_labels_imagesPaths

    @classmethod
    def labelsEncoder(self , data: pd.DataFrame , labels: str ) -> pd.DataFrame:
        #Encodage  des labels des images
        #validation des entrées
        assert isinstance(data , pd.DataFrame) , "L'argument 1 doit être une DataFrame !"
        assert isinstance(labels , str) and labels in data.columns , "L'argument 2 doit être un nom d'une colonne de DataFrame 'arg1' !"
        
        encoder = LabelEncoder()
        data[labels] = encoder.fit_transform (data[labels])
        
        num_classes = len(encoder.classes_)
        #validation de l'opération de l'encodage
        assert data[labels].nunique() == num_classes , "L'opération de l'encodage n'est pas bien aboutie !"

        return data

    @classmethod
    def loadAndProcessImage(self , imagePath: str , sizeOfImage: tuple[int , int] , mode: str= 'RGB') -> np.ndarray:
        """
            charge l'image et la redimensionner avant de la rendre sous forme d'un'un tableau ((n , n , 3) -> RGB ou (n , n , 1) -> niveau de gris)
            selon le choix passé par l'argument 'mode'.
        """
        #validation des entrées
        assert isinstance(imagePath , str) , "L'argument 1 est un chemin vers une image sous forme de 'String' !"
        assert os.path.exists (imagePath) , f"L'image est introuvable: {imagePath } !"
        assert (
            isinstance(sizeOfImage , tuple) and np.all(isinstance(number , int)  and  number > 0 for number in sizeOfImage) 
        ) , "La taille de l'image doit être un tuple de deux entiers strictement positifs !"
        assert isinstance(mode , str) and mode in ['RGB' , 'Gray'] , "l'argument 3 'mode' doit être 'RGB' ou 'Gray' !"

        if mode == 'Gray':
            #mettre l'image en gris 
            image = cv2.imread(imagePath , cv2.IMREAD_GRAYSCALE)
            
        else:
            #l'image en RGB
            image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
            
        assert image is not None , f"L'image est introuvable: {imagePath } !"
        #redimensionnement de l'image
        image = cv2.resize(image , sizeOfImage)
        
        #normaliser l'image (x - min)/(max-min)
        image = image.astype(np.float32 )/255.0 

        #validation de la sortie
        assert isinstance(image , np.ndarray) and all(np.all((card >= 0) & (card <= 1)) for card in image) , \
                                                                                                    "L'image n'était pas traîtée correctement !"
        if mode == 'Gray':
            assert image.shape == (sizeOfImage) , "Le redimensionnement de l'image n'était pas fait correctement !"
        else:
            assert image.shape == sizeOfImage + (3,) , "Le redimensionnement de l'image n'était pas fait correctement !"
        
        return image

    @classmethod
    def processAllImages(self , labelsAndPaths: pd.DataFrame , sizeOfImage: tuple[int , int] , mode: str= 'RGB') -> np.ndarray:
        """
            transforme toutes les images en tableau multidimensionnel  et la target en vecteur
        """
        #validation des entrées
        assert isinstance(labelsAndPaths , pd.DataFrame) , "L'argument 1 doit être de type 'DataFrame' de pandas !"
        assert (
                'image_path' in labelsAndPaths.columns and 'label' in labelsAndPaths.columns 
                ), "Le Dataframe 'argument 1' doit contenir les chemins vers les fichiers images et le étiquettes de chacune d'elles !"
        assert (
                  isinstance(sizeOfImage , tuple) and np.all(isinstance(number , int)  and  number > 0 for number in sizeOfImage) 
                ) , "La taille de l'image doit être un tuple de deux entiers strictement positifs !"
        assert isinstance(mode , str) and mode in ['RGB' , 'Gray'] , "l'argument 3 'mode' doit être 'RGB' ou 'Gray' !"
        
        #prétaitement des images
        X = np.array([self.loadAndProcessImage(path , sizeOfImage , mode) for path in tqdm(labelsAndPaths['image_path'])])
        y = self.labelsEncoder(labelsAndPaths , 'label' )['label'].values

        # vérification des  dimensions
        assert X.shape[0] == y.shape[0] , "Dimension incohérent entre les données X et target y !"
        if mode == 'Gray':
            d = (len(labelsAndPaths['image_path']),) + sizeOfImage  
            assert (    
                X.shape == d
            ), f"le shape de X  {X.shape} doit correspondre à {d} !"
        else:
            d = (len(labelsAndPaths['image_path']),) + sizeOfImage + (3,)
            assert (
                      X.shape == d
            ), f"le shape de X  {X.shape} doit correspondre à {d} !"
            

        return X , y


    @classmethod
    def splitData(self , X: np.ndarray , y: np.ndarray , test_size: float , stratification: bool= True , 
                  sameResultOfSplit: bool= True)  -> tuple[np.ndarray , ...]:
        #divise les données et le traget en sous ensembles de train, de validation et de test
        #validation des entrées
        assert isinstance(X , np.ndarray) , "l'argument1 doit être de type ndarray de numpy !"
        assert isinstance(y , np.ndarray) , "l'argument 2 doit être de type ndarray de numpy !"
        assert X.shape[0] == y.shape[0] , "des données et le target doivent avoir le même nombre d'enregistrements !"
        assert isinstance(test_size , float) and  1 > test_size > 0 , "la taille de test doit être un réel entre 0 et 1 !"
        assert isinstance(stratification , bool) , "L'argument 3 'stratification' doit prendre soit 'True' soit 'False' comme valeur!"
        assert isinstance(sameResultOfSplit , bool) , "L'argument 4 'sameResultOfSplit' doit prendre soit 'True' soit 'False' comme valeur!"
        

        #division en train et test selon les paramètres 'booleen' passés à la méthode
        if stratification:
            if sameResultOfSplit:
                X_train , X_test , y_train , y_test = train_test_split(X , y , test_size= test_size  , stratify= y , random_state= 42)
            else:
                X_train , X_test , y_train , y_test = train_test_split(X , y , test_size= test_size , stratify= y)
        else:
            if sameResultOfSplit:
                X_train , X_test , y_train , y_test = train_test_split(X , y , test_size= test_size , random_state= 42)
            else:
                X_train , X_test , y_train , y_test = train_test_split(X , y , test_size= test_size)

        #validation des résultat avant de les retourner
        assert X_train.shape[0] == y_train.shape[0] , "X_train et y_train doivent avoir le même nombre d'échantillons!"
        assert X_train.shape[0] == y_train.shape[0] , "X_test et y_test doivent avoir le même nombre d'échantillons!"
        assert X_train.shape[1] == X_test.shape[1] , "X_train et X_test doivent avoir le même nombre de variables!"

        return X_train , X_test , y_train , y_test

    @classmethod
    def oneHotEncoder(self , y: np.ndarray) -> np.ndarray:
        #encode les labels en one-hot: sous forme de veteur contenant des zéros et un
        #validation de l'entrée
        assert isinstance(y , np.ndarray) and y.ndim == 1 , "L'argument doit être de type ndarray d'une seule dimension !"

        y_one_hot = np.array(OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1)))

        #validation de la sortie
        assert (
            isinstance(y_one_hot , np.ndarray) and y_one_hot.shape == (y.shape[0] , len(np.unique(y)))
        ) , "Le résultat de oneHotEncoder n'est un ndarray ou ses dimensions sont incorrectes !"

        return y_one_hot

    @classmethod
    def fit_transform(self , path: str , file_namesAndLabelsOfImages: str= None , sizeOfImage: tuple[int , int]= (32 , 32) , mode: str= 'RGB',
    val_size: float= 0.2 , test_size: float= 0.2 ,  stratification: bool= True , sameResultOfSplit: bool= True) -> tuple[np.ndarray, ...]:
        """        
            cette méthode est un pipeline qui fait appel à les autres méthode pour charger, prétraiter et diviser le data et retourne en fin 
            retourner les ensembles de train, de validation et de test.
        """
        #validation des l'entrées
        assert isinstance(path , str) , "L'argument 1 passée à cette méthode doit être un chemin vers un dossier 'string' !"
        assert isinstance(file_namesAndLabelsOfImages , (type(None) , str)) , "L'arguement 2 doit être un nom d'un fichier csv 'String' ou None !"
        assert (
            isinstance(sizeOfImage , tuple) and np.all(isinstance(number , int)  and  number > 0 for number in sizeOfImage) 
        ) , "La taille de l'image doit être un tuple de deux entiers strictement positifs !"
        assert isinstance(mode , str) and mode in ['RGB' , 'Gray'] , "l'argument 4 'mode' doit être 'RGB' ou 'Gray' !"
        assert isinstance(test_size , float) and  1 > test_size > 0 , "la taille de test doit être un réel entre 0 et 1 !"
        assert isinstance(val_size , float) and  1 > val_size > 0 , "la taille de validation doit être un réel entre 0 et 1 !"
        assert isinstance(stratification , bool) , "L'argument 7 'stratification' doit prendre soit 'True' soit 'False' comme valeur!"
        assert isinstance(sameResultOfSplit , bool) , "L'argument 8 'sameResultOfSplit' doit prendre soit 'True' soit 'False' comme valeur!"

        #chargement du data selon le deux cas: à partir d'un fichier csv ou à partir d'un répertoire
        if file_namesAndLabelsOfImages == None:
            data_dataFrame = self.loadDataFromDirectory(path)
        else:
            data_dataFrame = self.loadData_csvFile(path , file_namesAndLabelsOfImages)

        print(data_dataFrame.columns)
        #encodage des labels
        data_dataFrame = self.labelsEncoder(data_dataFrame , 'label')
        #transformation des images en gris ou en RGB, les redimensionner et les transformer en matrices et la normalisation
        X , y = self.processAllImages(data_dataFrame , sizeOfImage , mode)
        #division des données en train, validation et test
        X_temp , X_test , y_temp , y_test = self.splitData(X , y , test_size , True , True)
        X_train , X_val , y_train , y_val = self.splitData(X_temp , y_temp , val_size*1.25 , True , True)

        #Encodage One-Hot des targets
        y_train = self.oneHotEncoder(y_train)
        y_val = self.oneHotEncoder(y_val)
        y_test = self.oneHotEncoder(y_test)

        #validation des sorties
        assert X_train.shape[0] == y_train.shape[0] , "les premières dimensions de X_train et y_train doit être égales !"
        assert X_val.shape[0] == y_val.shape[0] , "les premières dimensions de X_val et y_val doit être égales !"
        assert X_test.shape[0] == y_test.shape[0] , "les premières dimensions de X_test et y_test doivent être égales !"
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1] , "X_train, X_val, X_test doivent avoir le même nombre de features !"
        assert y_train.shape[1] == y_val.shape[1] == y_test.shape[1] , "y_train, y_val, y_test doivent avoir la même deuxième dimension !"

        return X_train , X_val , X_test , y_train , y_val , y_test



######################################################### classe PreprocessingData  ######################################################################

class PreprocessingData:
    """
     La classe 'PreprocessingData' pour le prétraitement des données et ces arguments d'instanciation sont:
        # numericalOutliersStrategy: stratégie de traîtement des outliers ['keep': les garder , 'mean': les remplacer par Moyenne, 'median': par la médiane ]
        # normalisationStrategy: la technique pour la mise à la même échelle des variables numériques ['normalisation': [0,1] ,'standardisation':
        [m=0 et sigma = 1 ] ]
        # missingNumericalValuesStrategy: méthode pour traîter les vals. num. manqantes ['drop': supprimer les lignes contenant des vals. manqs. , 
        'mean': remp. par  moy , 'median': remp. par mediane ]
        # missingCategoricalValuesStrategy: stratégie pour traîter les vals. manq. catég. ['drop': supprimer les ligne avec des vals. manq. , 'most': 
        remp. par mode , 'less': remp. par la modalité la moins fréquente.
        # ordinalcategoricalVariables: liste des noms des variables (entêtes d'un fichier csv par ex.) catégo. ordinales
        # numericalsWithInvalidZeros: liste des noms des var. num. dont la valeur 0 est jugée invalide ou insignifiante (ex. rythme cardiaque==0 ==> 
        mort pas logique!)
        # de plus, la fonction divise le dataset en train et test avec une méthode dont dispose appelée splitData(...) avec des arguments test_size, 
        stratification (True par défaut / False) et sameResultOfSplit (True par défaut / False)
        
    """
    
    def __init__(self , numericalOutliersStrategy: str= 'keep' , normalisationStrategy: str='standardisation' , 
                 missingNumericalValuesStrategy: str= 'mean' , missingCategoricalValuesStrategy: str= 'most' , 
                 ordinalcategoricalVariables: list[str]= None , numericalsWithInvalidZeros: list[str]= None):
        #verification des entrées
        assert numericalOutliersStrategy in ['mean' , 'median' , 'keep'] , "la technique de traîter les outliers doit être 'mean' ou 'median' ou 'keep'"
        assert normalisationStrategy in ['standardisation' , 'normalisation'] , (
            "la technique de normalisation doit être 'standardisation'ou 'normalisation'"
        )
        
        assert missingNumericalValuesStrategy in ['drop' , 'mean' , 'median'] , (
            """les valeurs numériques manquantes doivent être traîtées comme suit: suppression de ligne: 'drop', remplacement par la moyenne:
            'mean'ou par la médiane 'median'"""
            )
       
        assert missingCategoricalValuesStrategy in ['drop' , 'most' , 'less'] , (
            """les valeurs catégorielles manquantes doivent être traîtées comme suit: suppression de ligne: 'drop', remplacement par le mode:
            'most'ou par la modalité la moins fréquentes 'less'""")
        
        assert ordinalcategoricalVariables is None or (isinstance(ordinalcategoricalVariables, list) and
                                                       all(isinstance(var, str) for var in ordinalcategoricalVariables)), (
        "L'argument 'ordinalcategoricalVariables' doit être une liste de chaînes (noms de variables ordinales) ou None."
        )

        assert numericalsWithInvalidZeros is None or (isinstance(numericalsWithInvalidZeros, list) and
                                                       all(isinstance(var, str) for var in numericalsWithInvalidZeros)), (
        "L'argument 'numericalsWithInvalidZeros' doit être une liste de: (noms de variables ordinales dont la valeur 0 est insignifiante) ou (None)!"
        )

        #initialisation des attributs
        self.numericalOutliersStrategy = numericalOutliersStrategy
        self.normalisationStrategy = normalisationStrategy
        self.missingNumericalValuesStrategy = missingNumericalValuesStrategy
        self.missingCategoricalValuesStrategy = missingCategoricalValuesStrategy
        self.data = None
        self.preProcessedData = None
        self.categoricalVariables = None
        self.numericalVariables = None
        self.ordinalcategoricalVariables = ordinalcategoricalVariables
        self.numericalsWithInvalidZeros = numericalsWithInvalidZeros

    def separateTypeOfVaraibles(self):
        #séparation des deux type de variables contenues dans le dataset pour les préparer aux traîtement
        self.numericalVariables = self.data.select_dtypes(include = ['number'])
        self.categoricalVariables = self.data.select_dtypes(exclude = ['number'])
        
    
    def missingNumericalValuesProcess(self):
        ## traîtement des valeurs numériques manquantes selon la stratégie choisie (moyenne , médiane , suppression) passée 
        ## lors de l'inctanciation de l'objet PreprocessingData
        #vérification de l'attribut 'numericalVariables'
        assert isinstance(self.numericalVariables , pd.DataFrame) , "L'attribut 'numericalVariables' doit être un DataFrame pandas !"
        assert self.numericalVariables is not None and not self.numericalVariables.empty, "'numericalVariables' ne doit pas être None ni vide !"

        #traitement des valeurs manquantes selon la stratégie choisie
        if self.missingNumericalValuesStrategy == 'mean':
            self.numericalVariables.fillna(self.numericalVariables.mean() , inplace= True)
            
        elif self.missingNumericalValuesStrategy == 'median':
            self.numericalVariables.fillna(self.numericalVariables.median() , inplace= True)
        else:
            self.numericalVariables.dropna(inplace= True)

   
    def missingCategoricalValuesProcess(self):
        # # traîtement des valeurs catégorielles manquantes selon la stratégie choisie (mode , modalité moins fréquente , suppression)
        #vérification de l'attribut 'categoricalVariables'
        assert isinstance(self.categoricalVariables , pd.DataFrame) , "L'attribut 'categoricalVariables' doit être un DataFrame pandas !"
        assert self.categoricalVariables is not None and not self.categoricalVariables.empty, "'categoricalVariables' ne doit pas être None ni vide !"

        #traîtement des valeurs manquantes selon la stratégie choisie
        if self.missingCategoricalValuesStrategy == 'most':
            for var in tqdm(self.categoricalVariables.columns , desc= "Traîtement des valeurs catégorielles manquantes:"):
                mostModality = self.categoricalVariables[var].mode()[0]
                self.categoricalVariables.fillna(mostModality , inplace= True)
            
        elif self.missingCategoricalValuesStrategy == 'less':
            for var in tqdm(self.categoricalVariables.columns , desc= "Traîtement des valeurs catégorielles manquantes:"):
                lessModality = self.categoricalVariables[var].value_counts().idxmin()
                self.categoricalVariables.fillna(lessModality , inplace= True)
        else:
            self.categoricalVariables.dropna(inplace= True)

    def normalizeOrStandardizeNumericalVariables(self , target: str= None):
        #la normalisation ou la standardisation des variables numériques selon le choix fait lors de l'instanciation de l'objet de la présente classe
        #argument 'target' sert à l'excepter de la normalisation ou la standardisation s'il figure parmi les variables numériques
        #vérifications
        assert isinstance(self.numericalVariables , pd.DataFrame) , "L'attribut 'numericalVariables' doit être un DataFrame pandas !"
        assert self.numericalVariables is not None and not self.numericalVariables.empty, "'numericalVariables' ne doit pas être None ni vide !"
        assert not self.numericalVariables.isnull().values.any() , ( 
        "l'attribut 'numericalVariables' contient des valeurs manquantes! Il faut les traîter d'abord."
        )
        assert isinstance(target , str) and target in self.data.columns , "l'argument 2 'target' doit être un nom (string) d'une variable de dataset!"

        #affichage de la barre de progression de l'opération de normalisation ou de standardisation evec la bibliothèque python 'tqdm'
        with tqdm(total=1 , desc=f"{self.normalisationStrategy.capitalize()} des variables numériques" , unit= "étape") as pbar:
        
            #instanciation de normaliseur et de standardiseur
            standardScaler = StandardScaler()
            minMaxScaler = MinMaxScaler()
    
            #isoler le target pour l'excepter de la normal. /stand. s'il est renseigné lors de la pelle à la fméthode fit_transform
            if target in self.numericalVariables.columns:
                target_var = self.numericalVariables[[target]].copy()
                self.numericalVariables.drop(target , axis= 1 , inplace= True)
                
            #la standardisation ou la normalisation des variables
            if self.normalisationStrategy == 'standardisation':
                self.numericalVariables = pd.DataFrame(standardScaler.fit_transform(self.numericalVariables) , columns= self.numericalVariables.columns)
            else:
                self.numericalVariables = pd.DataFrame(minMaxScaler.fit_transform(self.numericalVariables) , columns= self.numericalVariables.columns)
    
            #rassembler les variables numériques avec le target si il était isolé ci-dessus 
            if target != None:
                self.numericalVariables = pd.concat([self.numericalVariables , target_var] , axis= 1)

            pbar.update(1)
            
    def encodeCategoricalVariables(self):
        #encoder les variables catégorielles selon le type ordinal ou non ordinal
        #vérifications
        assert isinstance(self.categoricalVariables , pd.DataFrame) , "L'attribut 'categoricalVariables' doit être un DataFrame pandas !"
        assert self.categoricalVariables is not None and not self.categoricalVariables.empty, "'numericalVariables' ne doit pas être None ni vide !"
        assert not self.categoricalVariables.isnull().values.any() , (
        "l'attribut 'categoricalVariables' contient des valeurs manquantes! Il faut les traîter d'abord!"
        )
        #instanciation des encodeurs ordinal et non ordinal
        encoderNotOrdinal = OneHotEncoder( sparse_output = False)
        encoderOrdinal = OrdinalEncoder()
        #Encoder les variables catégorielles
        for var in tqdm(self.categoricalVariables.columns , desc= "Encodage des variables catégorielles:"):
            if self.ordinalcategoricalVariables is not None and var in self.ordinalcategoricalVariables:
                self.categoricalVariables[var] = encoderOrdinal.fit_transform(self.categoricalVariables[[var]])
            else:
                self.categoricalVariables[var] = encoderNotOrdinal.fit_transform(self.categoricalVariables[[var]])

    def manageOutliersInNumericalVariables(self):
        #traiter les valeurs abérrantes par la stratégie choisie('mean': remplacemnt par la moyenne , 'median' : par la médiane
        #, 'keep': garder les) pendant l'instanciation de l'objet PreprocessingData
        #vérifications
        assert isinstance(self.numericalVariables , pd.DataFrame) , "L'attribut 'numericalVariables' doit être un DataFrame pandas !"
        assert self.numericalVariables is not None and not self.numericalVariables.empty, "'numericalVariables' ne doit pas être None ni vide !"
        assert not self.numericalVariables.isnull().values.any() , (
        "l'attribut 'numericalVariables' contient des valeurs manquantes! Il faut les traîter d'abord."
        )
        for var in tqdm(self.numericalVariables.columns , desc= "Traîtement des valeurs numériques aberrantes:"):
            if self.numericalOutliersStrategy == 'keep':
                break
            else: 
                #détection des outliers par la méthode IQR
                Q1 = self.numericalVariables[var].quantile(0.25) # 1er quartile 
                Q3 = self.numericalVariables[var].quantile(0.75) # 3e quartile
                IQR = Q3 - Q1
                lowerBound = Q1 - 1.5 * IQR 
                upperBound = Q3 + 1.5 * IQR
                if self.numericalOutliersStrategy =='mean':
                    self.numericalVariables[var] = np.where((self.numericalVariables[var] < upperBound) | 
                            (self.numericalVariables[var] > bord_superieur), self.numericalVariables[var].mean(), self.numericalVariables[var])
                else:
                    self.numericalVariables[var] = np.where((self.numericalVariables[var] < upperBound) | 
                            (self.numericalVariables[var] > bord_superieur), self.numericalVariables[var].median(), self.numericalVariables[var])
            
    def replaceInvalidZerosWithMedian(self):
        #remplace les zéros invalide c'est-à-dire insignifiants par la médiane
        #vérification de l'attribut 
        assert isinstance(self.numericalVariables , pd.DataFrame) , "L'attribut 'numericalVariables' doit être un DataFrame pandas !"
        assert (self.numericalVariables is not None) and (not self.numericalVariables.empty), "'numericalVariables' ne doit pas être None ni vide !"
        assert not self.numericalVariables.isnull().values.any() , (
        "l'attribut 'numericalVariables' contient des valeurs manquantes! Il faut les traîter d'abord."
        )
        #remplacement des zéros invalides par la médiane
        if self.numericalsWithInvalidZeros != None and len(self.numericalsWithInvalidZeros) > 0:
            for var in tqdm(self.numericalsWithInvalidZeros , desc= "Remplacment des zéros invalides:"):
                median = self.numericalVariables.loc[self.numericalVariables[var] != 0 ,var].median()
                self.numericalVariables[var] = self.numericalVariables[var].replace(0, median)


    def fit_transform(self , X: pd.DataFrame , target: str= None) -> pd.DataFrame:
        #le prétraiment de dataset d'entrée selon les paramètres transmis au constructeur de la classe 'PreprocessingData'
        #le résultat sera affecté à l'attribut preprocessedData renvoyé après sous forme d'un DataFrame
        
        #vérivication de la première'entrée
        assert isinstance(X , pd.DataFrame) , "l'argument passé à cette méthode est de type DataFrame de Pandas"
        assert isinstance(target , str) and target in X.columns , "l'argument 2 'target' doit être un nom (string) d'une variable de dataset!"
        self.data = X.copy()
        #prétraitement du dataset
        self.separateTypeOfVaraibles()            #séparation des variables
    
        if self.numericalVariables is not None and not self.numericalVariables.empty:
            self.missingNumericalValuesProcess()      #traîtement des valeurs numériques manquantes
            print('Traîtement des valeurs numériques manquantes -->','Stratégie:' ,self.missingNumericalValuesStrategy)
            self.replaceInvalidZerosWithMedian()  #remplacement des zéros jugés invalides dans les variables numériques (ex: 0 en âge est invalide)
            print('Remplacement des 0 invalides -->','dans les variables:', self.numericalsWithInvalidZeros , 'Statégie: median')
            self.manageOutliersInNumericalVariables()      #traîtement des valeurs numériques abérrantes
            print('Traîtement des valeurs numériques aberrantes -->', 'Stratégie', self.numericalOutliersStrategy)
            self.normalizeOrStandardizeNumericalVariables(target)    #normalisation ou standardisation des variables numériques
            print('Les données sont mises à la même échelle -->', 'Stratégie:' , self.normalisationStrategy)

        if self.categoricalVariables is not None and not self.categoricalVariables.empty:
            self.missingCategoricalValuesProcess()    #traîtement des valeurs catégorielles manquantes
            print('Traîtement des valeurs catégorielles manquantes -->','Stratégie:' ,self.missingCategoricalValuesStrategy)
            self.encodeCategoricalVariables()                # encodage des variables catégorielles
            print('Encodage des variables catégorielles non ordinales-->','Stratégie: OneHoteEncoder')
            print('Encodage des variables catégorielles  ordinales-->','variables:',self.ordinalcategoricalVariables , 'Stratégie: OrdinalEncoder')
        


        #affectation du résultat à l'attribut preProcessedData
        print("🎉 Le dataset est prétraîté selon les choix ci-dessus que vous avez faits lors de la création de l'objet de cette classe 🎉")
        
        self.preProcessedData = pd.concat( [self.numericalVariables , self.categoricalVariables] , axis = 1 )

        #vérification de l'attribut preProcessedData avant de le renvoyer
        assert (self.numericalVariables is not None) and (not self.numericalVariables.empty) , (
                                                "l'opération de préprocessing n'est pas aboutie, l'attribut 'preProcessedData' est vide!"
        )
        assert isinstance(self.preProcessedData , pd.DataFrame) , "la valeur de retour de la fonction 'fit_transform' est de type DataFrame!"

        return self.preProcessedData

    def splitData(self , target: str= None , test_size: float= 0.2 , stratification: bool= True ,
                                                               sameResultOfSplit: bool= True) -> tuple[np.ndarray , ...]:
        #séparation de la target des features et la division des deux en deux ensembles d'entraînement et de test:
        #X_train , y_train , X_test et y_test qui seront renvoyés en fin de la fonction
        #validation des entrées
        assert self.preProcessedData is not None , "Le data n'est pas encore traîter, preProcessedData est None!"
        assert isinstance(target , str) and target in self.preProcessedData.columns , "L'argument 1 'target' doit être une colonne de dataframe 'arg1'!"
        assert isinstance(test_size , float) and  0 < test_size < 1 , "La taille de test 'test_size' doit être un réel entre 0 et 1 !"
        assert isinstance(stratification , bool) , "L'argument 3 'stratification' doit prendre soit 'True' soit 'False' comme valeur!"
        assert isinstance(sameResultOfSplit , bool) , "L'argument 4 'sameResultOfSplit' doit prendre soit 'True' soit 'False' comme valeur!"

        #séparation du la target des autres features
        features = (self.preProcessedData.drop([target] , axis= 1)).to_numpy()
        target = (self.preProcessedData[[target]]).to_numpy()

        #division en train et test selon les paramètres 'booleen' passés à la méthode
        if stratification:
            if sameResultOfSplit:
                X_train , X_test , y_train , y_test = train_test_split(features , target , test_size= test_size  , stratify= target , random_state= 42)
            else:
                X_train , X_test , y_train , y_test = train_test_split(features , target , test_size= test_size , stratify= target)
        else:
            if sameResultOfSplit:
                X_train , X_test , y_train , y_test = train_test_split(features , target , test_size= test_size , random_state= 42)
            else:
                X_train , X_test , y_train , y_test = train_test_split(features , target , test_size= test_size)

        #validation des résultat avant de les retourner

        assert X_train.shape[0] == y_train.shape[0] , "X_train et y_train doivent avoir le même nombre d'échantillons!"
        assert X_train.shape[0] == y_train.shape[0] , "X_test et y_test doivent avoir le même nombre d'échantillons!"
        assert X_train.shape[1] == X_test.shape[1] , "X_train et X_test doivent avoir le même nombre de variables!"

        return X_train , X_test , y_train , y_test
        
        