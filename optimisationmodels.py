# module optimisationmodels

"""
ce module contient des implémentation sous forme de classe des algorithmes d'optimisation
"""

#importations
import numpy as np


########################################################### Algorithme Adam  ##############################################################################

class AdamOptimizer:
    """
        L'optimiseur Adam applicable à tout le réseau ou à une seule couche du réseau: adapté à MLP et aux réseaux de convolution CNN à la fois et selon
        la méthode appliquée ('updateForAllLayers(...) --> réseau tout entier <> updateForOneLayer(...) --> une seule couche'.
    """
    
    def __init__(self, learning_rate: float= 0.01, beta1: float= 0.9 , beta2: float= 0.999 , epsilon: float= 1e-8):
        """
            le constructeur des objets de type AdamOptimizer en initialisant les paramètres beta1 et beta2, le learning rate et epsilon qui sert à
            éviter la division par zéro
        """
        #validation des entrées
        assert isinstance(learning_rate , float) , "Le pas d'apprentissage 'Arg1' doit être un réel !"
        assert isinstance(beta1 , float) , "Le coef bata1 'Arg2' doit être un réel !"
        assert isinstance(beta2 , float) , "Le coef bata2 'Arg3' doit être un réel !"
        assert isinstance(epsilon , float) , "La constante epsilon 'Arg4' doit être un réel très proche de zéro !"

        #création et initialisation des attributs 
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        #attributs pour le cas d'application de l'optimiseur sur le réseau tout entier
        self.m_weights = None # poids du premier moment
        self.v_weights = None # poids du deuxième moment
        self.m_biases  = None  # biais du premier moment
        self.v_biases  = None  # biais du deuxième moment
        self.t_all     = 0        # Compteur de pas en cas d'application sur la totalité du réseau

        #attributs pour le cas d'application de l'optimiseur sur une seule couche
        self.m_w = None  # Premier moment pour les poids
        self.v_w = None  # Second moment pour les poids
        self.m_b = None  # Premier moment pour les biais
        self.v_b = None  # Second moment pour les biais
        self.t_one = 0   # Compteur de pas en cas d'application sur une seule couche

    def updateForAllLayers(self , weights: list[np.ndarray] , biases: list[np.ndarray] , d_weights: list[np.ndarray] , d_biases: list[np.ndarray]) -> \
    tuple[list[np.ndarray] , list[np.ndarray]]:
        """
            Elle mise à jour les poids et les biais transmises comme arguments de toutes les couches du réseau en utilisant les moment (vitesse 
            et variance des gradients)  et elle les retourne à la fin.
        """
        #validation des entrées
        assert isinstance(weights , list) and all(isinstance(element , np.ndarray) for element in weights) , \
                                                                                                   "L'arg1 doit être une liste de tableau numpy !"
        assert isinstance(biases , list) and all(isinstance(element , np.ndarray) for element in biases) , \
                                                                                                   "L'arg2 doit être une liste de tableau numpy !"
        assert isinstance(d_weights , list) and all(isinstance(element , np.ndarray) for element in d_weights) , \
                                                                                                   "L'arg3 doit être une liste de tableau numpy !"
        assert isinstance(d_biases , list) and all(isinstance(element , np.ndarray) for element in d_biases) , \
                                                                                                   "L'arg4 doit être une liste de tableau numpy !"
        assert len(weights) == len(d_weights), "Nombre de couches incohérent pour les poids !"
        assert len(biases) == len(d_biases), "Nombre de couches incohérent pour les biais !"
        
        # Initialisation des moments à zéros pour la première fois
        if self.m_weights is None:
            self.m_weights = [np.zeros_like(w) for w in weights]
            self.v_weights = [np.zeros_like(w) for w in weights]
            self.m_biases  = [np.zeros_like(b) for b in biases]
            self.v_biases  = [np.zeros_like(b) for b in biases]
            
        #incémentation de step
        self.t_all += 1

        #mise à jour des poids et biais
        for i in range(len(weights)):
            # calcul et correction des moments (vitesse m et variance v)
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * d_weights[i]
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (d_weights[i] ** 2)

            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * d_biases[i]
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (d_biases[i] ** 2)

            # Correction des moments m et v
            m_hat_w = self.m_weights[i] / (1 - self.beta1 ** self.t_all)
            v_hat_w = self.v_weights[i] / (1 - self.beta2 ** self.t_all)

            m_hat_b = self.m_biases[i] / (1 - self.beta1 ** self.t_all)
            v_hat_b = self.v_biases[i] / (1 - self.beta2 ** self.t_all)

            # mise à jour des poid et biais
            weights[i] -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            biases[i] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

        #validation des sorties
        assert all(np.isfinite(w).all() for w in weights) and all(np.isfinite(b).all() for b in biases), \
                                                                                "Les poids et les biais ne doivent contenir ni 'NaN' ni 'Inf' !"

        return weights , biases


    def updateForOneLayer(self , weights: np.ndarray , biases: np.ndarray , d_weights: np.ndarray , d_biases: np.ndarray) ->  \
    tuple[np.ndarray , np.ndarray]:
        """
            Elle mise à jour les poids et les biais d'une seule couche transmises comme arguments en utilisant les moment (vitesse et variance
            des gradients) et elle les retourne à la fin.
        """
        #validation des entrées
        assert isinstance(weights , np.ndarray) , "L'arg1 doit être de type 'ndarray' de 'numpy' !"
        assert isinstance(biases , np.ndarray) , "L'arg1 doit être de type 'ndarray' de 'numpy' !"
        assert isinstance(d_weights , np.ndarray) , "L'arg1 doit être de type 'ndarray' de 'numpy' !"
        assert isinstance(d_biases , np.ndarray) , "L'arg1 doit être de type 'ndarray' de 'numpy' !"

        assert weights.shape == d_weights.shape , "Les poids et leur gradients doivent avoir le même shape !"
        assert biases.shape  == d_biases.shape , "Les poids et leur gradients doivent avoir le même shape !"
        
        # Initialisation des moments à zéros pour la première fois
        if self.m_w is None:
            self.m_w = np.zeros_like(weights)
            self.v_w = np.zeros_like(weights)
            self.m_b = np.zeros_like(biases)
            self.v_b = np.zeros_like(biases)

            
        #incrémentation du step
        self.t_one += 1

        #calcul des moments des poids
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * d_weights
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (d_weights ** 2)

        #calcul des moments des biais
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * d_biases
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (d_biases ** 2)

        #correction des moments de la couche
        m_w_hat = self.m_w / (1 - self.beta1 ** self.t_one)
        v_w_hat = self.v_w / (1 - self.beta2 ** self.t_one)

        m_b_hat = self.m_b / (1 - self.beta1 ** self.t_one)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t_one)

        #mise à jour des poids et biais de la couche
        weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        biases  -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
        
        #validation des sorties
        assert np.all(np.isfinite(weights)) and np.all(np.isfinite(biases)) , "Il ne faut pas que les sorties contiennent des valeurs indéfinies !"

        return weights, biases
        

####################################################### L'algorithme SGD ##################################################################################

class SGDOptimiser:
    """
        L'optimiseur SGD adapté à la fois à un réseau LMP et à une couche toute seule pour le cas d'un CNN
    """
    def __init__(self , learning_rate: float= 0.01 , _lambda: float= 0.0):
        """
            Le constructeur de la classe qui retourne un objet de type 'SGDOptimiser' avec un learning rate défini (par défaut : 0.01)
            et le coefficient de régularisation L2 (par défaut : 0 , pas de régularisation)
        """
        #validation de l'entrées
        assert isinstance(learning_rate , float) and learning_rate > 0 , "Le learning rate doit être un réel strictement positif !"
        assert isinstance(_lambda , float) and _lambda >= 0 , "Le coef. de régularisation doit être un réel positif !"
        
        #initialisation du pas d'apprentissage et de coefficient de régularisation L2
        self.learning_rate = learning_rate
        self._lambda       = _lambda
        

    def updateForAllLayers(self, weights: list[np.ndarray] , biases: list[np.ndarray] , d_weights: list[np.ndarray], d_biases: list[np.ndarray]) -> \
    tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Mise à jour des poids et biais de tout un réseau (liste de couches).
        """
        #validation des entrées
        assert isinstance(weights , list) and all(isinstance(element , np.ndarray) for element in weights) , \
                                                                                                   "L'arg1 doit être une liste de tableau numpy !"
        assert isinstance(biases , list) and all(isinstance(element , np.ndarray) for element in biases) , \
                                                                                                   "L'arg2 doit être une liste de tableau numpy !"
        assert isinstance(d_weights , list) and all(isinstance(element , np.ndarray) for element in d_weights) , \
                                                                                                   "L'arg3 doit être une liste de tableau numpy !"
        assert isinstance(d_biases , list) and all(isinstance(element , np.ndarray) for element in d_biases) , \
                                                                                                   "L'arg4 doit être une liste de tableau numpy !"
        assert len(weights) == len(d_weights), "Nombre de couches incohérent pour les poids !"
        assert len(biases) == len(d_biases), "Nombre de couches incohérent pour les biais !"

        #la mise à jour des poids et biais du toutes les couches du réseau
        for i in range(len(weights)):
            weights[i] -= self.learning_rate * (d_weights[i] + self._lambda * weights[i])
            biases[i] -= self.learning_rate * d_biases[i]

        #validation des sorties
        assert all(np.isfinite(w).all() for w in weights) and all(np.isfinite(b).all() for b in biases), \
                                                                                "Les poids et les biais ne doivent contenir ni 'NaN' ni 'Inf' !"

        return weights, biases
        

    def updateForOneLayer(self , weights: np.ndarray , biases: np.ndarray , d_weights: np.ndarray , d_biases: np.ndarray) -> \
    tuple[np.ndarray , np.ndarray]:
        """
            Mise à jour des poids et biais d'une seule couche à laquelle il s'applique le SGD.
        """
        #validation des entrées
        assert isinstance(weights , np.ndarray) , "l'arg1 doit être de type 'ndarray' de 'numpy' !"
        assert isinstance(biases , np.ndarray) , "l'arg2 doit être de type 'ndarray' de 'numpy' !"
        assert isinstance(d_weights , np.ndarray) , "l'arg3 doit être de type 'ndarray' de 'numpy' !"
        assert isinstance(d_biases , np.ndarray) , "l'arg4 doit être de type 'ndarray' de 'numpy' !"
        
        assert weights.shape == d_weights.shape, "dimensions incohérentes entre poids 'arg1' et gradients 'arg3'!"
        assert biases.shape == d_biases.shape, "dimensions incohérentes entre biais 'arg2' et gradients 'arg4' !"

        #mise à jour des poids et biais de la couche
        weights -= self.learning_rate * (d_weights + self._lambda * weights)
        biases -= self.learning_rate * d_biases

        #validation des sorties
        assert np.all(np.isfinite(weights)) , "les poids ne doivent contenir ni 'NaN' ni 'inf' !"
        assert np.all(np.isfinite(biases)) , "les biais ne doivent contenir ni 'NaN' ni 'inf' !"

        return weights, biases


####################################################### L'algorithme Momentum ###########################################################################

class MomentumOptimizer:
    """
    Optimiseur SGD avec Momentum : ajoute un terme d'inertie pour la stabilité du modèle et améliorer sa vitesse de convergence. il est 
    adapté aux MLP (applicable au réseau tout entier) et aux CNN (applicable à une seule couche).
    
    """

    def __init__(self , learning_rate: float = 0.01 , gamma: float = 0.9):
        """
            Le constructeur des des objets de type 'MomentumOptimizer' en initialisant les deux attribut 'learning_rate' et 'gamma'
            par des valeurs passées comme arguments et retourne un objet 'MomentumOptimizer'.
        """
        assert isinstance(learning_rate , float) and learning_rate > 0 , "L'argument 1 'learning_rate' doit être un réel strictement positif !"
        assert isinstance(gamma , float) and 0 <= gamma <= 1 , "Le coef. de momentum 'gamma' doit être un réel entre 0 et 1 !"
        
        self.learning_rate = learning_rate    #pas d'apprentissage
        self.gamma         = gamma            #coefficient de momentum

        #momentums pour une seule couche
        self.v_w = None
        self.v_b = None

        #momentums pour plusieurs couches
        self.v_weights = None
        self.v_biases  = None


    def updateForAllLayers(self , weights: list[np.ndarray], biases: list[np.ndarray] , d_weights: list[np.ndarray] , d_biases: list[np.ndarray]) -> \
    tuple[list[np.ndarray] , list[np.ndarray]]:
        """
            Elle mise à jour les gradients et les biais avec Momentum  d'un réseau tout entier qu'elle retourne à la fin de chaque step.
        """
        #validation des entrées
        assert isinstance(weights , list) and all(isinstance(element , np.ndarray) for element in weights) , \
                                                                                                   "L'arg1 doit être une liste de tableau numpy !"
        assert isinstance(biases , list) and all(isinstance(element , np.ndarray) for element in biases) , \
                                                                                                   "L'arg2 doit être une liste de tableau numpy !"
        assert isinstance(d_weights , list) and all(isinstance(element , np.ndarray) for element in d_weights) , \
                                                                                                   "L'arg3 doit être une liste de tableau numpy !"
        assert isinstance(d_biases , list) and all(isinstance(element , np.ndarray) for element in d_biases) , \
                                                                                                   "L'arg4 doit être une liste de tableau numpy !"
        assert len(weights) == len(d_weights) , "Nombre de couches incohérent pour les poids !"
        assert len(biases) == len(d_biases) , "Nombre de couches incohérent pour les biais !"

        ##Initialisation des momentums des poids et des biais à zéros pour la première fois (pour toutes les couches)
        if self.v_weights is None:
            self.v_weights = [np.zeros_like(w) for w in weights]
            self.v_biases = [np.zeros_like(b) for b in biases]
            
        #calcul des momentums des couches et la mise à jour des poids et biais de chaque couche
        for i in range(len(weights)):
            #calcul des momentum
            self.v_weights[i] = self.gamma * self.v_weights[i] + d_weights[i]
            self.v_biases[i]  = self.gamma * self.v_biases[i]  + d_biases[i]

            #mise à jour
            weights[i] -= self.learning_rate * self.v_weights[i]
            biases[i]  -= self.learning_rate * self.v_biases[i]

        #validation des sorties
        assert all(np.all(np.isfinite(w)) for w in weights) , "Il ne faut pas que les poids contiennent des valeurs indéfinies ('Nan' ou 'inf')!"
        assert all(np.all(np.isfinite(b)) for b in biases) , "Il ne faut pas que les biais contiennent des valeurs indéfinies ('Nan' ou 'inf')!"

        return weights, biases

    

    def updateForOneLayer(self , weights: np.ndarray , biases: np.ndarray , d_weights: np.ndarray , d_biases: np.ndarray) -> \
    tuple[np.ndarray , np.ndarray]:
        """
            Elle mise à jour des gradients et des biais avec Momentum  d'une seule couche qu'elle retourne à la fin de chaque step.
        """
        #validation des entrées
        assert isinstance(weights , np.ndarray) , "l'arg1 doit être de type 'ndarray' de 'numpy' !"
        assert isinstance(biases , np.ndarray) , "l'arg2 doit être de type 'ndarray' de 'numpy' !"
        assert isinstance(d_weights , np.ndarray) , "l'arg3 doit être de type 'ndarray' de 'numpy' !"
        assert isinstance(d_biases , np.ndarray) , "l'arg4 doit être de type 'ndarray' de 'numpy' !"

        assert weights.shape == d_weights.shape , "dimensions incohérentes entre poids 'arg1' et gradients 'arg3'!"
        assert biases.shape == d_biases.shape , "dimensions incohérentes entre biais 'arg2' et gradients 'arg4' !"

        #Initialisation du momentum des poids et des biais à zéros pour la première fois
        if self.v_w is None:
            self.v_w = np.zeros_like(weights)
            self.v_b = np.zeros_like(biases)

        #Calcul des momentums pour es poids et les biais de la couche
        self.v_w = self.gamma * self.v_w + self.learning_rate * d_weights
        self.v_b = self.gamma * self.v_b + self.learning_rate * d_biases

        #mise à gour des poids et biais de la couche
        weights -= self.v_w
        biases  -= self.v_b
        
        #validation des sorties
        assert np.all(np.isfinite(weights)) , "les poids ne doivent contenir ni 'NaN' ni 'inf' !"
        assert np.all(np.isfinite(biases)) , "les biais ne doivent contenir ni 'NaN' ni 'inf' !"

        return weights, biases
        
