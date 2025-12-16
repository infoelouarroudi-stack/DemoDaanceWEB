import numpy as np
import cv2
import os
import pickle
import sys
import math
from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton

class GenNeirest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Neirest neighbor method: it select the image in videoSke that has the skeleton closest to the skeleton
    """
    def __init__(self, videoSkeTgt):
        self.videoSkeletonTarget = videoSkeTgt

    def generate(self, ske):
        """ generator of image from skeleton """
        # Initialisation de la recherche du minimum
        min_dist = float('inf')
        best_idx = -1
        
        # On parcourt tous les squelettes de la base de données cible
        # videoSkeletonTarget contient toutes les paires (Image, Squelette) de la vidéo cible
        for i in range(self.videoSkeletonTarget.skeCount()):
            target_ske = self.videoSkeletonTarget.ske[i]
            
            # La méthode .distance() est déjà implémentée dans la classe Skeleton
            # Elle calcule la différence de position des articulations
            dist = ske.distance(target_ske)
            
            # Si on trouve un squelette plus proche que le précédent meilleur, on le garde
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        
        # Une fois la boucle finie, on récupère l'image correspondant au meilleur squelette trouvé
        if best_idx != -1:
            image = self.videoSkeletonTarget.readImage(best_idx)
            # Conversion en float [0,1] pour la compatibilité avec DanceDemo
            return image.astype(float) / 255.0
        else:
            # Sécurité : si la base est vide (ne devrait pas arriver), on renvoie du blanc
            return np.ones((64, 64, 3), dtype=float)
