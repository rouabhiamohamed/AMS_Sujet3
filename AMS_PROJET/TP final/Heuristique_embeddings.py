import stanza
from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess
import networkx as nx
import pandas as pd
from collections import defaultdict
import re

from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN


###Liste des livres et chapitres à extraire
books = [
    (list(range(0, 19)), "paf"),
    (list(range(0, 18)), "lca"),
]

# Charger le modèle SBERT pour obtenir les embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Un modèle léger pour les comparaisons de similarité

Listeperso = []

for chapters, book_code in books:
    for chapter in chapters:
        if book_code == "paf":
            repertory = "prelude_a_fondation"
        else:
            repertory = "les_cavernes_d_acier"
        if chapter!=0:
            with open(f"Partie2/{repertory}_chapter_{chapter}_characters.txt", "r") as file:
                listePersonnagesTrier = [line.strip() for line in file if line.strip()]

            for perso in listePersonnagesTrier:
                Listeperso.append(perso)


            listeNomsPersonnages = []

            for nom in listePersonnagesTrier:
                variantesNoms = [nom]  # Utiliser une liste pour stocker les variantes
                for autreNom in listePersonnagesTrier:
                    if autreNom != nom and (autreNom in nom or nom in autreNom) or (nom.upper() == autreNom or nom == autreNom.upper()):  # Vérifier si c'est une variante
                        if autreNom not in variantesNoms:  # Ajouter uniquement si ce n'est pas déjà présent
                            variantesNoms.append(autreNom)

                # Vérifier si une liste équivalente n'est pas déjà dans listeNomsPersonnages
                if not any(sorted(variantesNoms) == sorted(existing) for existing in listeNomsPersonnages):
                    listeNomsPersonnages.append(variantesNoms)

            # Supprimer les listes qui sont des sous-listes d'autres
            for variantesNoms in listeNomsPersonnages[:]:  # Copier pour éviter les conflits pendant la suppression
                for s in listeNomsPersonnages[:]:
                    if variantesNoms != s and all(nom in s for nom in variantesNoms):  # Si variantesNoms est un sous-ensemble de s
                        if variantesNoms in listeNomsPersonnages:  # Vérifier que variantesNoms n'a pas déjà été supprimé
                            listeNomsPersonnages.remove(variantesNoms)
                    elif variantesNoms != s and all(nom in variantesNoms for nom in s):  # Si s est un sous-ensemble de variantesNoms
                        if s in listeNomsPersonnages:  # Vérifier que s n'a pas déjà été supprimé
                            listeNomsPersonnages.remove(s)

            for index, variantesNoms in enumerate(listeNomsPersonnages):
                for variantesNoms2 in listeNomsPersonnages[:]:  # Copie de la liste pour éviter les problèmes
                    trouver = False
                    for nom in variantesNoms2:
                        if nom in variantesNoms and not trouver:
                            trouver = True
                            for nomsAAjouter in variantesNoms2:
                                if nomsAAjouter not in variantesNoms:  # Ajouter uniquement des éléments uniques
                                    listeNomsPersonnages[index].append(nomsAAjouter)


            # Supprimer les listes qui sont des sous-listes d'autres
            for variantesNoms in listeNomsPersonnages[:]:  # Copier pour éviter les conflits pendant la suppression
                for s in listeNomsPersonnages[:]:
                    if variantesNoms != s and all(nom in s for nom in variantesNoms):  # Si variantesNoms est un sous-ensemble de s
                        if variantesNoms in listeNomsPersonnages:  # Vérifier que variantesNoms n'a pas déjà été supprimé
                            listeNomsPersonnages.remove(variantesNoms)
                    elif variantesNoms != s and all(nom in variantesNoms for nom in s):  # Si s est un sous-ensemble de variantesNoms
                        if s in listeNomsPersonnages:  # Vérifier que s n'a pas déjà été supprimé
                            listeNomsPersonnages.remove(s)
            

            listeDeNomsComplete = []
            for variantesNoms in listeNomsPersonnages[:]:
                for personnage in variantesNoms:
                    listeDeNoms = []
                    listeDeNomsSimple = []
                    if len(personnage.split()) >= 2 and '.' not in personnage.split()[0]:
                        for personnageAComparer in variantesNoms:
                            if len(personnageAComparer.split()) >= 2 and '.' not in personnageAComparer.split()[0]:
                                if personnage.split()[-1] != personnageAComparer.split()[-1]:
                                    listeDeNoms.append(personnageAComparer)
                                    listeDeNomsComplete.append(personnageAComparer)
                    for personnageARajouter in variantesNoms:
                        if len(personnageARajouter.split()) == 1 and personnageARajouter not in listeDeNoms:
                            # Vérifier si personnageARajouter est un sous-ensemble de nomAComparer
                            for nomAComparer in listeDeNoms:
                                if personnageARajouter in nomAComparer:
                                    listeDeNomsSimple.append(personnageARajouter)
                    listeDeNomsConcatenee = listeDeNoms + listeDeNomsSimple

                    # Enlever les éléments de listeDeNomsConcatenee de la liste originale
                    for elem in listeDeNomsConcatenee:
                        if elem in variantesNoms:
                            variantesNoms.remove(elem)
                            
                    listeNomsPersonnages.append(listeDeNomsConcatenee)

            #Pour nettoyer la liste des listes vides 
            listeNomsPersonnages = [variantesNoms for variantesNoms in listeNomsPersonnages if variantesNoms]

            print("/////////////////",book_code,chapter,"////////////////////")

            for i in listeNomsPersonnages:
                print(i)
