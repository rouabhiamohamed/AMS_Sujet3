import stanza
from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess
import networkx as nx
import pandas as pd
from collections import defaultdict
import re
from flair.data import Sentence
from flair.models import SequenceTagger
from rapidfuzz import fuzz
from rapidfuzz import process
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
                
                
            embeddings = model.encode(listePersonnagesTrier)
            
            similarities = cosine_similarity(embeddings)

            # Afficher la matrice de similarité
            # print(similarities)
            
            seuil = 0.7

            # Trouver les paires d'alias avec une similarité supérieure au seuil
            alias_pairs = []
            for i in range(len(listePersonnagesTrier)):
                for j in range(i + 1, len(listePersonnagesTrier)):
                    if similarities[i][j] > seuil:
                        alias_pairs.append((listePersonnagesTrier[i], listePersonnagesTrier[j]))
            
            print("/////////////////",book_code,chapter,"////////////////////")
            # Afficher les alias trouvés
            # for pair in alias_pairs:
                # print(f"Alias trouvé : {pair[0]} <-> {pair[1]}")
            
            alias_clusters = []
            
            db = DBSCAN(eps=0.4, min_samples=1, metric="cosine").fit(embeddings)
            # Afficher les clusters trouvés
            for cluster in set(db.labels_):
                if cluster != -1:  # Exclure les points "bruit" (cluster -1 dans DBSCAN)
                    cluster_aliases = [
                        listePersonnagesTrier[i] for i, label in enumerate(db.labels_) if label == cluster
                    ]
                    alias_clusters.append(cluster_aliases)

            # Afficher la liste de listes d'alias
            for i, cluster in enumerate(alias_clusters):
                print(f"Cluster {i}: {cluster}")
            
ListeAllperso = list(dict.fromkeys(Listeperso))
embeddings = model.encode(ListeAllperso)
            
similarities = cosine_similarity(embeddings)

# Afficher la matrice de similarité
# print(similarities)

seuil = 0.8

# Trouver les paires d'alias avec une similarité supérieure au seuil
alias_pairs = []
for i in range(len(ListeAllperso)):
    for j in range(i + 1, len(ListeAllperso)):
        if similarities[i][j] > seuil:
            alias_pairs.append((ListeAllperso[i], ListeAllperso[j]))

print("/////////////////",book_code,chapter,"////////////////////")
# Afficher les alias trouvés
# for pair in alias_pairs:
    # print(f"Alias trouvé : {pair[0]} <-> {pair[1]}")

alias_clusters = []

db = DBSCAN(eps=0.4, min_samples=1, metric="cosine").fit(embeddings)
# Afficher les clusters trouvés
for cluster in set(db.labels_):
    if cluster != -1:  # Exclure les points "bruit" (cluster -1 dans DBSCAN)
        cluster_aliases = [
            ListeAllperso[i] for i, label in enumerate(db.labels_) if label == cluster
        ]
        alias_clusters.append(cluster_aliases)

# Afficher la liste de listes d'alias
for i, cluster in enumerate(alias_clusters):
    print(f"Cluster {i}: {cluster}")            
                      
            
            
            
            
            
            
# listeNomsPersonnages = []

# for nom in listePersonnagesTrier:
    # variantesNoms = [nom]  # Utiliser une liste pour stocker les variantes
    # for autreNom in listePersonnagesTrier:
        # if autreNom != nom and (autreNom in nom or nom in autreNom) or (nom.upper() == autreNom or nom == autreNom.upper()):  # Vérifier si c'est une variante
            # if autreNom not in variantesNoms:  # Ajouter uniquement si ce n'est pas déjà présent
                # variantesNoms.append(autreNom)
    
    # Vérifier si une liste équivalente n'est pas déjà dans listeNomsPersonnages
    # if not any(sorted(variantesNoms) == sorted(existing) for existing in listeNomsPersonnages):
        # listeNomsPersonnages.append(variantesNoms)

# Supprimer les listes qui sont des sous-listes d'autres
# for variantesNoms in listeNomsPersonnages[:]:  # Copier pour éviter les conflits pendant la suppression
    # for s in listeNomsPersonnages[:]: 
        # if variantesNoms != s and all(nom in s for nom in variantesNoms):  # Si variantesNoms est un sous-ensemble de s
            # if variantesNoms in listeNomsPersonnages:  # Vérifier que variantesNoms n'a pas déjà été supprimé
                # listeNomsPersonnages.remove(variantesNoms)
        # elif variantesNoms != s and all(nom in variantesNoms for nom in s):  # Si s est un sous-ensemble de variantesNoms
            # if s in listeNomsPersonnages:  # Vérifier que s n'a pas déjà été supprimé
                # listeNomsPersonnages.remove(s)



# print(chapter, book_code,"////////////////////////////////////////////////////////")

# Afficher les groupes formés
# for variantes in listeNomsPersonnages:
    # print(f"Personnage principal: {variantes[0]}")
    # print(f"Alias: {variantes}")
    # print("---")