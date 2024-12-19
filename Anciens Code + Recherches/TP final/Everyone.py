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

# Initialisation du pipeline de traitement de texte pour le français
nlp = stanza.Pipeline('fr', processors='tokenize,mwt,pos,ner')
        


###Liste des livres et chapitres à extraire        
books = [
    (list(range(0, 19)), "paf"),
    (list(range(0, 18)), "lca"),
]



listeNomsPersonnagesPaf = []
listeNomsPersonnagesLca = []

for chapters, book_code in books:
    Listeperso = []
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
                
        
    listePersonnagesTrier = list(dict.fromkeys(Listeperso))
    #print(listePersonnagesTrier)
    
    listeNomsPersonnages = []

    for nom in listePersonnagesTrier:
        variantesNoms = [nom]  # Utiliser une liste pour stocker les variantes
        for autreNom in listePersonnagesTrier:
            if autreNom != nom and (autreNom in nom or nom in autreNom) or (nom.upper() == autreNom or nom == autreNom.upper()):  # Vérifier si c'est une variante
                docNom1 = nlp(nom)
                tokenNom1 = [word.upos for sent in docNom1.sentences for word in sent.words]
                docNom2 = nlp(autreNom)
                tokenNom2 = [word.upos for sent in docNom2.sentences for word in sent.words]
                if not (len(nom.split()) >= 2 and len(autreNom.split()) >= 2 and ("NOUN" in tokenNom1 and "NOUN" in tokenNom2)):
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

    print("//////////////////Famille ensemble/////////////////////")
    for variantes in listeNomsPersonnages:
        print(variantes)


                    
    # Pour séparer les alias qui n'ont pas les même noms de famille                 
    # for variantesNoms in listeNomsPersonnages[:]:
        # for personnage in variantesNoms:
            # listeDeNoms = []
            # listeDeNomsSimple = []
            # if len(personnage.split()) >= 2 and '.' not in personnage.split()[0]:
                # for personnageAComparer in variantesNoms:
                    # if len(personnageAComparer.split()) >= 2 and '.' not in personnageAComparer.split()[0]:
                        # if personnage.split()[-1] != personnageAComparer.split()[-1]:
                            # listeDeNoms.append(personnageAComparer)
            # for personnageARajouter in variantesNoms:
                # if len(personnageARajouter.split()) == 1 and personnageARajouter not in listeDeNoms:
                    ##Vérifier si personnageARajouter est un sous-ensemble de nomAComparer
                    # for nomAComparer in listeDeNoms:
                        # if personnageARajouter in nomAComparer:
                            # listeDeNomsSimple.append(personnageARajouter)
            # listeDeNomsConcatenee = listeDeNoms + listeDeNomsSimple

            ##Enlever les éléments de listeDeNomsConcatenee de la liste originale
            # for elem in listeDeNomsConcatenee:
                # if elem in variantesNoms:
                    # variantesNoms.remove(elem)
                    
            # listeNomsPersonnages.append(list(dict.fromkeys(listeDeNomsConcatenee)))

    ##Pour nettoyer la liste des listes vides 
    # listeNomsPersonnages = [variantesNoms for variantesNoms in listeNomsPersonnages if variantesNoms]

    ##Pour séparer les alias avec les mêmes noms de famille, mais pas les mêmes prénoms
    # for variantesNoms in listeNomsPersonnages[:]:
        # for personnage in variantesNoms:
            # listeDeNoms = []
            # listeDeNomsSimple = []
            # nomDeFamille = ""
            # if len(personnage.split()) >= 2 and len(personnage.split()[0])>2:
                # for personnageAComparer in variantesNoms:
                    # if len(personnageAComparer.split()) >= 2 and len(personnageAComparer.split()[0])>2 and personnage!=personnageAComparer:
                        # docPrenom1 = nlp(personnage)
                        # tokenPrenom1 = [(word.upos, word.feats.split('=')[1].split('|')[0] if word.feats and 'Gender' in word.feats else None) for sent in docPrenom1.sentences for word in sent.words]
                        # docPrenom2 = nlp(personnageAComparer)
                        # tokenPrenom2 = [(word.upos, word.feats.split('=')[1].split('|')[0] if word.feats and 'Gender' in word.feats else None) for sent in docPrenom2.sentences for word in sent.words]
                        # first_token1 = tokenPrenom1[0]
                        # first_token2 = tokenPrenom2[0]
                        # if personnage.split()[0] != personnageAComparer.split()[0] and \
                            # (("NOUN" in first_token1 and "NOUN" in first_token2) and
                            # "Masc" in first_token1 and "Fem" in first_token2) or \
                            # (("PROPN" in first_token1 and "PROPN" in first_token2)):
                            # listeDeNoms.append(personnageAComparer)
                            # nomDeFamille = personnageAComparer.split()[-1]
                            
            # for personnageARajouter in variantesNoms:
                # if len(personnageARajouter.split()) == 1 and personnageARajouter not in listeDeNoms:
                    ##Vérifier si personnageARajouter est un sous-ensemble de nomAComparer
                    # for nomAComparer in listeDeNoms:
                        # if personnageARajouter in nomAComparer:
                            # listeDeNomsSimple.append(personnageARajouter)
            # listeDeNomsConcatenee = listeDeNoms + listeDeNomsSimple
            ##Enlever les éléments de listeDeNomsConcatenee de la liste originale
            # for elem in listeDeNomsConcatenee:
                # if elem in variantesNoms:
                    # variantesNoms.remove(elem)
                    
            # listeNomsPersonnages.append(listeDeNomsConcatenee)

    ##Pour nettoyer la liste des listes vides 
    # listeNomsPersonnages = [variantesNoms for variantesNoms in listeNomsPersonnages if variantesNoms]

    ##Supprimer les listes qui sont des sous-listes d'autres
    # for variantesNoms in listeNomsPersonnages[:]:  # Copier pour éviter les conflits pendant la suppression
        # for s in listeNomsPersonnages[:]:
            # if variantesNoms != s and all(nom in s for nom in variantesNoms):  # Si variantesNoms est un sous-ensemble de s
                # if variantesNoms in listeNomsPersonnages:  # Vérifier que variantesNoms n'a pas déjà été supprimé
                    # listeNomsPersonnages.remove(variantesNoms)
            # elif variantesNoms != s and all(nom in variantesNoms for nom in s):  # Si s est un sous-ensemble de variantesNoms
                # if s in listeNomsPersonnages:  # Vérifier que s n'a pas déjà été supprimé
                    # listeNomsPersonnages.remove(s)
                    
                    

    # print("//////////////////Famille pas ensemble/////////////////////")
    # for variantes in listeNomsPersonnages:
        # print(variantes)
    
    
    if book_code == "paf":
        listeNomsPersonnagesPaf = listeNomsPersonnages
    else :
        listeNomsPersonnagesLca = listeNomsPersonnages
            
            
            
            




