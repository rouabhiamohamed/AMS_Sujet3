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
            
            with open(f"Bigrammes/Bigrammes_of_{repertory}_chapter_{chapter+1}.txt", 'r') as file:
                # Liste pour stocker les bigrammes sous forme de liste de listes
                listeBigramme = [line.strip().split() for line in file if line.strip()]
            
            for noms in listePersonnagesTrier :
                if(len(noms.split())==1):
                    for bigramme in listeBigramme:
                        docBi = nlp(bigramme[0])
                        PosBigramme = [word.upos for sent in docBi.sentences for word in sent.words]
                        if(bigramme[1]==noms and "DET" in PosBigramme):
                            print(noms)
           
            for perso in listePersonnagesTrier:
                Listeperso.append(perso)
                
        
    listePersonnagesTrier = list(dict.fromkeys(Listeperso))
    
    
    
    listeNomsPersonnages = []

    #D'abord les noms de famille
    for nom in listePersonnagesTrier:
        variantesNoms = [nom]  # Utiliser une liste pour stocker les variantes
        for autreNom in listePersonnagesTrier:
            if (len(nom.split()) >= 2 and autreNom != nom and (autreNom in nom.split()[-1] or nom.split()[-1] in autreNom)):  # Vérifier si c'est une variante
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
                    
    #Puis rajouter les prénoms isoler dans une liste
    for index, variantesNoms in enumerate(listeNomsPersonnages):
        for variantesNomsAAjouter in listeNomsPersonnages[:]:  # Copie de la liste pour éviter les problèmes
            if(len(variantesNomsAAjouter)==1 and len(variantesNoms)>1):
                trouver = False
                for nomsAComparer in variantesNoms :
                    if variantesNomsAAjouter[0] in nomsAComparer and not trouver:
                        trouver = True
                        listeNomsPersonnages[index].append(variantesNomsAAjouter[0])
    
    # Supprimer les listes qui sont des sous-listes d'autres
    for variantesNoms in listeNomsPersonnages[:]:  # Copier pour éviter les conflits pendant la suppression
        for s in listeNomsPersonnages[:]:
            if variantesNoms != s and all(nom in s for nom in variantesNoms):  # Si variantesNoms est un sous-ensemble de s
                if variantesNoms in listeNomsPersonnages:  # Vérifier que variantesNoms n'a pas déjà été supprimé
                    listeNomsPersonnages.remove(variantesNoms)
            elif variantesNoms != s and all(nom in variantesNoms for nom in s):  # Si s est un sous-ensemble de variantesNoms
                if s in listeNomsPersonnages:  # Vérifier que s n'a pas déjà été supprimé
                    listeNomsPersonnages.remove(s)
                    
    
    # Pour séparer les alias avec les mêmes noms de famille, mais pas les mêmes prénoms
    fin=False
    while(fin==False):
        fin=True
        for variantesNoms in listeNomsPersonnages[:]:
            for personnage in variantesNoms:
                listeDeNoms = []
                listeDeNomsSimple = []
                nomDeFamille = ""
                if len(personnage.split()) >= 2 and len(personnage.split()[0])>2:
                    for personnageAComparer in variantesNoms:
                        if len(personnageAComparer.split()) >= 2 and len(personnageAComparer.split()[0])>2 and personnage!=personnageAComparer:
                            docPrenom1 = nlp(personnage)
                            tokenPrenom1 = [word.upos for sent in docPrenom1.sentences for word in sent.words]
                            docPrenom2 = nlp(personnageAComparer)
                            tokenPrenom2 = [word.upos for sent in docPrenom2.sentences for word in sent.words]
                            if personnage.split()[0] != personnageAComparer.split()[0] and \
                                ("PROPN" in tokenPrenom1[0] and "PROPN" in tokenPrenom2[0]):
                                listeDeNoms.append(personnageAComparer)
                                nomDeFamille = personnageAComparer.split()[-1]
                                fin=False
                                
                for personnageARajouter in variantesNoms:
                    if len(personnageARajouter.split()) == 1 and personnageARajouter not in listeDeNoms:
                        # Vérifier si personnageARajouter est un sous-ensemble de nomAComparer
                        for nomAComparer in listeDeNoms:
                            if personnageARajouter in nomAComparer:
                                listeDeNomsSimple.append(personnageARajouter)
                listeDeNomsConcatenee = listeDeNoms + listeDeNomsSimple
                # Enlever les éléments de listeDeNomsConcatenee de la liste originale
                for elem in listeDeNomsConcatenee:
                    if elem in variantesNoms and elem != nomDeFamille:
                        variantesNoms.remove(elem)
                        
                listeNomsPersonnages.append(listeDeNomsConcatenee)
    
    #Pour nettoyer la liste des listes vides 
    listeNomsPersonnages = [variantesNoms for variantesNoms in listeNomsPersonnages if variantesNoms]
    
    # Supprimer les listes qui sont des sous-listes d'autres
    for variantesNoms in listeNomsPersonnages[:]:  # Copier pour éviter les conflits pendant la suppression
        for s in listeNomsPersonnages[:]:
            if variantesNoms != s and all(nom in s for nom in variantesNoms):  # Si variantesNoms est un sous-ensemble de s
                if variantesNoms in listeNomsPersonnages:  # Vérifier que variantesNoms n'a pas déjà été supprimé
                    listeNomsPersonnages.remove(variantesNoms)
            elif variantesNoms != s and all(nom in variantesNoms for nom in s):  # Si s est un sous-ensemble de variantesNoms
                if s in listeNomsPersonnages:  # Vérifier que s n'a pas déjà été supprimé
                    listeNomsPersonnages.remove(s)
    
    print("//////////////////Nom de Famille/////////////////////")
    for variantes in listeNomsPersonnages:
        print(variantes)
    
    if book_code == "paf":
        listeNomsPersonnagesPaf = listeNomsPersonnages
    else :
        listeNomsPersonnagesLca = listeNomsPersonnages
            
            
            
            




