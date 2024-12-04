import stanza
from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess
import networkx as nx
import pandas as pd
from collections import defaultdict
import re
from flair.data import Sentence
from flair.models import SequenceTagger

# Charger le modèle NER pour le français
tagger = SequenceTagger.load("flair/ner-french")

# Télécharger le modèle Stanza pour le français si nécessaire
stanza.download('fr')

# Initialisation du pipeline de traitement de texte pour le français
nlp = stanza.Pipeline('fr', processors='tokenize,mwt,pos,ner')
        
###Liste des livres et chapitres à extraire        
books = [
    (list(range(0, 19)), "paf"),
    (list(range(0, 18)), "lca"),
]

df_dict = {"ID": [], "graphml": []}

lieuxASupprimer = [] #Liste pour stocker les éléments à supprimer

listeLieuxAll = []

for chapters, book_code in books:
    for chapter in chapters:
        if book_code == "paf":
            repertory = "prelude_a_fondation"
        else:
            repertory = "les_cavernes_d_acier"
        if chapter!=0:
            with open(f"reseaux-de-personnages-de-fondation-session-2/{repertory}/chapter_{chapter}.txt.preprocessed", "r") as file:
                texte = file.read()
           
            texte = re.sub(r'\b[A-ZÀ-ÿ0-9]+\b', lambda match: match.group(0).lower(), texte)    # nettoyage des mots tout en majuscule
            # Créer une phrase à partir du texte
            sentence = Sentence(texte)

            # Prédire les étiquettes NER
            tagger.predict(sentence)

            # Afficher la phrase avec les entités
            # print(sentence)

            # Liste pour stocker les entités de type LOC
            loc_entities = []

            # Itérer sur les entités reconnues et ajouter les LOC à la liste
            for entity in sentence.get_spans('ner'):
                if entity.get_label('ner').value == 'LOC':  # Vérifier si l'entité est un lieu
                    loc_entities.append(entity.text)
                    
            #Pour ajouter dans la liste final de lieux 
            for lieux in loc_entities:
                listeLieuxAll.append(lieux)



for chapters, book_code in books:
    for chapter in chapters:
        if book_code == "paf":
            repertory = "prelude_a_fondation"
        else:
            repertory = "les_cavernes_d_acier"
        if chapter!=0:
            with open(f"reseaux-de-personnages-de-fondation-session-2/{repertory}/chapter_{chapter}.txt.preprocessed", "r") as file:
                texte = file.read()
                
            texte = re.sub(r'\b[A-ZÀ-ÿ0-9]+\b', lambda match: match.group(0).lower(), texte)    # nettoyage des mots tout en majuscule
            #texte = re.sub(r'-', " ", texte)    
            # Traitement du texte
            doc = nlp(texte)
 
            ##Bout de code pour enlever les faux noms détecter par le ner, grâce aux POS
            listePROPN = []
            listeFiltre = []            
            for sent in doc.sentences:
                for word in sent.words:
                    if(word.upos=="VERB" 
                    or word.upos=="AUX"
                    or word.upos=="INTJ" 
                    or word.upos=="ADJ" 
                    or word.upos=="ADP" 
                    or word.upos=="ADV" 
                    or word.upos=="PRON"
                    or (word.upos == "NOUN" and word.feats and "Number=Plur" in word.feats)
                    or (word.upos == "PROPN" and word.feats and "Number=Plur" in word.feats)):
                        listeFiltre.append(word.text)
                    elif (word.upos=="PROPN"):
                        listePROPN.append(word.text)
                    wordUposPred = word.upos
                    
            
            for wordPROPN in listePROPN :
                if(wordPROPN in listeFiltre):
                    listeFiltre.remove(wordPROPN)
                  
            
            ### Liste pour stocker les personnages extraits
            listePersonnages = []
            listeLieux = []
            listeNomsPersonnages = []
            listeRetirer = []
            

            ### Extraction des entités nommées de type "PER" (personne)
            for ent in doc.ents:
                if ent.type == "PER" and len(ent.text)>2 and ent.text not in listeFiltre:
                    ent.text = ent.text.replace("\n", " ").strip() ###Certaines entités ont des \n, on les enlève
                    listePersonnages.append(ent.text)
                elif ent.type == "PER" and len(ent.text)>2 and ent.text in listeFiltre:
                    listeRetirer.append(ent.text)
                # if ent.type == "LOC":
                    # listeLieux.append(ent.text)
            
           
            # Liste pour stocker les personnages uniques après filtrage
            listePersonnagesTrier = list(dict.fromkeys(listePersonnages))  # Utiliser un set pour obtenir les personnages uniques
            
            
            
            
            lieux_set = set(listeLieuxAll)

            # Créer une copie de listePersonnagesTrier pour itérer sans modifier la liste pendant la boucle
            for personnage in listePersonnagesTrier[:]:
                # Si le personnage est un lieu détecté, on le retire de la liste
                if personnage in lieux_set:
                    listePersonnagesTrier.remove(personnage)
                    listeRetirer.append(personnage)

            # Écriture dans un fichier par chapitre
            with open(f"Partie2/{repertory}_chapter_{chapter}_characters.txt", "w") as file:
                file.write("\n".join(listePersonnagesTrier))
           



            print(chapter, book_code)
            print("////////////////Liste de Noms/////////////////")
            print(listePersonnagesTrier)
            print("////////////////Liste de mots retirer/////////////////")
            print(list(dict.fromkeys(listeRetirer)))
            # print("////////////////Liste de lieux/////////////////")
            # print(listeLieux)
            print("\n")
            
           
    
print("////////////////Liste de Lieux à retirer/////////////////")
print(list(dict.fromkeys(listeLieuxAll)))