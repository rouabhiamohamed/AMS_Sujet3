import stanza
import networkx as nx
import pandas as pd
from collections import defaultdict
import re
from flair.data import Sentence
from flair.models import SequenceTagger
import numpy as np
from pyvis.network import Network

# Charger le modèle NER pour le français
tagger = SequenceTagger.load("flair/ner-french")

###Liste des livres        
books = [
    "Fondation_et_empire_sample",
    "Fondation_foudroyée_sample",
    "Fondation_sample",
    "Seconde_Fondation_sample",
    "Terre_et_Fondation_sample"
]

def load_and_split_text(book_path):
    with open(book_path, "r") as file:
        texte = file.read()
    
    # Utilisation d'une expression régulière pour découper le texte en pages sur la base des délimiteurs <1>, <2>, etc.
    pages = re.split(r'<->', texte)
    return pages


#Liste de tout les lieux de l'entièreté du corpus
listeLieuxAll = []

##Bout de code qui permet de mettre dans une liste tout les lieux du Corpus grâce à Flair
for book in books:
    print(f"////////////////////////////////////Livre {book}////////////////////////////////////")
    pages = load_and_split_text(f"Textes_Processed/{book}.txt")
    
    for page in pages:
        # Créer une phrase à partir du texte
        sentence = Sentence(page)

        # Prédire les étiquettes NER
        tagger.predict(sentence)

        # Liste pour stocker les entités de type LOC
        loc_entities = []

        # Itérer sur les entités reconnues et ajouter les LOC à la liste
        for entity in sentence.get_spans('ner'):
            if entity.get_label('ner').value == 'LOC':  # Vérifier si l'entité est un lieu
                loc_entities.append(entity.text)
                
        #Pour ajouter dans la liste final de lieux 
        for lieux in loc_entities:
            listeLieuxAll.append(lieux)
            
print("//////////////////Répertorisation de tout les Lieux dans une liste : Fait !/////////////////////")
print(listeLieuxAll)