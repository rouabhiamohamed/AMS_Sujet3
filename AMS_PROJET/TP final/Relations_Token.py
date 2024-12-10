import spacy
import networkx as nx
from flair.data import Sentence
from flair.models import SequenceTagger
from collections import defaultdict
from itertools import combinations

# Charger le modèle NER pour la reconnaissance d'entités
tagger = SequenceTagger.load("flair/ner-french")

# Charger spaCy pour le traitement des textes
nlp = spacy.blank("fr")

# Définir la distance maximale pour considérer deux personnages comme étant en co-occurrence
CO_OCCURRENCE_DISTANCE = 25

def detect_cooccurrences(tokens, resolved_entities):
    entity_positions = defaultdict(list)
    for i, token in enumerate(tokens):
        for group in resolved_entities:
            if token in group:
                entity_positions[group].append(i)
    
    cooccurrences = {}
    for entity1, entity2 in combinations(entity_positions.keys(), 2):
        for pos1 in entity_positions[entity1]:
            for pos2 in entity_positions[entity2]:
                if abs(pos1 - pos2) <= CO_OCCURRENCE_DISTANCE:
                    cooccurrences[entity1] = entity2
    return cooccurrences
    
    
###Liste des livres et chapitres à extraire        
books = [
    (list(range(0, 19)), "paf"),
    (list(range(0, 18)), "lca"),
]

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
            with open(f"reseaux-de-personnages-de-fondation-session-2/{repertory}/chapter_{chapter}.txt.preprocessed", "r") as file:
                texte = file.read()

            doc = nlp(texte)
            tokens = [token.text for token in doc]
            
            cooccurrences = detect_cooccurrences(tokens, listePersonnagesTrier)
            print("/////////////////",book_code,chapter,"////////////////////")
            print(cooccurrences)

    
    
    