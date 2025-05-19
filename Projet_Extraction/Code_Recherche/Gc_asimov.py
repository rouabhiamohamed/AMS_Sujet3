import gc
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

# Télécharger le modèle Stanza pour le français si nécessaire
stanza.download('fr')

# Initialisation du pipeline de traitement de texte pour le français
nlp = stanza.Pipeline('fr', processors='tokenize,mwt,pos,ner')

### Liste des livres        
books = [
    "Fondation_et_empire_sample",
    "Fondation_foudroyée_sample",
    "Fondation_sample",
    "Seconde_Fondation_sample",
    "Terre_et_Fondation_sample"
]

df_dict = {"ID": [], "graphml": []}

def Filtre_Ner_Pos(doc, listeFiltre, tokens):    
    listePROPN = []
    for sent in doc.sentences:
        tokens.extend([word.text for word in sent.words])
        for word in sent.words:
            if word.upos in {"VERB", "AUX", "INTJ", "ADJ", "ADP", "ADV", "X", "PRON", "NOUN"} or \
               (word.upos == "NOUN" and "Number=Plur" in word.feats) or \
               (word.upos == "PROPN" and "Number=Plur" in word.feats):
                listeFiltre.append(word.text)
            elif word.upos == "PROPN":
                listePROPN.append(word.text)
    
    for wordPROPN in listePROPN:
        if wordPROPN in listeFiltre:
            listeFiltre.remove(wordPROPN)

def Extraction_Per(doc, listePersonnages, listeFiltre):
    for ent in doc.ents:
        if ent.type == "PER" and len(ent.text) > 2 and ent.text not in listeFiltre:
            ent.text = ent.text.replace("\n", " ").strip()
            listePersonnages.append(ent.text)
        elif ent.type == "PER" and len(ent.text) > 2 and ent.text in listeFiltre:
            listeRetirer.append(ent.text)

def Tri_Lieux(listePersonnagesTrier, listeLieuxAll, listeRetirer):
    lieux_set = set(listeLieuxAll)
    for personnage in listePersonnagesTrier[:]:
        if personnage in lieux_set:
            listePersonnagesTrier.remove(personnage)
            listeRetirer.append(personnage)

def Tri_Parasite(listePersonnagesTrier, listeRetirer): 
    for personnage in listePersonnagesTrier[:]:
        remove = False
        docVerif = nlp(personnage)
        for sent in docVerif.sentences:
            for word in sent.words:
                if word.upos in {"INTJ", "DET", "VERB", "ADP"} or \
                   (word.upos == "NOUN" and "Number=Plur" in word.feats) or \
                   (len(personnage.split()) == 1 and word.upos == "X"):
                    listePersonnagesTrier.remove(personnage)
                    listeRetirer.append(personnage)
                    break

def Ranger_Par_Noms(listePersonnagesTrier, listeNomsPersonnages):                                    
    for nom in listePersonnagesTrier:
        variantesNoms = [nom]  
        for autreNom in listePersonnagesTrier:
            if len(nom.split()) >= 2 and autreNom != nom and (autreNom in nom.split()[-1] or nom.split()[-1] in autreNom):
                if autreNom not in variantesNoms:  
                    variantesNoms.append(autreNom)
        if not any(sorted(variantesNoms) == sorted(existing) for existing in listeNomsPersonnages):
            listeNomsPersonnages.append(variantesNoms)

# Continuez avec les autres fonctions de la même manière

# Optimisation pour libérer la mémoire
def clear_memory():
    gc.collect()

# Liste de tout les lieux de l'entièreté du corpus
listeLieuxAll = []

# Traitement par livre
for book in books:
    with open(f"Textes_Processed/{book}.txt", "r") as file:
        texte = file.read()
    
    texte = re.sub(r'\b[A-ZÀ-ÿ0-9]+\b', lambda match: match.group(0).lower(), texte)    
    sentence = Sentence(texte)

    tagger.predict(sentence)
    loc_entities = [entity.text for entity in sentence.get_spans('ner') if entity.get_label('ner').value == 'LOC']

    listeLieuxAll.extend(loc_entities)

# Traiter chaque livre
for book in books:
    G = nx.Graph()
    
    with open(f"Textes_Processed/{book}.txt", "r") as file:
        texte = file.read()
    
    doc = nlp(texte)

    listePersonnages = []
    listeRetirer = []
    listeFiltre = []
    tokens = []

    Filtre_Ner_Pos(doc, listeFiltre, tokens)
    Extraction_Per(doc, listePersonnages, listeFiltre)
    
    listePersonnagesTrier = list(dict.fromkeys(listePersonnages))
    
    Tri_Lieux(listePersonnagesTrier, listeLieuxAll, listeRetirer)
    Tri_Parasite(listePersonnagesTrier, listeRetirer)

    listeNomsPersonnages = []
    Ranger_Par_Noms(listePersonnagesTrier, listeNomsPersonnages)

    Supprimer_Sous_Liste(listeNomsPersonnages)
    Ranger_Par_Prenoms(listeNomsPersonnages)
    Supprimer_Sous_Liste(listeNomsPersonnages)
    Separation_Alias(listeNomsPersonnages)

    listeNomsPersonnages = [variantesNoms for variantesNoms in listeNomsPersonnages if variantesNoms]
    Supprimer_Sous_Liste(listeNomsPersonnages)

    for i in range(len(listeNomsPersonnages)):
        listeNomsPersonnages[i] = list(dict.fromkeys(listeNomsPersonnages[i])) 

    print(f"//////////////////{book}////////////////////")

    for i in listeNomsPersonnages:
        print(i)

    print(f"/////////////////Liste d'éléments retirer de ce chapitre////////////////////")
    print(list(dict.fromkeys(listeRetirer)))
    
    dictionnaireRelationsUnique = {}
    entity_positions = Association_Position_personnage(tokens, listePersonnagesTrier)
    Relations(listeNomsPersonnages, entity_positions, dictionnaireRelationsUnique)

    dictionnaireRelationsListe = {} 

    for i in listeNomsPersonnages:
        first_element = list(i)[0] if isinstance(i, set) else i[0]
        if first_element not in dictionnaireRelationsListe:
            dictionnaireRelationsListe[first_element] = []
        
        for key, value in dictionnaireRelationsUnique.items():
            if key in i:
                if value not in dictionnaireRelationsListe[first_element]:
                    dictionnaireRelationsListe[first_element].append(value)

    for source, targets in dictionnaireRelationsListe.items():
        for target in targets:
            for group in listeNomsPersonnages:
                if target in group:
                    if G.has_edge(source, group[0]):
                        G[source][group[0]]['weight'] += 1
                    else:
                        G.add_edge(source, group[0], weight=1)

    for group in listeNomsPersonnages:
        group = list(group)
        first_element = group[0]
        remaining_elements = ";".join(group[1:])

        if first_element not in G.nodes:
            G.add_node(first_element)
        G.nodes[first_element]["names"] = f"{first_element};{remaining_elements}" if remaining_elements else first_element

    for node in G.nodes:
        if "names" not in G.nodes[node]:
            G.nodes[node]["names"] = node
    
    net = Network(notebook=True)
    net.from_nx(G)
    net.show(f"graphes/{book}.html")

    df_dict["ID"].append(f"{book}")
    graphml = "".join(nx.generate_graphml(G))
    df_dict["graphml"].append(graphml)

    clear_memory()  # Libérer la mémoire après chaque livre

df = pd.DataFrame(df_dict)
df.set_index("ID", inplace=True)
df.to_csv("./my_submission.csv")
