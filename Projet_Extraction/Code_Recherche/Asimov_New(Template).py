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

###Liste des livres        
books = [
    "Fondation_et_empire_sample",
    "Fondation_foudroyée_sample",
    "Fondation_sample",
    "Seconde_Fondation_sample",
    "Terre_et_Fondation_sample"
]

df_dict = {"ID": [], "graphml": []}


##Bout de code pour enlever les faux noms détecter par le ner, grâce aux POS
def Filtre_Ner_Pos(doc,listeFiltre,tokens):    
    listePROPN = []
    for sent in doc.sentences:
        tokens.extend([word.text for word in sent.words])
        for word in sent.words:
            if(word.upos=="VERB" 
            or word.upos=="AUX"
            or word.upos=="INTJ" 
            or word.upos=="ADJ" 
            or word.upos=="ADP" 
            or word.upos=="ADV" 
            or word.upos=="X"
            or word.upos=="PRON"
            or word.upos=="NOUN"
            or (word.upos == "NOUN" and word.feats and "Number=Plur" in word.feats)
            or (word.upos == "PROPN" and word.feats and "Number=Plur" in word.feats)):
                listeFiltre.append(word.text)
            elif (word.upos=="PROPN"):
                listePROPN.append(word.text)
            
    for wordPROPN in listePROPN :
        if(wordPROPN in listeFiltre):
            listeFiltre.remove(wordPROPN)

### Extraction des entités nommées de type "PER" (personne)
def Extraction_Per(doc,listePersonnages,listeFiltre):
    for ent in doc.ents:
        if ent.type == "PER" and len(ent.text)>2 and ent.text not in listeFiltre:
            ent.text = ent.text.replace("\n", " ").strip() ###Certaines entités ont des \n, on les enlève
            listePersonnages.append(ent.text)
        elif ent.type == "PER" and len(ent.text)>2 and ent.text in listeFiltre:
            listeRetirer.append(ent.text)

### Tri des entités détecté Per mais qui sont des lieux 
def Tri_Lieux(listePersonnagesTrier,listeLieuxAll,listeRetirer):
    lieux_set = set(listeLieuxAll)

    # Créer une copie de listePersonnagesTrier pour itérer sans modifier la liste pendant la boucle
    for personnage in listePersonnagesTrier[:]:
        # Si le personnage est un lieu détecté, on le retire de la liste
        if personnage in lieux_set:
            listePersonnagesTrier.remove(personnage)
            listeRetirer.append(personnage)

### Tri des personnages "Parasites" qui n'ont pas été enlever 
def Tri_Parasite(listePersonnagesTrier,listeRetirer): 
    for personnage in listePersonnagesTrier[:]:
        remove = False
        docVerif = nlp(personnage)
        for sent in docVerif.sentences:
            for word in sent.words:
                if(word.upos=="INTJ" or word.upos=="DET" or word.upos=="VERB" or word.upos=="ADP"):
                    listePersonnagesTrier.remove(personnage)
                    listeRetirer.append(personnage)
                    break
                elif(word.upos=="NOUN" and word.feats and "Number=Plur" in word.feats):
                    listePersonnagesTrier.remove(personnage)
                    listeRetirer.append(personnage)
                elif(len(personnage.split()) == 1 and word.upos=="X"):
                    listePersonnagesTrier.remove(personnage)
                    listeRetirer.append(personnage)

def Ranger_Par_Noms(listePersonnagesTrier,listeNomsPersonnages):                                    
    for nom in listePersonnagesTrier:
        variantesNoms = [nom]  # Utiliser une liste pour stocker les variantes
        for autreNom in listePersonnagesTrier:
            if (len(nom.split()) >= 2 and autreNom != nom and (autreNom in nom.split()[-1] or nom.split()[-1] in autreNom)):  # Vérifier si c'est une variante
                if autreNom not in variantesNoms:  # Ajouter uniquement si ce n'est pas déjà présent
                    variantesNoms.append(autreNom)

        # Vérifier si une liste équivalente n'est pas déjà dans listeNomsPersonnages
        if not any(sorted(variantesNoms) == sorted(existing) for existing in listeNomsPersonnages):
            listeNomsPersonnages.append(variantesNoms) 
    
def Ranger_Par_Prenoms(listeNomsPersonnages):                                                                     
    for index, variantesNoms in enumerate(listeNomsPersonnages):
        for variantesNomsAAjouter in listeNomsPersonnages[:]:  # Copie de la liste pour éviter les problèmes
            if(len(variantesNomsAAjouter)==1 and len(variantesNoms)>1):
                trouver = False
                for nomsAComparer in variantesNoms :
                    if variantesNomsAAjouter[0] in nomsAComparer and not trouver:
                        trouver = True
                        listeNomsPersonnages[index].append(variantesNomsAAjouter[0])    

# Pour séparer les alias avec les mêmes noms de famille, mais pas les mêmes prénoms    
def Separation_Alias(listeNomsPersonnages):                                                                     
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
    
# Supprimer les listes qui sont des sous-listes d'autres
def Supprimer_Sous_Liste(listeNomsPersonnages):
    for variantesNoms in listeNomsPersonnages[:]:  # Copier pour éviter les conflits pendant la suppression
        for s in listeNomsPersonnages[:]:
            if variantesNoms != s and all(nom in s for nom in variantesNoms):  # Si variantesNoms est un sous-ensemble de s
                if variantesNoms in listeNomsPersonnages:  # Vérifier que variantesNoms n'a pas déjà été supprimé
                    listeNomsPersonnages.remove(variantesNoms)
            elif variantesNoms != s and all(nom in variantesNoms for nom in s):  # Si s est un sous-ensemble de variantesNoms
                if s in listeNomsPersonnages:  # Vérifier que s n'a pas déjà été supprimé
                    listeNomsPersonnages.remove(s)


def Association_Position_personnage(tokens,listePersonnagesTrier):
    entity_positions = defaultdict(list)
    for i, token in enumerate(tokens):
        for group in listePersonnagesTrier:
            if token in group:
                entity_positions[group].append(i)
    return entity_positions
                
def Relations(listeNomsPersonnages, entity_positions, dictionnaireRelationsUnique):
    # Parcourir les noms de chaque personnage
    for i in listeNomsPersonnages:
        for j in i:  # Parcourt chaque alias (variantes) du personnage
            entity1 = j
            for k in listeNomsPersonnages:
                for l in k:
                    if k != i:  # Assurer que ce sont deux personnages distincts
                        entity2 = l
                        for pos1 in entity_positions[entity1]:
                            for pos2 in entity_positions[entity2]:
                                if abs(pos1 - pos2) <= 25: # abs : distance absolue entre deux positions est utilisée, peu importe si pos1 est plus petit ou plus grand que pos2.
                                    dictionnaireRelationsUnique[entity1] = entity2
                                       

#Liste de tout les lieux de l'entièreté du corpus
listeLieuxAll = []

##Bout de code qui permet de mettre dans une liste tout les lieux du Corpus grâce à Flair
for book in books:
    with open(f"Textes_Processed/{book}.txt", "r") as file:
        texte = file.read()
    
    texte = re.sub(r'\b[A-ZÀ-ÿ0-9]+\b', lambda match: match.group(0).lower(), texte)    # nettoyage des mots tout en majuscule
    
    # Créer une phrase à partir du texte
    sentence = Sentence(texte)

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

for book in books:


    G = nx.Graph()

    with open(f"Textes_Processed/{book}.txt", "r") as file:
        texte = file.read()
        
    # Traitement du texte
    doc = nlp(texte)

    listePersonnages = []
    listeRetirer = []
    listeFiltre = []
    tokens = []

    ##Bout de code pour enlever les faux noms détecter par le ner, grâce aux POS
    Filtre_Ner_Pos(doc,listeFiltre,tokens)

    ### Liste pour stocker les personnages extraits
    Extraction_Per(doc,listePersonnages,listeFiltre)
    
    # Liste pour stocker les personnages uniques après filtrage
    listePersonnagesTrier = list(dict.fromkeys(listePersonnages))  # Utiliser un set pour obtenir les personnages uniques
    
    ### Tri des entités détecté Per mais qui sont des lieux 
    Tri_Lieux(listePersonnagesTrier,listeLieuxAll,listeRetirer)
            
    ### Tri des personnages "Parasites" qui n'ont pas été enlever 
    Tri_Parasite(listePersonnagesTrier,listeRetirer)
        
    listeNomsPersonnages = []
    
    #D'abord les noms de famille
    Ranger_Par_Noms(listePersonnagesTrier,listeNomsPersonnages)

    # Supprimer les listes qui sont des sous-listes d'autres
    Supprimer_Sous_Liste(listeNomsPersonnages)
                    
    #Puis rajouter les prénoms isoler dans une liste
    Ranger_Par_Prenoms(listeNomsPersonnages)

    # Supprimer les listes qui sont des sous-listes d'autres
    Supprimer_Sous_Liste(listeNomsPersonnages)
                    
    # Pour séparer les alias avec les mêmes noms de famille, mais pas les mêmes prénoms
    Separation_Alias(listeNomsPersonnages)

    #Pour nettoyer la liste des listes vides 
    listeNomsPersonnages = [variantesNoms for variantesNoms in listeNomsPersonnages if variantesNoms]

    # Supprimer les listes qui sont des sous-listes d'autres
    Supprimer_Sous_Liste(listeNomsPersonnages)
                                
    ##Pour nettoyer les doublons apparu                
    for i in range(len(listeNomsPersonnages)):
        listeNomsPersonnages[i] = list(dict.fromkeys(listeNomsPersonnages[i])) 

    print("/////////////////",book,"////////////////////")

    for i in listeNomsPersonnages:
        print(i)
    print("/////////////////Liste d'éléments retirer de ce chapitre////////////////////")
    print(list(dict.fromkeys(listeRetirer)))
    
    ##Dictionnaire qui contiendra la liste de toutes les relations entre toutes les entités
    dictionnaireRelationsUnique = {}
    
    entity_positions = Association_Position_personnage(tokens,listePersonnagesTrier)
    
    Relations(listeNomsPersonnages,entity_positions,dictionnaireRelationsUnique)
    
    
    ###Dictionnaire qui contiendra la liste de toutes les entités aillant des relations, ainsi que leurs relations
    ### key : EntitéSource, value : Liste d'EntitéTarget
    dictionnaireRelationsListe = {} 

    ###Bout de code qui, via dictionnaireRelationsUnique, rempli le Dictionnaire de Liste de Relation : dictionnaireRelationsListe
    for i in listeNomsPersonnages:
        # Obtenir le premier élément de l'ensemble ou de la liste
        first_element = list(i)[0] if isinstance(i, set) else i[0]
        
        if first_element not in dictionnaireRelationsListe:
            dictionnaireRelationsListe[first_element] = []
        
        for key, value in dictionnaireRelationsUnique.items():
            if key in i:
                if value not in dictionnaireRelationsListe[first_element]:
                    dictionnaireRelationsListe[first_element].append(value)

    ###Bout de code qui rempli le graphe en ajoutant les noeuds et les arrêtes via dictionnaireRelationsListe
    for source, targets in dictionnaireRelationsListe.items():
        for target in targets:
            for group in listeNomsPersonnages:
                if target in group:
                    if G.has_edge(source, group[0]):
                        # Si l'arête existe déjà, augmenter son poids
                        G[source][group[0]]['weight'] += 1
                    else:
                        # Sinon, créer une nouvelle arête avec poids 1
                        G.add_edge(source, group[0], weight=1)

    for group in listeNomsPersonnages:
        group = list(group)  # Convertir en liste pour un accès facile
        first_element = group[0]  # Prendre le premier élément comme nom principal
        remaining_elements = ";".join(group[1:])  # Joindre les variantes, s'il y en a

        # Ajouter le nœud avec ses attributs s'il n'est pas déjà présent
        if first_element not in G.nodes :
            G.add_node(first_element)  # Ajouter un nœud isolé
        G.nodes[first_element]["names"] = f"{first_element};{remaining_elements}" if remaining_elements else first_element

    # Ajout des attributs manquants
    for node in G.nodes:
        if "names" not in G.nodes[node]:
            G.nodes[node]["names"] = node
        
    # Visualisation du graphe avec pyvis
    net = Network(notebook=True)
    net.from_nx(G)
    net.show(f"graphes/{book}.html")
    
    df_dict["ID"].append(book)
    graphml = "".join(nx.generate_graphml(G))
    df_dict["graphml"].append(graphml)
                

df = pd.DataFrame(df_dict)
df.set_index("ID", inplace=True)
df.to_csv("./my_submission.csv")