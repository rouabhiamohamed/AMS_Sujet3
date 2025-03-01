import stanza
import networkx as nx
import pandas as pd
from collections import defaultdict
import re
from flair.data import Sentence
from flair.models import SequenceTagger
import numpy as np
from pyvis.network import Network
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


tagger = SequenceTagger.load("flair/ner-french") # Charger le modèle NER pour le français
stanza.download('fr') # Télécharger le modèle Stanza pour le français si nécessaire
nlp = stanza.Pipeline('fr', processors='tokenize,mwt,pos,ner') # Initialisation du pipeline de traitement de texte pour le français
analyzer = SentimentIntensityAnalyzer()#Pour Vader

#Liste des livres        
books = [
    "Fondation_et_empire_sample"
    #"Fondation_sample",
    #"Seconde_Fondation_sample",
    #"Terre_et_Fondation_sample",
    #"Fondation_foudroyée_sample"
]

df_dict = {"ID": [], "graphml": []}

def DecoupageTexteEnPages(book_path):
    with open(book_path, "r") as file:
        texte = file.read()
    pages = re.split(r'<->', texte)
    return pages

##Bout de code pour enlever les faux noms détecter par le ner, grâce aux POS
def Filtre_Ner_Pos(doc,listeFiltre):    
    listePROPN = []
    for sent in doc.sentences:
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
def Extraction_Per(doc,listePersonnages,listeFiltre,listeRetirer):
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


##Bout de code qui permet de mettre dans une liste tout les lieux du Corpus grâce à Flair
def AnalyseLieux():
    listeLieux = [] #Liste de tout les lieux de l'entièreté du corpus
    for book in books:
        print(f"////////////////////////////////////Livre {book}////////////////////////////////////")
        pages = DecoupageTexteEnPages(f"Textes_Processed/{book}.txt")
        for page in pages:
            sentence = Sentence(page) # Créer une phrase à partir du texte
            tagger.predict(sentence) # Prédire les étiquettes NER
            entite_LOC = [] # Liste pour stocker les entités de type LOC
            for entitee in sentence.get_spans('ner'): # Itérer sur les entités reconnues et ajouter les LOC à la liste
                if entitee.get_label('ner').value == 'LOC':  # Vérifier si l'entité est un lieu
                    entite_LOC.append(entitee.text)
            for lieux in entite_LOC: #Pour ajouter dans la liste final de lieux 
                listeLieux.append(lieux)
                
    print("//////////////////Répertorisation de tout les Lieux dans une liste : Fait !/////////////////////")
    return listeLieux


def AnalyseEntitee(book,page_sentiment,pages,listeRetirer):
    print(f"////////////////////////////////////Livre {book}////////////////////////////////////")
    listePersonnages = []
    listeFiltre = []

    for i, page in enumerate(pages):
        doc = nlp(page) # Traitement du texte
        Filtre_Ner_Pos(doc,listeFiltre) ##Bout de code pour enlever les faux noms détecter par le ner, grâce aux POS
        Extraction_Per(doc,listePersonnages,listeFiltre,listeRetirer) ### Liste pour stocker les personnages extraits
        listePersonnagesTrier = list(dict.fromkeys(listePersonnages)) # Liste pour stocker les personnages uniques après filtrage  # Utiliser un set pour obtenir les personnages uniques
        page_sentiment[i].append(Sentiment_page(page))
    
    return listePersonnagesTrier


def Association_Position_personnage(pages,listePersonnagesTrier):
    entity_positions = defaultdict(list)
    for i, page in enumerate(pages):
        for group in listePersonnagesTrier:
            if(group in page):
                entity_positions[group].append(i)
    return entity_positions

def Sentiment_page(page):
    sentiment_score = analyzer.polarity_scores(page)
    if sentiment_score['compound'] > 0:
        return 1
    elif sentiment_score['compound'] < 0:
        return -1
    else:
        return 0

def Relations(G,listeNomsPersonnages, entity_positions,page_sentiment):
    dictionnaireRelations = {}
    # Parcourir les noms de chaque personnage
    for i in listeNomsPersonnages:
        first_element = list(i)[0] if isinstance(i, set) else i[0]
        if first_element not in dictionnaireRelations:
            dictionnaireRelations[first_element] = []
        for j in i:  # Parcourt chaque alias (variantes) du personnage
            entity1 = j
            for k in listeNomsPersonnages:
                for l in k:
                    if k != i and j!=l and (j not in k) and (l not in i):  # Assurer que ce sont deux personnages distincts : k != i -> pour éviter les relations entre une même personne, j!=l and (j not in k) and (l not in i)-> éviter une relations avec quelqu'un qui a le même nom de famille
                        entity2 = l
                        for pos1 in entity_positions[entity1]:
                            for pos2 in entity_positions[entity2]:
                                if (pos1 == pos2):
                                    if entity1 in i:
                                        if entity2 not in dictionnaireRelations[first_element]:
                                            dictionnaireRelations[first_element].append(entity2)
                                            sentiment = page_sentiment[pos1][0]  # Sentiment de la page
                                            for group in listeNomsPersonnages:
                                                if entity2 in group:
                                                    AjoutNoeudsSentimentGraphe(G, first_element, group[0], sentiment)
    AfficherRelations(dictionnaireRelations)
                                    
def AjoutNoeudsSentimentGraphe(G, source, target, sentiment):
    # Choisir la couleur en fonction de la relation
    if sentiment == 1:
        color = "green"
    elif sentiment == 0:
        color = "grey"
    else:
        color = "red"
    
    # Ajouter l'arête avec un attribut de couleur
    if G.has_edge(source, target):
        # Si l'arête existe déjà, on augmente le poids (ou on peut garder le même poids)
        G[source][target]['weight'] += 1
    else:
        # Sinon, créer une nouvelle arête avec poids 1 et la couleur associée
        G.add_edge(source, target, weight=1, color=color)
                                       
                                       
def AjoutNoeudsGraphe(G, listeNomsPersonnages):
    for group in listeNomsPersonnages: # Ajout des nœuds dans le graphe
        group = list(group)
        first_element = group[0]  # Premier élément comme nom principal
        remaining_elements = ";".join(group[1:])  # Joindre les variantes
        if first_element not in G.nodes:
            G.add_node(first_element)
        G.nodes[first_element]["names"] = f"{first_element};{remaining_elements}" if remaining_elements else first_element

    for node in G.nodes:# Ajout des attributs manquants pour les nœuds
        if "names" not in G.nodes[node]:
            G.nodes[node]["names"] = node

def CreerGraphe(G,book):
    net = Network(notebook=True)# Visualisation du graphe avec pyvis
    net.from_nx(G)
    net.show(f"graphes/{book}.html")

    # Sauvegarde du graphe dans le fichier CSV
    df_dict["ID"].append(book)
    graphml = "".join(nx.generate_graphml(G))
    df_dict["graphml"].append(graphml)

def AfficherVariantes(listeNomsPersonnages):
    print("/////////////////Liste de variantes de ce livre////////////////////")
    for i in listeNomsPersonnages:
        print(i)

def AfficherMotsRetirer(listeRetirer):
    print("/////////////////Liste d'éléments retirer de ce livre////////////////////")
    print(list(dict.fromkeys(listeRetirer)))

def AfficherRelations(dictionnaireRelations):
    print("/////////////////Liste de relation de ce livre////////////////////")
    print(dictionnaireRelations)

def AnalyseLivre(book,listeLieux):
    G = nx.Graph()
    listeNomsPersonnages = []
    listeRetirer = []
    page_sentiment = defaultdict(list)
    pages = DecoupageTexteEnPages(f"Textes_Processed/{book}.txt")
    listePersonnagesTrier = AnalyseEntitee(book,page_sentiment,pages,listeRetirer)

    Tri_Lieux(listePersonnagesTrier,listeLieux,listeRetirer)### Tri des entités détecté Per mais qui sont des lieux 
    Tri_Parasite(listePersonnagesTrier,listeRetirer)### Tri des personnages "Parasites" qui n'ont pas été enlever 
    Ranger_Par_Noms(listePersonnagesTrier,listeNomsPersonnages)#D'abord les noms de famille
    Supprimer_Sous_Liste(listeNomsPersonnages)# Supprimer les listes qui sont des sous-listes d'autres
    Ranger_Par_Prenoms(listeNomsPersonnages)#Puis rajouter les prénoms isoler dans une liste
    Supprimer_Sous_Liste(listeNomsPersonnages)# Supprimer les listes qui sont des sous-listes d'autres
    Separation_Alias(listeNomsPersonnages)# Pour séparer les alias avec les mêmes noms de famille, mais pas les mêmes prénoms
    listeNomsPersonnages = [variantesNoms for variantesNoms in listeNomsPersonnages if variantesNoms]#Pour nettoyer la liste des listes vides 
    Supprimer_Sous_Liste(listeNomsPersonnages)# Supprimer les listes qui sont des sous-listes d'autres
               
    for i in range(len(listeNomsPersonnages)):##Pour nettoyer les doublons apparu
        listeNomsPersonnages[i] = list(dict.fromkeys(listeNomsPersonnages[i])) 

    print("/////////////////",book,"////////////////////")
    AfficherVariantes(listeNomsPersonnages)
    AfficherMotsRetirer(listeRetirer)
    
    entity_positions = Association_Position_personnage(pages,listePersonnagesTrier)
    Relations(G,listeNomsPersonnages,entity_positions,page_sentiment)
    AjoutNoeudsGraphe(G, listeNomsPersonnages)   
    CreerGraphe(G,book)

def ExportCVS():# Enregistrement des résultats dans un DataFrame et exportation en CSV
    df = pd.DataFrame(df_dict)
    df.set_index("ID", inplace=True)
    df.to_csv("./my_submission.csv")

def Extraction():
    listeLieux=AnalyseLieux()
    for book in books:
        AnalyseLivre(book,listeLieux)
    ExportCVS()

Extraction()