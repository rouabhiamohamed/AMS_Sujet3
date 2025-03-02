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
            if word.upos == "PROPN":
                listePROPN.append(word.text)
            elif word.upos in ["VERB", "AUX", "INTJ", "ADJ", "ADP", "ADV", "X", "PRON", "NOUN","NUM"] or \
                (word.upos == "NOUN" and word.feats and "Number=Plur" in word.feats):
                listeFiltre.append(word.text)

    for wordPROPN in listePROPN :
        if(wordPROPN in listeFiltre):
            listeFiltre.remove(wordPROPN)

### Extraction des entités nommées de type "PER" (personne)
def Extraction_Per(doc,listePersonnages,listeFiltre,listeRetirer):
    for ent in doc.ents:
        if ent.type == "PER" and len(ent.text) > 2:
            ent.text = ent.text.replace("\n", " ").strip()
            if ent.text not in listeFiltre:
                listePersonnages.append(ent.text)
            else:
                listeRetirer.append(ent.text)

### Tri des entités détecté Per mais qui sont des lieux 
def Tri_Lieux(listePersonnagesTrier,listeLieuxAll,listeRetirer):
    lieux_set = set(listeLieuxAll)
    for personnage in listePersonnagesTrier[:]: # Créer une copie de listePersonnagesTrier pour itérer sans modifier la liste pendant la boucle
        if personnage in lieux_set: # Si le personnage est un lieu détecté, on le retire de la liste
            listePersonnagesTrier.remove(personnage)
            listeRetirer.append(personnage)

### Tri des personnages "Parasites" qui n'ont pas été enlever 
def Tri_Parasite(listePersonnagesTrier,listeRetirer): 
    to_remove = []
    for personnage in listePersonnagesTrier:
        docVerif = nlp(personnage)
        for sent in docVerif.sentences:
            for word in sent.words:
                if word.upos in ["INTJ", "DET", "VERB", "ADP"] or \
                    (word.upos == "NOUN" and word.feats and "Number=Plur" in word.feats) or \
                    (len(personnage.split()) == 1 and word.upos == "X"):
                    to_remove.append(personnage)
                    break
    for p in to_remove:
        listePersonnagesTrier.remove(p)
        listeRetirer.append(p)

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
    for variantesNoms in listeNomsPersonnages[:]:
        for s in listeNomsPersonnages[:]:
            if variantesNoms != s and all(nom in s for nom in variantesNoms):  # Si variantesNoms est un sous-ensemble de s
                listeNomsPersonnages.remove(variantesNoms)
                break

##Bout de code qui permet de mettre dans une liste tout les lieux du Corpus grâce à Flair
def AnalyseLieux(book):
    print(f"//////////////////Répertorisation de tout les Lieux dans {book} : En cours... /////////////////////")
    listeLieux = [] #Liste de tout les lieux de l'entièreté du corpus
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
                
    print(f"//////////////////Répertorisation de tout les Lieux dans {book} : Fait !/////////////////////")
    return listeLieux

def AnalyseEntitee(book,page_sentiment,pages,listeRetirer):
    print(f"//////////////////Répertorisation de toutes les Entitées dans {book} : En cours... /////////////////////")
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
    dictionnairePopularite = {}
    dictionnaireDegree = {}

    # Parcourir les noms de chaque personnage
    for i in listeNomsPersonnages:
        first_element = list(i)[0] if isinstance(i, set) else i[0]
        if first_element not in dictionnaireRelations:
            dictionnaireRelations[first_element] = []
            dictionnaireDegree[first_element] = {}
            
        for j in i:  # Parcourt chaque alias (variantes) du personnage
            entity1 = j
            for k in listeNomsPersonnages:
                for l in k:
                    if k != i and j!=l and (j not in k) and (l not in i):  # Assurer que ce sont deux personnages distincts : k != i -> pour éviter les relations entre une même personne, j!=l and (j not in k) and (l not in i)-> éviter une relations avec quelqu'un qui a le même nom de famille
                        entity2 = l
                        degree = 0  # Initialisation du degree pour cette paire spécifique
                        arrete = False
                        for pos1 in entity_positions[entity1]:
                            for pos2 in entity_positions[entity2]:
                                if (pos1 == pos2):
                                    arrete=True
                                    if entity2 not in dictionnaireRelations[first_element]:
                                        dictionnaireRelations[first_element].append(entity2)
                                        sentiment = page_sentiment[pos1][0]  # Sentiment de la page
                                        
                                        AjoutNoeudsPoidGraphe(G, first_element, k[0])
                                        degree += sentiment  # Accumule le degree pour cette relation

                                        # Mise à jour du dictionnaire de popularité
                                        if first_element not in dictionnairePopularite:
                                            dictionnairePopularite[first_element] = 0
                                        if sentiment == 1:
                                            dictionnairePopularite[first_element] += 1
                                        elif sentiment == -1:
                                            dictionnairePopularite[first_element] -= 1
                        # Si une relation a été trouvée (arrete=True), ajouter la couleur à l'arête
                        if arrete:
                            # Fusionner les degrés pour ce couple de personnages
                            if k[0] not in dictionnaireDegree[first_element]:
                                dictionnaireDegree[first_element][k[0]] = 0
                            dictionnaireDegree[first_element][k[0]] += degree
                            
                            # Ajouter l'arête si elle n'existe pas encore (on fait à cause des noms de famille en double)
                            if not G.has_edge(first_element, k[0]):
                                G.add_edge(first_element, k[0], weight=1)

                            # Ajout de la couleur à l'arête pour l'affichage
                            AjoutCouleurArrete(G, first_element, k[0], dictionnaireDegree[first_element][k[0]])

    AfficherRelations(dictionnaireRelations)
    return dictionnairePopularite
                                    
def AjoutNoeudsPoidGraphe(G, source, target):
    if G.has_edge(source, target): # Ajouter l'arête avec un attribut de couleur
        G[source][target]['weight'] += 1 # Si l'arête existe déjà, on augmente le poids (ou on peut garder le même poids)
    else:
        G.add_edge(source, target, weight=1)# Sinon, créer une nouvelle arête avec poids 1 

def AjoutCouleurArrete(G, source, target,sentiment):
    if sentiment > 0:
        color = "green"
    elif sentiment == 0:
        color = "grey"
    else:
        color = "red"
    G[source][target]['color'] = color

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


def DetecterCliques(G):
    cliques = list(nx.algorithms.clique.find_cliques(G))# Utilisation de l'algorithme de Bron–Kerbosch pour détecter les cliques
    print("Cliques détectées:")# Affichage des cliques trouvées
    for clique in cliques:
        print(clique)    
    return cliques

def CreerSousGrapheParClique(G, cliques,book):
    for clique in cliques:# Créer un sous-graphe pour chaque clique détectée
        subgraph = G.subgraph(clique)# Créer un sous-graphe avec uniquement les nœuds de la clique
        net = Network(notebook=True,cdn_resources='in_line')# Créer une visualisation du sous-graphe
        net.from_nx(subgraph)
        net.toggle_physics(True)
        for node in subgraph.nodes:
            names = subgraph.nodes[node].get("names", node)
            net.get_node(node)["title"] = f"Alias de {node}: {names}"
        net.show(f"graphes/cliques/{book}_clique_{'_'.join(clique)}.html")
        

def CreerGraphe(G,book):
    net = Network(notebook=True,cdn_resources='in_line')# Visualisation du graphe avec pyvis
    net.from_nx(G)
    net.toggle_physics(True)
    
    for node in G.nodes:    # affichage lors du clic sur un nœud
        names = G.nodes[node].get("names", node)# Récupére les noms associés au nœud
        net.get_node(node)["title"] = f"Alias de {node}: {names}"#affiche les noms associés au nœud

    net.show(f"graphes/{book}.html")

    # Sauvegarde du graphe dans le fichier CSV
    df_dict["ID"].append(book)
    graphml = "".join(nx.generate_graphml(G))
    df_dict["graphml"].append(graphml)

    cliques = DetecterCliques(G)
    CreerSousGrapheParClique(G, cliques,book)

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

def CreerClassementPopularite(dictionnairePopularite,book):
    rank = 1
    dictionnaireTrier = dict(sorted(dictionnairePopularite.items(), key=lambda item: item[1], reverse=True))
    with open(f"Ranking/Ranking_({book}).txt", "w") as file:
        file.write("////////////Ranking////////////\n")
        for nom, nbPop in dictionnaireTrier.items():
            file.write(f"{rank} - {nom} - Popularité : {nbPop}\n") 
            rank +=1
    

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

    AfficherVariantes(listeNomsPersonnages)
    AfficherMotsRetirer(listeRetirer)
    
    entity_positions = Association_Position_personnage(pages,listePersonnagesTrier)
    dictionnairePopularite = Relations(G,listeNomsPersonnages,entity_positions,page_sentiment)
    CreerClassementPopularite(dictionnairePopularite,book)
    AjoutNoeudsGraphe(G, listeNomsPersonnages)   
    CreerGraphe(G,book)
    
    print(f"//////////////////Répertorisation de toutes les Entitées dans {book} : Fait !/////////////////////")

def ExportCVS():# Enregistrement des résultats dans un DataFrame et exportation en CSV
    df = pd.DataFrame(df_dict)
    df.set_index("ID", inplace=True)
    df.to_csv("./my_submission.csv")

def Extraction():
    for book in books:
        listeLieux=AnalyseLieux(book)
        AnalyseLivre(book,listeLieux)
    ExportCVS()

Extraction()