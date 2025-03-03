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
    "Fondation_et_empire_sample",
    "Fondation_sample",
    "Seconde_Fondation_sample",
    "Terre_et_Fondation_sample",
    "Fondation_foudroyée_sample"
]

df_dict = {"ID": [], "graphml": []}

"""
    Découpe un fichier de livre en pages en utilisant un délimiteur spécifique <->.

    Arguments:
    - book_path (str): Le chemin d'accès au fichier texte du livre.

    Retour:
    - pages (list): Une liste de pages découpées à partir du texte du livre.
"""
def DecoupageTexteEnPages(book_path):
    with open(book_path, "r") as file:
        texte = file.read()
    pages = re.split(r'<->', texte)
    return pages

"""
    Identifie et extrait tous les lieux (LOC) dans un livre donné en utilisant le modèle NER de Flair.
    
    Arguments:
    book (str): Le nom du livre à analyser (sans l'extension '.txt').
    
    Retour:
    listeLieux (list): Liste des lieux (LOC) extraits du livre.
"""
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

"""
    Filtre les entités nommées incorrectes en fonction des étiquettes POS pour ne garder que les entités valides.

    Arguments:
    - doc (stanza.Doc): L'objet Doc de Stanza contenant les informations syntaxiques du texte.
    - listeFiltre (list): Liste des mots filtrés qui ne sont pas considérés comme des entités valides.

    Retour:
    - listeFiltre (list): Liste mise à jour avec les mots considérés comme invalides.
"""
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

"""
    Extrait les entités de type "personne" (PER) et les ajoute à la liste des personnages,
    tout en évitant les faux noms filtrés.

    Arguments:
    - doc (stanza.Doc): L'objet Doc de Stanza contenant les entités nommées détectées.
    - listePersonnages (list): Liste dans laquelle les personnages extraits seront ajoutés.
    - listeFiltre (list): Liste des mots à ne pas ajouter à la liste des personnages.
    - listeRetirer (list): Liste dans laquelle les entités à retirer seront ajoutés (par exemple, les lieux).

    Retour:
    - listePersonnages (list): Liste mise à jour avec les nouveaux personnages extraits.
    - listeRetirer (list): Liste mise à jour avec les entités à retirer.
"""
def Extraction_Per(doc,listePersonnages,listeFiltre,listeRetirer):
    for ent in doc.ents:
        if ent.type == "PER" and len(ent.text) > 2:
            ent.text = ent.text.replace("\n", " ").strip()
            if ent.text not in listeFiltre:
                listePersonnages.append(ent.text)
            else:
                listeRetirer.append(ent.text)

"""
    Analyse le sentiment d'une page (positif, négatif ou neutre) en utilisant l'analyseur de sentiment de Vader.
    
    Arguments:
    page (str): Le texte de la page à analyser.
    
    Retour:
    int: Le score de sentiment (-1 pour négatif, 0 pour neutre, 1 pour positif).
"""
def Sentiment_page(page):
    sentiment_score = analyzer.polarity_scores(page)
    if sentiment_score['compound'] > 0:
        return 1
    elif sentiment_score['compound'] < 0:
        return -1
    else:
        return 0

"""
    Extrait les entités nommées et leur sentiment pour chaque page d'un livre, en filtrant et classant les personnages.
    
    Arguments:
    book (str): Le nom du livre à analyser.
    page_sentiment (dict): Dictionnaire qui contiendra les scores de sentiment par page.
    pages (list): Liste des pages découpées du livre.
    listeRetirer (list): Liste des entités à ignorer (par exemple, des parasites ou des faux personnages).
    
    Retour:
    listePersonnagesTrier (list): Liste des personnages extraits et filtrés du livre.
"""
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

"""
    Tri les personnages qui sont en fait des lieux, en les retirant de la liste des personnages valides.

    Arguments:
    - listePersonnagesTrier (list): Liste des personnages après le filtrage initial.
    - listeLieuxAll (list): Liste de tous les lieux extraits du texte.
    - listeRetirer (list): Liste des entités à retirer.

    Retour:
    - listePersonnagesTrier (list): Liste mise à jour des personnages, après retrait des lieux.
    - listeRetirer (list): Liste mise à jour des entités à retirer.
"""
def Tri_Lieux(listePersonnagesTrier,listeLieuxAll,listeRetirer):
    lieux_set = set(listeLieuxAll)
    for personnage in listePersonnagesTrier[:]: # Créer une copie de listePersonnagesTrier pour itérer sans modifier la liste pendant la boucle
        if personnage in lieux_set: # Si le personnage est un lieu détecté, on le retire de la liste
            listePersonnagesTrier.remove(personnage)
            listeRetirer.append(personnage)

"""
    Tri les personnages parasites en fonction des vérifications grammaticales et les retire si nécessaire.

    Arguments:
    - listePersonnagesTrier (list): Liste des personnages après filtrage initial.
    - listeRetirer (list): Liste des entités à retirer.

    Retour:
    - listePersonnagesTrier (list): Liste mise à jour après retrait des personnages parasites.
    - listeRetirer (list): Liste mise à jour des entités à retirer.
"""
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

"""
    Regroupe les personnages par nom de famille, créant des groupes de variantes de noms similaires.

    Arguments:
    - listePersonnagesTrier (list): Liste des personnages après filtrage initial.
    - listeNomsPersonnages (list): Liste dans laquelle les variantes de noms seront regroupées.

    Retour:
    - listeNomsPersonnages (list): Liste mise à jour avec les groupes de variantes de noms.
"""
def Ranger_Par_Noms(listePersonnagesTrier, listeNomsPersonnages):                                    
    for nom in listePersonnagesTrier:
        variantesNoms = [nom] # Créer une liste de variantes à partir du nom actuel
        for autreNom in listePersonnagesTrier:
            if nom != autreNom and len(nom.split()) >= 2 and any(part in autreNom for part in nom.split()[-1:]): # Vérifier si les noms sont des variantes (partage d'une partie du dernier nom)
                    if autreNom not in variantesNoms:
                        variantesNoms.append(autreNom)
        
        if not any(set(variantesNoms) == set(existing) for existing in listeNomsPersonnages): # Ajouter la liste de variantes si elle n'est pas déjà présente dans `listeNomsPersonnages`
            listeNomsPersonnages.append(variantesNoms)

"""
    Trie les personnages selon leurs prénoms et les ajoute aux groupes de noms correspondants.

    Arguments:
    - listeNomsPersonnages (list): Liste des groupes de personnages regroupés par nom de famille.

    Retour:
    - listeNomsPersonnages (list): Liste mise à jour avec les prénoms ajoutés aux groupes.
""" 
def Ranger_Par_Prenoms(listeNomsPersonnages):
    for index, variantesNoms in enumerate(listeNomsPersonnages):
        if len(variantesNoms) > 1: # Vérifier si le personnage a plus d'une variante de nom
            for variantesNomsAAjouter in [v for v in listeNomsPersonnages if len(v) == 1]: # Parcourir la liste des autres variantes
                if any(variantesNomsAAjouter[0] in nom for nom in variantesNoms): # Si le prénom à ajouter est une sous-partie d'un des noms de variantes
                    listeNomsPersonnages[index].append(variantesNomsAAjouter[0])
    
"""
    Sépare les alias (différents prénoms mais mêmes noms de famille) pour éviter les doublons.

    Arguments:
    - listeNomsPersonnages (list): Liste des groupes de personnages regroupés par nom de famille.

    Retour:
    - listeNomsPersonnages (list): Liste mise à jour sans les alias doublons.
"""
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
                            docPrenom2 = nlp(personnageAComparer)
                            tokenPrenom1 = [word.upos for sent in docPrenom1.sentences for word in sent.words]
                            tokenPrenom2 = [word.upos for sent in docPrenom2.sentences for word in sent.words]
                            if personnage.split()[0] != personnageAComparer.split()[0] and \
                                ("PROPN" in tokenPrenom1[0] and "PROPN" in tokenPrenom2[0]):
                                listeDeNoms.append(personnageAComparer)
                                nomDeFamille = personnageAComparer.split()[-1]
                                fin=False
                                
                for personnageARajouter in variantesNoms:# Ajouter des prénoms à la listeDeNomsSimple si ce sont des sous-ensembles d'un nom existant
                    if len(personnageARajouter.split()) == 1 and personnageARajouter not in listeDeNoms:
                        if any(personnageARajouter in nomAComparer for nomAComparer in listeDeNoms):
                            listeDeNomsSimple.append(personnageARajouter)
                listeDeNomsConcatenee = listeDeNoms + listeDeNomsSimple
                
                for elem in listeDeNomsConcatenee:# Enlever les éléments de listeDeNomsConcatenee de la liste originale
                    if elem in variantesNoms and elem != nomDeFamille:
                        variantesNoms.remove(elem)
                        
                listeNomsPersonnages.append(listeDeNomsConcatenee)
    
"""
    Supprime les sous-listes dans listeNomsPersonnages qui sont vides.

    Arguments:
    - listeNomsPersonnages (list): Liste des groupes de personnages regroupés par nom de famille.

    Retour:
    - listeNomsPersonnages (list): Liste mise à jour avec les sous-listes non vides.
"""
def Supprimer_Sous_Liste(listeNomsPersonnages):
    for variantesNoms in listeNomsPersonnages[:]:
        for s in listeNomsPersonnages[:]:
            if variantesNoms != s and all(nom in s for nom in variantesNoms):  # Si variantesNoms est un sous-ensemble de s
                listeNomsPersonnages.remove(variantesNoms)
                break

"""
    Associe les positions des personnages aux pages où ils apparaissent dans le texte.
    
    Arguments:
    pages (list): Liste des pages du livre.
    listePersonnagesTrier (list): Liste des personnages extraits du livre.
    
    Retour:
    entity_positions (defaultdict): Dictionnaire associant chaque personnage à une liste de positions de pages où il apparaît.
"""
def Association_Position_personnage(pages,listePersonnagesTrier):
    entity_positions = defaultdict(list)
    for i, page in enumerate(pages):
        for group in listePersonnagesTrier:
            if(group in page):
                entity_positions[group].append(i)
    return entity_positions

"""
    Ajoute des arêtes au graphe G avec un poids représentant la force de la relation entre deux personnages.
    
    Arguments:
    G (networkx.Graph): Le graphe dans lequel l'arête est ajoutée.
    source (str): Le personnage source de la relation.
    target (str): Le personnage cible de la relation.
    
    Retour:
    None: La fonction modifie directement le graphe passé en argument.
"""                                   
def AjoutNoeudsPoidGraphe(G, source, target):
    if G.has_edge(source, target): # Ajouter l'arête avec un attribut de couleur
        G[source][target]['weight'] += 1 # Si l'arête existe déjà, on augmente le poids (ou on peut garder le même poids)
    else:
        G.add_edge(source, target, weight=1)# Sinon, créer une nouvelle arête avec poids 1 

"""
    Assigne une couleur aux arêtes du graphe en fonction du sentiment associé à la relation.
    
    Arguments:
    G (networkx.Graph): Le graphe contenant l'arête à colorier.
    source (str): Le personnage source de la relation.
    target (str): Le personnage cible de la relation.
    sentiment (int): Le sentiment associé à la relation (1 pour positif, -1 pour négatif, 0 pour neutre).
    
    Retour:
    None: La fonction modifie directement les propriétés des arêtes dans le graphe.
"""
def AjoutCouleurArrete(G, source, target,sentiment):
    if sentiment > 0:
        color = "green"
    elif sentiment == 0:
        color = "grey"
    else:
        color = "red"
    G[source][target]['color'] = color

"""
    Crée des relations entre les personnages en fonction de leur cooccurrence sur les pages et leur sentiment, puis construit un graphe des relations.
    
    Arguments:
    G (networkx.Graph): Le graphe de relations entre les personnages.
    listeNomsPersonnages (list): Liste des noms de personnages.
    entity_positions (dict): Dictionnaire des positions des personnages dans le texte.
    page_sentiment (dict): Dictionnaire des sentiments par page.
    
    Retour:
    dictionnairePopularite (dict): Dictionnaire avec la popularité des personnages basée sur leur sentiment.
"""
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
            
        for entity1 in i:  # Parcourt chaque alias (variantes) du personnage
            for k in listeNomsPersonnages:
                for entity2 in k:
                    if k != i and entity1!=entity2 and (entity1 not in k) and (entity2 not in i):  # Assurer que ce sont deux personnages distincts : k != i -> pour éviter les relations entre une même personne, j!=l and (j not in k) and (l not in i)-> éviter une relations avec quelqu'un qui a le même nom de famille
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

"""
    Ajoute des nœuds pour chaque personnage et leurs variantes dans le graphe G.
    
    Arguments:
    G (networkx.Graph): Le graphe dans lequel les nœuds sont ajoutés.
    listeNomsPersonnages (list): Liste des personnages et leurs alias à ajouter en tant que nœuds.
    
    Retour:
    None: La fonction modifie directement le graphe passé en argument en y ajoutant des nœuds.
"""
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

"""
    Détecte les cliques dans le graphe (groupes de personnages fortement connectés).
    
    Arguments:
    G (networkx.Graph): Le graphe des relations entre les personnages.
    
    Retour:
    cliques (list): Liste des cliques détectées dans le graphe. Chaque clique est une liste de nœuds.
"""
def DetecterCliques(G):
    cliques = list(nx.algorithms.clique.find_cliques(G))# Utilisation de l'algorithme de Bron–Kerbosch pour détecter les cliques
    print("Cliques détectées:")# Affichage des cliques trouvées
    for clique in cliques:
        print(clique)    
    return cliques

"""
    Crée un sous-graphe pour chaque clique et l'affiche dans un fichier HTML.
    
    Arguments:
    G (networkx.Graph): Le graphe contenant toutes les relations.
    cliques (list): Liste des cliques détectées dans le graphe.
    book (str): Le nom du livre, utilisé pour nommer les fichiers générés.
    
    Retour:
    None: La fonction génère un fichier HTML pour chaque clique sous forme de sous-graphe.
"""
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
        
"""
    Crée une visualisation du graphe complet des personnages et de leurs relations, puis l'affiche dans un fichier HTML.
    
    Arguments:
    G (networkx.Graph): Le graphe des relations entre les personnages.
    book (str): Le nom du livre, utilisé pour nommer le fichier HTML généré.
    
    Retour:
    None: La fonction génère un fichier HTML contenant le graphe complet.
"""
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

"""
    Affiche toutes les variantes de noms des personnages dans le livre.
    
    Arguments:
    listeNomsPersonnages (list): Liste des personnages et de leurs alias.
    
    Retour:
    None: La fonction affiche les variantes de noms des personnages.
"""
def AfficherVariantes(listeNomsPersonnages):
    print("/////////////////Liste de variantes de ce livre////////////////////")
    for i in listeNomsPersonnages:
        print(i)

"""
    Affiche tous les mots ou entités retirés du livre.
    
    Arguments:
    listeRetirer (list): Liste des entités retirées lors de l'analyse.
    
    Retour:
    None: La fonction affiche les éléments retirés du livre.
"""
def AfficherMotsRetirer(listeRetirer):
    print("/////////////////Liste d'éléments retirer de ce livre////////////////////")
    print(list(dict.fromkeys(listeRetirer)))

"""
    Affiche toutes les relations détectées entre les personnages.
    
    Arguments:
    dictionnaireRelations (dict): Dictionnaire des relations entre les personnages.
    
    Retour:
    None: La fonction affiche les relations entre les personnages.
"""
def AfficherRelations(dictionnaireRelations):
    print("/////////////////Liste de relation de ce livre////////////////////")
    print(dictionnaireRelations)

"""
    Crée un classement des personnages en fonction de leur popularité, basée sur le sentiment, et l'enregistre dans un fichier.
    
    Arguments:
    dictionnairePopularite (dict): Dictionnaire des popularités des personnages.
    book (str): Le nom du livre, utilisé pour nommer le fichier de classement généré.
    
    Retour:
    None: La fonction génère un fichier texte contenant le classement des personnages.
"""
def CreerClassementPopularite(dictionnairePopularite,book):
    rank = 1
    dictionnaireTrier = dict(sorted(dictionnairePopularite.items(), key=lambda item: item[1], reverse=True))
    with open(f"Ranking/Ranking_({book}).txt", "w") as file:
        file.write("////////////Ranking////////////\n")
        for nom, nbPop in dictionnaireTrier.items():
            file.write(f"{rank} - {nom} - Popularité : {nbPop}\n") 
            rank +=1
    
"""
    Effectue l'analyse complète d'un livre, y compris l'extraction des entités, la gestion des relations, et la création du graphe de relations.
    
    Arguments:
    book (str): Le nom du livre à analyser.
    listeLieux (list): Liste des lieux extraits du livre.
    
    Retour:
    None: La fonction effectue l'analyse et génère les résultats.
"""
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

"""
    Exporte les résultats sous forme de CSV.

    Arguments:
    Aucun.
    
    Retour:
    Aucun. La fonction génère et enregistre un fichier CSV dans le répertoire courant.
"""
def ExportCVS():
    df = pd.DataFrame(df_dict)
    df.set_index("ID", inplace=True)
    df.to_csv("./my_submission.csv")

"""
    Effectue le processus d'analyse sur tous les livres listés et enregistre les résultats.

    Arguments:
    Aucun. La liste des livres est définie plus haut dans le code sous le nom `books`.
    
    Retour:
    Aucun. La fonction effectue le processus d'analyse et enregistre les résultats dans un fichier CSV.
"""
def Extraction():
    for book in books:
        listeLieux=AnalyseLieux(book)
        AnalyseLivre(book,listeLieux)
    ExportCVS()

Extraction()