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


# Charger le modèle NER pour le français
tagger = SequenceTagger.load("flair/ner-french")

# Télécharger le modèle Stanza pour le français si nécessaire
stanza.download('fr')

# Initialisation du pipeline de traitement de texte pour le français
nlp = stanza.Pipeline('fr', processors='tokenize,mwt,pos,ner')

#Pour Vader
analyzer = SentimentIntensityAnalyzer()


###Liste des livres        
books = [
    "Fondation_et_empire_sample"
    #"Fondation_sample",
    #"Seconde_Fondation_sample",
    #"Terre_et_Fondation_sample",
    #"Fondation_foudroyée_sample"
]

df_dict = {"ID": [], "graphml": []}

def load_and_split_text(book_path):
    with open(book_path, "r") as file:
        texte = file.read()
    
    # Utilisation d'une expression régulière pour découper le texte en pages sur la base des délimiteurs <1>, <2>, etc.
    pages = re.split(r'<->', texte)
    return pages


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

def Relations(listeNomsPersonnages, entity_positions):
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
    return dictionnaireRelations
                                    

    
def add_edges_with_sentiment_relation(G, source, target, relation):
    # Choisir la couleur en fonction de la relation
    if relation == 1:
        color = "green"
    elif relation == 0:
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

for book in books:


    G = nx.Graph()
    listePersonnages = []
    listeRetirer = []
    listeFiltre = []
    tokens = []

    page_sentiment = defaultdict(list)

    print(f"////////////////////////////////////Livre {book}////////////////////////////////////")
    pages = load_and_split_text(f"Textes_Processed/{book}.txt")
    for i, page in enumerate(pages):

        # Traitement du texte
        doc = nlp(page)

        ##Bout de code pour enlever les faux noms détecter par le ner, grâce aux POS
        Filtre_Ner_Pos(doc,listeFiltre,tokens)

        ### Liste pour stocker les personnages extraits
        Extraction_Per(doc,listePersonnages,listeFiltre)
        
        # Liste pour stocker les personnages uniques après filtrage
        listePersonnagesTrier = list(dict.fromkeys(listePersonnages))  # Utiliser un set pour obtenir les personnages uniques

        page_sentiment[i].append(Sentiment_page(page))
    
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
    
    entity_positions = Association_Position_personnage(pages,listePersonnagesTrier)
    
    dictionnaireRelations = Relations(listeNomsPersonnages,entity_positions)

    print("/////////////////////////")
    print(dictionnaireRelations)
    print("/////////////////////////")

    # Remplissage du graphe avec les relations
    for source, targets in dictionnaireRelations.items():
        for target in targets:
            for group in listeNomsPersonnages:
                if target in group:
                    sentiment_color = 'grey'  # Par défaut, couleur neutre
                    for pos1 in entity_positions[source]:
                        for pos2 in entity_positions[target]:
                            if pos1 == pos2:  # Vérification de la même page
                                sentiment = page_sentiment[pos1][0]  # Sentiment de la page
                                if sentiment == 1:  # Sentiment positif
                                    sentiment_color = 'green'
                                elif sentiment == -1:  # Sentiment négatif
                                    sentiment_color = 'red'

                                if G.has_edge(source, group[0]):
                                    G[source][group[0]]['weight'] += 1
                                else:
                                    G.add_edge(source, group[0], weight=1, color=sentiment_color)

    # Ajout des nœuds dans le graphe
    for group in listeNomsPersonnages:
        group = list(group)
        first_element = group[0]  # Premier élément comme nom principal
        remaining_elements = ";".join(group[1:])  # Joindre les variantes

        if first_element not in G.nodes:
            G.add_node(first_element)

        G.nodes[first_element]["names"] = f"{first_element};{remaining_elements}" if remaining_elements else first_element

    # Ajout des attributs manquants pour les nœuds
    for node in G.nodes:
        if "names" not in G.nodes[node]:
            G.nodes[node]["names"] = node
        
    # Visualisation du graphe avec pyvis
    net = Network(notebook=True)
    net.from_nx(G)

    net.show(f"graphes/{book}.html")
    
    # Sauvegarde du graphe dans le fichier CSV
    df_dict["ID"].append(book)
    graphml = "".join(nx.generate_graphml(G))
    df_dict["graphml"].append(graphml)

# Enregistrement des résultats dans un DataFrame et exportation en CSV
df = pd.DataFrame(df_dict)
df.set_index("ID", inplace=True)
df.to_csv("./my_submission.csv")