import stanza
from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess
import networkx as nx
import pandas as pd
from collections import defaultdict
import re
from flair.data import Sentence
from flair.models import SequenceTagger

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

# Charger le modèle NER pour le français
tagger = SequenceTagger.load("flair/ner-french")

# Télécharger le modèle Stanza pour le français si nécessaire
stanza.download('fr')

# Initialisation du pipeline de traitement de texte pour le français
nlp = stanza.Pipeline('fr', processors='tokenize,mwt,pos,ner')
        
# Charger le modèle SBERT pour obtenir les embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Un modèle léger pour les comparaisons de similarité

###Liste des livres et chapitres à extraire        
books = [
    (list(range(1, 20)), "paf"),
    (list(range(1, 19)), "lca"),
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
    
        G = nx.Graph()
        
        
        if book_code == "paf":
            repertory = "prelude_a_fondation"
        else:
            repertory = "les_cavernes_d_acier"

        # if chapter==0:

            # df_dict["ID"].append("{}{}".format(book_code, chapter))

            # graphml = "".join(nx.generate_graphml(G))
            # df_dict["graphml"].append(graphml)
        # else:
        
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
                
               
        print(chapter, book_code)
        print("////////////////Liste de Noms/////////////////")
        print(listePersonnagesTrier)
        print("////////////////Liste de mots retirer/////////////////")
        print(list(dict.fromkeys(listeRetirer)))
        # print("////////////////Liste de lieux/////////////////")
        # print(listeLieux)
        print("\n")
                        
                        
                        
                        
        embeddings = model.encode(listePersonnagesTrier)                
                        
        listeNomsPersonnages = []
            
        db = DBSCAN(eps=0.4, min_samples=1, metric="cosine").fit(embeddings)
        # Afficher les clusters trouvés
        for cluster in set(db.labels_):
            if cluster != -1:  # Exclure les points "bruit" (cluster -1 dans DBSCAN)
                cluster_aliases = [
                    listePersonnagesTrier[i] for i, label in enumerate(db.labels_) if label == cluster
                ]
                listeNomsPersonnages.append(cluster_aliases)
        
        
        ###Bout de code qui permet d'entrainer gensim avec le chapitre en cours de traitement   
        with open(f"reseaux-de-personnages-de-fondation-session-2/{repertory}/chapter_{chapter}.txt.preprocessed", "r") as file:
            documents = file.readlines()
        processed_docs = [simple_preprocess(doc) for doc in documents]
        dictionary = corpora.Dictionary(processed_docs)
        corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
        lsi = models.LsiModel(corpus, num_topics=200)
        index = similarities.MatrixSimilarity(lsi[corpus])

        ###Dictionnaire qui contiendra la liste de toutes les relations entre toutes les entités
        dictionnaireRelationsUnique = {}
        
        ###Bout de code qui permet d'établir les relations entre deux entités et les stock dans un dictionnaire
        ### key : EntitéSource, value : EntitéTarget
        for i in listeNomsPersonnages:
            for j in i :
                query_nom1 = j
                for k in listeNomsPersonnages:
                    for l in k :
                        if(k!=i):
                            query_nom2 = l
                            query_bow_nom1 = dictionary.doc2bow(simple_preprocess(query_nom1))
                            query_lsi_nom1 = lsi[query_bow_nom1]
                            sims_nom1 = index[query_lsi_nom1]

                            query_bow_nom2 = dictionary.doc2bow(simple_preprocess(query_nom2))
                            query_lsi_nom2 = lsi[query_bow_nom2]
                            sims_nom2 = index[query_lsi_nom2]

                            for doc_id, sim_nom1 in enumerate(sims_nom1):
                                sim_nom2 = sims_nom2[doc_id]
                                if sim_nom1 > 0.09 and sim_nom2 > 0.09:
                                    dictionnaireRelationsUnique[query_nom1]=query_nom2
                                    #print(f"Document {doc_id} : {query_nom1} et {query_nom2} ont une relation dans ce passage.")
        
        
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

        print(dictionnaireRelationsListe)
        
        ###Bout de code qui rempli le graphe en ajoutant les noeuds et les arrêtes via dictionnaireRelationsListe
        for source, targets in dictionnaireRelationsListe.items():
            for target in targets:
                for group in listeNomsPersonnages:
                    if target in group: 
                        print("//////////Relations////////////")
                        print(source, group[0])
                        G.add_edge(source, group[0])

        for group in listeNomsPersonnages:
            group = list(group)  # Convertir en liste pour un accès facile
            first_element = group[0]  # Prendre le premier élément comme nom principal
            remaining_elements = ";".join(group[1:])  # Joindre les variantes, s'il y en a

            # Ajouter le nœud avec ses attributs s'il n'est pas déjà présent
            if first_element not in G.nodes :
                print("//////////////Gnodes////////////////////")
                print(first_element)
                G.add_node(first_element)  # Ajouter un nœud isolé
            G.nodes[first_element]["names"] = f"{first_element};{remaining_elements}" if remaining_elements else first_element

        # Ajout des attributs manquants
        for node in G.nodes:
            if "names" not in G.nodes[node]:
                G.nodes[node]["names"] = node
        
        
        df_dict["ID"].append("{}{}".format(book_code, chapter-1))
        graphml = "".join(nx.generate_graphml(G))
        df_dict["graphml"].append(graphml)
                    

df = pd.DataFrame(df_dict)
df.set_index("ID", inplace=True)
df.to_csv("./my_submission.csv")