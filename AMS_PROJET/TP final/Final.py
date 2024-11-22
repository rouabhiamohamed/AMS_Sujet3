import stanza
from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess
import networkx as nx
import pandas as pd
from collections import defaultdict

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

# Traitement des livres et chapitres
for chapters, book_code in books:
    for chapter in chapters:
    
        G = nx.Graph()
    
        # Définir le répertoire en fonction du code du livre
        if book_code == "paf":
            repertory = "prelude_a_fondation"
        else:
            repertory = "les_cavernes_d_acier"
            
        ###Si le chapitre est 0, alors on rempli vide le cvs 
        if chapter==0:

            df_dict["ID"].append("{}{}".format(book_code, chapter))

            graphml = "".join(nx.generate_graphml(G))
            df_dict["graphml"].append(graphml)
            
        else:
            ### Ouvrir le fichier correspondant au chapitre 'chapter' du livre 'book_code'
            with open(f"reseaux-de-personnages-de-fondation-session-2/{repertory}/chapter_{chapter}.txt.preprocessed", "r") as file:
                texte = file.read()

            # Traitement du texte
            doc = nlp(texte)

            ### Liste pour stocker les personnages extraits
            listePersonnages = []
            listeNomsPersonnages = []

            ### Extraction des entités nommées de type "PER" (personne)
            for ent in doc.ents:
                if ent.type == "PER":
                    ent.text = ent.text.replace("\n", " ").strip() ###Certaines entités ont des \n, on les enlève
                    listePersonnages.append(ent.text)

            # Liste pour stocker les personnages uniques après filtrage
            listePersonnagesTrier = set(listePersonnages)  # Utiliser un set pour obtenir les personnages uniques

            ### Bout de code pour vérifier et fusionner les listes de variantes de noms
            for nom in listePersonnagesTrier:
                variantesNoms = {nom}  # Utiliser un set pour éviter les doublons
                for autreNom in listePersonnagesTrier:
                    if autreNom != nom and (autreNom in nom or nom in autreNom) or (nom.upper()==autreNom or nom==autreNom.upper()):  # Vérifier si c'est une variante
                        variantesNoms.add(autreNom)  # Ajouter la variante au set 
                if variantesNoms not in listeNomsPersonnages:  # Vérifier si l'ensemble n'est pas déjà dans la liste
                    listeNomsPersonnages.append(variantesNoms)


            ###Bout de code qui permet d'enlever les redondances d'ensemble, si un ensemble possède des sous-ensemble dans la liste alors on les supprime
            for variantesNoms in listeNomsPersonnages:
                for s in listeNomsPersonnages: 
                    if (s.issubset(variantesNoms) and s != variantesNoms) :
                        listeNomsPersonnages.remove(s)
                    elif (variantesNoms.issubset(s) and s != variantesNoms):
                        listeNomsPersonnages.remove(variantesNoms)

                
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
            
            print(chapter, book_code)
            
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
                    G.add_edge(source, target)

            for group in listeNomsPersonnages:
                group = list(group)  # Convertir en liste pour un accès facile
                first_element = group[0]  # Prendre le premier élément comme nom principal
                remaining_elements = ";".join(group[1:])  # Joindre les variantes, s'il y en a

                # Ajouter le nœud avec ses attributs s'il n'est pas déjà présent
                if first_element not in G.nodes:
                    G.add_node(first_element)  # Ajouter un nœud isolé
                G.nodes[first_element]["names"] = f"{first_element};{remaining_elements}" if remaining_elements else first_element

            # Ajout des attributs manquants
            for node in G.nodes:
                if "names" not in G.nodes[node]:
                    G.nodes[node]["names"] = node
            
            
            df_dict["ID"].append("{}{}".format(book_code, chapter))

            graphml = "".join(nx.generate_graphml(G))
            df_dict["graphml"].append(graphml)
                    

df = pd.DataFrame(df_dict)
df.set_index("ID", inplace=True)
df.to_csv("./my_submission.csv")


