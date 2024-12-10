import stanza
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import networkx as nx
import pandas as pd
import re
from flair.data import Sentence
from flair.models import SequenceTagger

# Charger le modèle NER pour le français
tagger = SequenceTagger.load("flair/ner-french")

# Télécharger le modèle Stanza pour le français si nécessaire
stanza.download('fr')

# Initialisation du pipeline de traitement de texte pour le français
nlp = stanza.Pipeline('fr', processors='tokenize,mwt,pos,ner')

# Charger le modèle SBERT pour obtenir les embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Un modèle léger pour les comparaisons de similarité

### Liste des livres et chapitres à extraire        
books = [
    (list(range(1, 20)), "paf"),
    (list(range(1, 19)), "lca"),
]

df_dict = {"ID": [], "graphml": []}

lieuxASupprimer = []  # Liste pour stocker les éléments à supprimer
listeLieuxAll = []

# Extraction des lieux
for chapters, book_code in books:
    for chapter in chapters:
        if book_code == "paf":
            repertory = "prelude_a_fondation"
        else:
            repertory = "les_cavernes_d_acier"
        if chapter != 0:
            with open(f"reseaux-de-personnages-de-fondation-session-2/{repertory}/chapter_{chapter}.txt.preprocessed", "r") as file:
                texte = file.read()

            # Traitement du texte avec Stanza pour extraire les lieux
            sentence = Sentence(texte)
            tagger.predict(sentence)
            loc_entities = [entity.text for entity in sentence.get_spans('ner') if entity.get_label('ner').value == 'LOC']
            listeLieuxAll.extend(loc_entities)

# Analyse des personnages et des relations
for chapters, book_code in books:
    for chapter in chapters:
        G = nx.Graph()

        if book_code == "paf":
            repertory = "prelude_a_fondation"
        else:
            repertory = "les_cavernes_d_acier"

        with open(f"reseaux-de-personnages-de-fondation-session-2/{repertory}/chapter_{chapter}.txt.preprocessed", "r") as file:
            texte = file.read()

        # Traitement du texte pour l'extraction des entités
        doc = nlp(texte)
        
        # Filtrage des noms propres
        listePROPN = [word.text for sent in doc.sentences for word in sent.words if word.upos == "PROPN"]
        listeFiltre = [word.text for sent in doc.sentences for word in sent.words if word.upos not in ["PROPN", "NOUN"]]

        listePersonnages = []
        for ent in doc.ents:
            if ent.type == "PER" and len(ent.text) > 2 and ent.text not in listeFiltre:
                listePersonnages.append(ent.text)

        # Filtrer les personnages qui sont aussi des lieux
        listePersonnagesTrier = list(dict.fromkeys(listePersonnages))  # Liste unique

        lieux_set = set(listeLieuxAll)
        for personnage in listePersonnagesTrier[:]:
            if personnage in lieux_set:
                listePersonnagesTrier.remove(personnage)

        # Affichage des personnages trouvés
        print(f"Chapitre {chapter}, Livre {book_code}")
        print("Liste des Personnages:", listePersonnagesTrier)

        # Encoder les noms des personnages pour obtenir les embeddings
        embeddings = model.encode(listePersonnagesTrier)

        # Appliquer DBSCAN pour trouver des alias
        db = DBSCAN(eps=0.4, min_samples=1, metric="cosine").fit(embeddings)

        listeNomsPersonnages = []
        for cluster in set(db.labels_):
            if cluster != -1:  # Exclure les points "bruit"
                cluster_aliases = [listePersonnagesTrier[i] for i, label in enumerate(db.labels_) if label == cluster]
                listeNomsPersonnages.append(cluster_aliases)

        print("Alias détectés:", listeNomsPersonnages)

        # Créer un graphe des relations entre les personnages
        dictionnaireRelationsListe = {}
        for cluster in listeNomsPersonnages:
            first_element = list(cluster)[0]
            if first_element not in dictionnaireRelationsListe:
                dictionnaireRelationsListe[first_element] = []

            for other_cluster in listeNomsPersonnages:
                if other_cluster != cluster:
                    for element in other_cluster:
                        if element not in dictionnaireRelationsListe[first_element]:
                            dictionnaireRelationsListe[first_element].append(element)

        # Construire le graphe avec les relations
        for source, targets in dictionnaireRelationsListe.items():
            for target in targets:
                G.add_edge(source, target)

        # Ajouter les nœuds au graphe
        for group in listeNomsPersonnages:
            first_element = list(group)[0]
            if first_element not in G.nodes:
                G.add_node(first_element)
            G.nodes[first_element]["names"] = ";".join(group)

        # Sauvegarder le graphe en format GraphML
        df_dict["ID"].append(f"{book_code}{chapter-1}")
        graphml = "".join(nx.generate_graphml(G))
        df_dict["graphml"].append(graphml)

# Sauvegarder les résultats dans un fichier CSV
df = pd.DataFrame(df_dict)
df.set_index("ID", inplace=True)
df.to_csv("./my_submission.csv")
