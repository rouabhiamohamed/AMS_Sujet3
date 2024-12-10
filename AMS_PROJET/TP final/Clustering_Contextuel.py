from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
import networkx as nx
import re


# 1. Chargement du modèle
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

###Liste des livres et chapitres à extraire        
books = [
    (list(range(0, 19)), "paf"),
    (list(range(0, 18)), "lca"),
]

# Charger le modèle SBERT pour obtenir les embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Un modèle léger pour les comparaisons de similarité

for chapters, book_code in books:
    for chapter in chapters:
        if book_code == "paf":
            repertory = "prelude_a_fondation"
        else:
            repertory = "les_cavernes_d_acier"
        if chapter!=0:
            
            print("/////////////////",book_code,chapter,"////////////////////")
            
            # 2. Exemple de texte et détection des noms
            with open(f"reseaux-de-personnages-de-fondation-session-2/{repertory}/chapter_{chapter}.txt.preprocessed", "r") as file:
                texte = file.read()
        
            with open(f"Partie2/{repertory}_chapter_{chapter}_characters.txt", "r") as file:
                noms_extraits = [line.strip() for line in file if line.strip()]

            

            # 3. Extraction des contextes pour chaque nom
            contextes = {}
            for nom in noms_extraits:
                # Trouver toutes les occurrences du nom dans le texte et extraire une fenêtre de contexte
                for match in re.finditer(rf"\b{re.escape(nom)}\b", texte):
                    start, end = match.start(), match.end()
                    # Extraire une fenêtre de 50 caractères avant et après
                    contexte = texte[max(0, start - 50):min(len(texte), end + 50)]
                    if nom not in contextes:
                        contextes[nom] = []
                    contextes[nom].append(contexte)

            # 4. Génération des embeddings contextuels
            embeddings = []
            nom_index_map = []
            for nom, contexte_list in contextes.items():
                for contexte in contexte_list:
                    embeddings.append(model.encode(contexte))  # Embedding basé sur le contexte
                    nom_index_map.append(nom)  # Pour suivre quel nom correspond à quel embedding

            # 5. Clustering avec DBSCAN sur les embeddings contextuels
            db = DBSCAN(eps=0.8, min_samples=2, metric='cosine')
            labels = db.fit_predict(embeddings)

            # 6. Regroupement des noms par clusters
            clusters = {}
            for idx, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(nom_index_map[idx])

            # Affichage des clusters
            print("Clusters d'alias détectés (avec contexte) :")
            for cluster_id, aliases in clusters.items():
                print(f"Cluster {cluster_id}: {aliases}")

            # 7. Construction d'un graphe
            G = nx.Graph()

            for cluster_id, aliases in clusters.items():
                main_alias = aliases[0]  # Choisir le premier nom comme alias principal
                for alias in aliases:
                    G.add_node(alias, names=";".join(aliases))  # Ajouter tous les alias comme attribut
                    if alias != main_alias:
                        G.add_edge(main_alias, alias)

            # Visualisation ou export
            nx.draw(G, with_labels=True)
