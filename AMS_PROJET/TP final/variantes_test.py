from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import numpy as np


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
            with open(f"Partie2/{repertory}_chapter_{chapter}_characters.txt", "r") as file:
                personnages = [line.strip() for line in file if line.strip()]
                
            # Encoder les personnages
            embeddings = model.encode(personnages)

            # Appliquer DBSCAN
            db = DBSCAN(eps=0.5, min_samples=1, metric="cosine")
            labels = db.fit_predict(embeddings)

            # Regrouper les alias par cluster dans une liste de listes
            clusters = []
            unique_labels = set(labels)  # Trouver les labels uniques

            for label in unique_labels:
                cluster = [personnages[i] for i in range(len(labels)) if labels[i] == label]
                clusters.append(cluster)

            # Maintenant clusters est une liste de listes
            print("/////////////////",book_code,chapter,"////////////////////")
            print(clusters)

