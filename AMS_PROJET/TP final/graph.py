import networkx as nx
import pandas as pd
from collections import defaultdict
import spacy

nlp = spacy.load("fr_core_news_sm")

# (chapitres, code du livre)
books = [
    (list(range(0, 19)), "paf"),
    (list(range(0, 18)), "lca"),
]

df_dict = {"ID": [], "graphml": []}

alias_dict = {}

for chapters, book_code in books:
    for chapter in chapters:

        G = nx.Graph()

        # Définir le répertoire en fonction du code du livre
        if book_code == "paf":
            repertory = "prelude_a_fondation"
        else:
            repertory = "les_cavernes_d_acier"
        if chapter==0:
            G = nx.Graph()

            df_dict["ID"].append("{}{}".format(book_code, chapter))

            graphml = "".join(nx.generate_graphml(G))
            df_dict["graphml"].append(graphml)
        else:
            # Ouvrir le fichier correspondant
            with open(f"reseaux-de-personnages-de-fondation-session-2/{repertory}/chapter_{chapter}.txt.preprocessed", "r") as file:
                content = file.read()

            # Traitement du texte avec spaCy
            doc = nlp(content)

            # Initialisation d'un dictionnaire pour stocker les relations
            relations = defaultdict(set)

            # Parcourir les phrases du texte
            for sent in doc.sents:
                # Extraire les personnages dans la phrase
                personnes_dans_phrase = [ent.text for ent in sent.ents if ent.label_ == "PER" and len(ent.text) > 1 and ent.text[0].isupper()]

                # Ajouter des relations entre chaque paire de personnages dans la même phrase
                for i in range(len(personnes_dans_phrase)):
                    for j in range(i + 1, len(personnes_dans_phrase)):
                        p1 = personnes_dans_phrase[i]
                        p2 = personnes_dans_phrase[j]
                        relations[p1].add(p2)
                        relations[p2].add(p1)  # Ajouter aussi l'inverse (relation symétrique)

            # Afficher les relations détectées
            print("Relations détectées:")
            for p1, related_personnes in relations.items():
                for p2 in related_personnes:
                    G.add_edge(p1, p2)


            for personnage in G.nodes:
                if personnage not in alias_dict:
                    # Si le personnage n'a pas encore d'alias, on le définit simplement par son nom
                    alias_dict[personnage] = [personnage]

                # Assigner les alias trouvés au nœud
                G.nodes[personnage]["names"] = ";".join(alias_dict[personnage])

            df_dict["ID"].append("{}{}".format(book_code, chapter))

            graphml = "".join(nx.generate_graphml(G))
            df_dict["graphml"].append(graphml)

df = pd.DataFrame(df_dict)
df.set_index("ID", inplace=True)
df.to_csv("./my_submission.csv")
