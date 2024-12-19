from collections import defaultdict
import spacy

# Charger le modèle de NLP (assurez-vous que le modèle est installé)
nlp = spacy.load("fr_core_news_lg")  # Remplacez par le modèle approprié

# Liste des livres et chapitres
books = [
    (list(range(1, 19)), "paf"),
    (list(range(1, 18)), "lca"),
]

list = []

# Traitement des livres et chapitres
for chapters, book_code in books:
    for chapter in chapters:
        # Définir le répertoire en fonction du code du livre
        if book_code == "paf":
            repertory = "prelude_a_fondation"
        else:
            repertory = "les_cavernes_d_acier"

        # Ouvrir le fichier correspondant
        with open(f"reseaux-de-personnages-de-fondation-session-2/{repertory}/chapter_{chapter}.txt.preprocessed", "r") as file:
            content = file.read()

        # Traitement du texte avec spaCy
        doc = nlp(content)


        # Extraire les entités "PERSON" (les noms des personnages)
        personnages = [ent.text for ent in doc.ents if ent.label_ == "LOC"]
        for perso in personnages:
           list.append(perso)

print(list)

       
