from collections import defaultdict
import spacy

# Charger le modèle de NLP (assurez-vous que le modèle est installé)
nlp = spacy.load("fr_core_news_sm")  # Remplacez par le modèle approprié

# Liste des livres et chapitres
books = [
    (list(range(1, 19)), "paf"),
    (list(range(1, 18)), "lca"),
]

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

        # Initialisation d'un dictionnaire pour stocker les relations
        relations = defaultdict(set)

        # Extraire les entités "PERSON" (les noms des personnages)
        # Pour déboguer ou tester les entités, décommentez la ligne suivante
        # personnages = [ent.text for ent in doc.ents if ent.label_ == "PER"]
        # for ent in doc.ents :
        #     print(ent.text)

        # Parcourir les phrases du texte
        for sent in doc.sents:
            # Extraire les personnages dans la phrase
            personnes_dans_phrase = [ent.text for ent in sent.ents if ent.label_ == "PER" and len(ent.text) > 1 and ent.text[0].isupper()]
            print(personnes_dans_phrase)
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
                print(f"{p1} -> {p2}")
