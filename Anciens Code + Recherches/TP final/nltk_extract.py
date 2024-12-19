import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree

# Télécharger les ressources nécessaires
nltk.download('maxent_ne_chunker_tab')
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Liste des livres et chapitres
books = [
    (list(range(1, 19)), "paf"),
    (list(range(1, 18)), "lca"),
]

locations = []
personnages = []

# Traitement des livres et chapitres
for chapters, book_code in books:
    for chapter in chapters:
        # Définir le répertoire en fonction du code du livre
        if book_code == "paf":
            repertory = "prelude_a_fondation"
        else:
            repertory = "les_cavernes_d_acier"

        # Ouvrir le fichier correspondant
        try:
            with open(f"reseaux-de-personnages-de-fondation-session-2/{repertory}/chapter_{chapter}.txt.preprocessed", "r") as file:
                content = file.read()
        except FileNotFoundError:
            print(f"Le fichier chapter_{chapter}.txt.preprocessed n'a pas été trouvé.")
            continue

        # Traitement du texte avec NLTK
        words = word_tokenize(content)  # Tokenisation
        tagged = pos_tag(words)  # Etiquetage grammatical
        
        # Extraction des entités nommées
        tree = ne_chunk(tagged)  # Reconnaissance des entités

        # Parcourir l'arbre pour extraire les lieux (GPE)
        for subtree in tree:
            if isinstance(subtree, Tree):
                label = subtree.label()
                if label == 'GPE':  # Lieux géographiques
                    location = ' '.join(word for word, tag in subtree)
                    locations.append(location)
                if label == 'PERSON':  # Personnages
                    personnage = ' '.join(word for word, tag in subtree)
                    personnages.append(personnage)

# Affichage des lieux extraits
locations = list(dict.fromkeys(locations))
locations = list(dict.fromkeys(locations))
for location in locations:
    if(location in personnages):
        locations.remove(location)
       
print(locations)

if("Cléon" in personnages):
    print("Cléon")