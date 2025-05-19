import stanza
from collections import defaultdict
import re

# Télécharger le modèle Stanza pour le français si nécessaire
stanza.download('fr')  # Télécharge et installe le modèle linguistique pour le français

# Initialisation du pipeline de traitement de texte pour le français
nlp = stanza.Pipeline('fr', processors='tokenize,mwt,pos,ner')
# Crée un pipeline de traitement avec les processeurs suivants : 
# 'tokenize' pour la tokenisation, 'mwt' pour le traitement des mots composés, 
# 'pos' pour l'étiquetage des parties du discours et 'ner' pour la reconnaissance des entités nommées.

### Liste des livres et chapitres à extraire        
books = [
    (list(range(1, 20)), "paf"),  # Livre "paf", avec les chapitres 1 à 19
    (list(range(1, 19)), "lca"),  # Livre "lca", avec les chapitres 1 à 18
]

# Parcours des livres et chapitres
for chapters, book_code in books:
    for chapter in chapters:
        # Détermine le répertoire correspondant selon le code du livre
        if book_code == "paf":
            repertory = "prelude_a_fondation"  # Livre "paf" -> répertoire "prelude_a_fondation"
        else:
            repertory = "les_cavernes_d_acier"  # Livre "lca" -> répertoire "les_cavernes_d_acier"
        
        # S'assure que le chapitre n'est pas le chapitre 0
        if chapter != 0:
            # Ouverture du fichier texte prétraité du chapitre courant
            with open(f"reseaux-de-personnages-de-fondation-session-2/{repertory}/chapter_{chapter}.txt.preprocessed", "r") as file:
                texte = file.read()  # Lecture du contenu du fichier
            
            # Remplacement des sauts de ligne inutiles
            texte = re.sub(r'(?<!-)\n', ' ', texte)  # Remplacer les sauts de ligne (\n) par des espaces, sauf quand précédé par un tiret
            texte = re.sub(r'-\n', '-', texte)  # Conserver les tirets avec les mots suivants
            texte = re.sub(r'—', "", texte)  # Supprimer les tirets longs "—"
            
            # Remplacer la ponctuation (points, points d'interrogation, points d'exclamation, parenthèses, guillemets) pour ajouter des retours à la ligne après
            texte = re.sub(r'\.(?!\s\»|\.|[^\s]\.| [a-z])','.\n', texte)
            texte = re.sub(r'\?(?!\s\»)','?\n', texte)
            texte = re.sub(r'\!(?!\s\»)','!\n', texte)
            texte = re.sub(r'\)(?!\s\»)',')\n', texte)
            texte = re.sub(r'\»(?!\.)','»\n', texte)
            
            
            # Traitement linguistique du texte avec Stanza
            doc = nlp(texte)
 
            # Parcours des phrases et mots du texte analysé
            for sent in doc.sentences:
                for word in sent.words:
                    # Vérifie si le mot est en majuscules et contient plus de deux lettres
                    if word.text.isupper() and len(word.text) > 2:
                        # Analyser le mot en majuscules
                        docMaj = nlp(word.text)
                        # Mettre le mot en majuscules avec la première lettre en majuscule
                        motEnMaj = word.text.capitalize()
                        # Vérifie si le mot est un nom propre (PROPN) dans le texte analysé
                        if (wordMaj.upos == "PROPN" for sentMaj in docMaj.sentences for wordMaj in sentMaj.words):
                            texte = texte.replace(word.text, motEnMaj)  # Remplace le mot en majuscules par sa version corrigée
            
            
            # Sauvegarde du texte traité dans un nouveau fichier
            with open(f"Textes_Processed/{repertory}_chapter_{chapter}.txt", "w") as file:
                file.write(texte)  # Écriture du texte traité dans un fichier de sortie
                print(f"Sauvegarde du fichier {repertory}_chapter_{chapter}.txt")
