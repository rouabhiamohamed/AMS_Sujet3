import stanza
import nltk
from nltk.util import bigrams
from nltk import FreqDist

# Téléchargez le modèle de langue de Stanza
stanza.download('fr')

# Initialisation du pipeline Stanza pour le français
nlp = stanza.Pipeline('fr')


###Liste des livres et chapitres à extraire        
books = [
    (list(range(1, 20)), "paf"),
    (list(range(1, 19)), "lca"),
]



for chapters, book_code in books:
    for chapter in chapters:
        if book_code == "paf":
            repertory = "prelude_a_fondation"
        else:
            repertory = "les_cavernes_d_acier"
        
        with open(f"Textes_Processed/{repertory}_chapter_{chapter}.txt", "r") as file:
            texte = file.read()


        # Traitement du texte avec Stanza
        doc = nlp(texte)

        # Récupérer les tokens (mots) du texte
        tokens = [word.text for sentence in doc.sentences for word in sentence.words]

        # Créer les bigrammes
        bigram_list = list(bigrams(tokens))

         # Compter la fréquence des bigrammes
        bigram_freq = FreqDist(bigram_list)

        # Écrire les bigrammes et leur fréquence dans un fichier
        with open(f"Bigrammes/Bigrammes_of_{repertory}_chapter_{chapter}.txt", 'w') as file:
            for bigram, freq in bigram_freq.items():
                # Écriture du bigramme et de sa fréquence
                file.write(f"{bigram[0]} {bigram[1]} {freq}\n")
    
    
    
    