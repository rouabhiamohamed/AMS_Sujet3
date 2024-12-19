import stanza
import networkx as nx
import pandas as pd
from collections import defaultdict
import re

# Télécharger le modèle Stanza pour le français si nécessaire
stanza.download('fr')

# Initialisation du pipeline de traitement de texte pour le français
nlp = stanza.Pipeline('fr', processors='tokenize,mwt,pos,ner')
        
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
        if chapter!=0:
            with open(f"reseaux-de-personnages-de-fondation-session-2/{repertory}/chapter_{chapter}.txt.preprocessed", "r") as file:
                texte = file.read()
                
            # texte = re.sub(r'\b[A-ZÀ-ÿ0-9]+\b', lambda match: match.group(0).lower(), texte)    # nettoyage des mots tout en majuscule
            #texte = re.sub(r'-', " ", texte)
            
            texte = re.sub(r'(?<!-)\n', ' ', texte) # remplacer les caractères de nouvelle ligne (\n) par des espaces, sauf lorsque le caractère précédant le \n est un tiret (-)
            texte = re.sub(r'-\n', '-', texte)
            texte = re.sub(r'—', "", texte)
            
            texte = re.sub(r'\.(?!\s\»|\.|[^\s]\.| [a-z])','.\n', texte)

            texte = re.sub(r'\?(?!\s\»)','?\n', texte)
            texte = re.sub(r'\!(?!\s\»)','!\n', texte)
            texte = re.sub(r'\)(?!\s\»)',')\n', texte)
            
            texte = re.sub(r'\»(?!\.)','»\n', texte)
    
        
            doc = nlp(texte)
 
            for sent in doc.sentences:
                for word in sent.words:
                    if word.text.isupper() and len(word.text)>2:
                        docMaj = nlp(word.text)
                        motEnMaj = word.text.capitalize()
                        if(wordMaj.upos =="PROPN" for sentMaj in docMaj.sentences for wordMaj in sentMaj.words):
                            print(motEnMaj)
                            texte = texte.replace(word.text, motEnMaj)
                    
            with open(f"Textes_Processed/{repertory}_chapter_{chapter}.txt", "w") as file:
                file.write(texte)