from collections import defaultdict
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
import re

with open("reseaux-de-personnages-de-fondation-session-2/prelude_a_fondation/chapter_1.txt.preprocessed", "r") as file:
    texte = file.read()

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxen_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

# Traitement du texte avec spaCy
mots = word_tokenize(texte)

tags = pos_tag(mots)

arbre = ne_chunk(tags)

print(arbre)

entities = []
for subtree in arbre:
     if isinstance(subtree, nltk.Tree):
          entity = " ".join(word for word, tag in subtree.leaves())
          entities.append(entity)

print("Entités nommées :", entities)
