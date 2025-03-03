import stanza
import networkx as nx
import pandas as pd
from collections import defaultdict
import re
from flair.data import Sentence
from flair.models import SequenceTagger
import numpy as np
from pyvis.network import Network

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Charger le modèle NER pour le français
#tagger = SequenceTagger.load("flair/ner-french")
analyzer = SentimentIntensityAnalyzer()


###Liste des livres        
books = [
    "Fondation_et_empire_sample"
]

def load_and_split_text(book_path):
    with open(book_path, "r") as file:
        texte = file.read()
    
    # Utilisation d'une expression régulière pour découper le texte en pages sur la base des délimiteurs <1>, <2>, etc.
    pages = re.split(r'<->', texte)
    return pages


def analyze_sentiment(text):
    sentiment_score = analyzer.polarity_scores(text)
    if sentiment_score['compound'] > 0:
        print("Sentiment positif")
    elif sentiment_score['compound'] < 0:
        print("Sentiment négatif")
    else:
        print("Sentiment neutre")
    return sentiment_score

#Liste de tout les lieux de l'entièreté du corpus
page_sentiment = defaultdict(list)

##Bout de code qui permet de mettre dans une liste tout les lieux du Corpus grâce à Flair
for book in books:
    print(f"////////////////////////////////////Livre {book}////////////////////////////////////")
    pages = load_and_split_text(f"Textes_Processed/{book}.txt")
    
    for i, page in enumerate(pages):
        page_sentiment[i].append(analyze_sentiment(page))

print(page_sentiment)
            