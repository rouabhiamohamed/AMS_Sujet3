import networkx as nx
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flair.data import Sentence
from flair.models import SequenceTagger
import re


# Initialiser le modèle de NER (Reconnaissance d'Entités Nommées)
tagger = SequenceTagger.load("flair/ner-french")

# Initialiser l'analyseur de sentiments
analyzer = SentimentIntensityAnalyzer()

def sentiment_analysis(text):
    sentiment_score = analyzer.polarity_scores(text)
    return sentiment_score['compound']

# Fonction pour extraire les personnages et analyser les relations
def extract_and_analyze_relations(pages):
    # Dictionnaire pour stocker les relations
    relations = {}
    G = nx.Graph()
    
    for page in pages:
        sentence = Sentence(page)
        tagger.predict(sentence)
        
        # Extraction des entités PERSON (personnages)
        characters = [entity.text for entity in sentence.get_spans('ner') if entity.get_label('ner').value == 'PER']
        
        # Analyser les relations entre les personnages
        for i in range(len(characters)):
            for j in range(i + 1, len(characters)):
                char1 = characters[i]
                char2 = characters[j]
                
                # Extraire le contexte autour des personnages
                context = page[max(page.find(char1) - 50, 0):min(page.find(char2) + len(char2) + 50, len(page))]
                
                # Analyser le sentiment du contexte pour déterminer la relation
                sentiment_score = sentiment_analysis(context)
                
                # Déterminer la polarité de la relation
                if sentiment_score > 0.1:
                    relation = 1  # Amitié
                elif sentiment_score < -0.1:
                    relation = -1  # Inamitié
                else:
                    relation = 0  # Neutralité

                # Ajouter la relation au graphe
                G.add_edge(char1, char2, weight=relation)
                relations[(char1, char2)] = relation
                
    return G, relations


def load_and_split_text(book_path):
    with open(book_path, "r") as file:
        texte = file.read()
    
    # Utilisation d'une expression régulière pour découper le texte en pages sur la base des délimiteurs <1>, <2>, etc.
    pages = re.split(r'<->', texte)
    return pages


books = [
    "Fondation_et_empire_sample"
    #"Fondation_sample",
    #"Seconde_Fondation_sample",
    #"Terre_et_Fondation_sample",
    #"Fondation_foudroyée_sample"
]
for book in books:
    pages = load_and_split_text(f"Textes_Processed/{book}.txt")
    for page in pages:
        G, relations = extract_and_analyze_relations(pages)

        # Identifier les cliques (groupes de personnages amis)
        cliques = list(nx.algorithms.clique.find_cliques(G))

        print("Relations entre personnages:", relations)
        print("Cliques d'amitié détectées:", cliques)

# Visualiser le graphe des relations avec pyvis
from pyvis.network import Network
net = Network(notebook=True)
net.from_nx(G)
net.show("graph.html")
