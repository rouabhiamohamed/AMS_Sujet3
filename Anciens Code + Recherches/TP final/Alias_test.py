import spacy
import pandas as pd
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import coreferee
import stanza

# Initialize models
nlp = spacy.load("fr_core_news_sm")  # Coreference resolution
nlp.add_pipe("coreferee")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # For sentence embeddings
stanza.download('fr')
nlp_stanza = stanza.Pipeline('fr', processors='lemma,tokenize,mwt,pos,depparse')

# Helper functions for alias detection and coreference resolution
def resolve_coreferences(text):
    doc = nlp(text)
    
    # Utilisation des chaînes de co-références
    coref_chains = doc._.coref_chains
    resolved_text = text
    
    # Si des co-références sont présentes
    if coref_chains:
        for chain in coref_chains:
            # Pour chaque chaîne, remplacez les co-références par un seul nom représentatif
            representative = chain.main.text  # Le premier élément de la chaîne (représentant la co-référence)
            for mention in chain.mentions:
                resolved_text = resolved_text.replace(mention.text, representative)
    
    return resolved_text

def detect_aliases(names):
    embeddings = model.encode(names)
    clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(embeddings)
    clusters = {i: [] for i in set(clustering.labels_) if i != -1}
    for idx, label in enumerate(clustering.labels_):
        if label != -1:
            clusters[label].append(names[idx])
    return clusters

def extract_characters(text):
    doc = nlp_stanza(text)
    characters = set()
    for sent in doc.sentences:
        for word in sent.words:
            if word.deprel == "nsubj" and word.upos == "PROPN":  # Proper nouns
                characters.add(word.text)
    return list(characters)

def create_graph(chapter_id, characters):
    G = nx.Graph()
    
    # Add nodes with names attribute
    for char in characters:
        # Resolve aliases (if any) and assign them as node names
        resolved_char = resolve_coreferences(char)
        G.add_node(resolved_char, names=resolved_char)
    
    # Create relationships (edges) based on some logic (e.g., co-appearance, dialogues)
    for char1 in characters:
        for char2 in characters:
            if char1 != char2:
                G.add_edge(char1, char2)
    
    return G

# Example input: list of chapters and book codes
books = [
    (list(range(0, 19)), "paf"),  # Prélude à Fondation
    (list(range(0, 18)), "lca")   # Les Cavernes d'Acier
]

# DataFrame for storing results
df_dict = {"ID": [], "graphml": []}

# Process each chapter in each book
for chapters, book_code in books:
    for chapter in chapters:
        if book_code == "paf":
            repertory = "prelude_a_fondation"
        else:
            repertory = "les_cavernes_d_acier"
        if chapter!=0:
            with open(f"reseaux-de-personnages-de-fondation-session-2/{repertory}/chapter_{chapter}.txt.preprocessed", "r") as file:
                texte = file.read()
            
            # Extract characters from chapter text
            characters = extract_characters(texte)
            
            # Detect and resolve aliases for characters
            aliases = detect_aliases(characters)
            
            # Create graph for this chapter
            G = create_graph(f"{book_code}{chapter}", characters)
            
            # Serialize graph to GraphML format
            graphml = "".join(nx.generate_graphml(G))
            
            # Add results to DataFrame
            df_dict["ID"].append(f"{book_code}{chapter}")
            df_dict["graphml"].append(graphml)

# Create a DataFrame to export the results
df = pd.DataFrame(df_dict)
df.set_index("ID", inplace=True)
df.to_csv("./graph_submission.csv")
