import stanza
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Charger le modèle Stanza pour le traitement du texte
nlp = stanza.Pipeline(lang='fr', processors='tokenize')

# Charger ton fichier de texte
with open("reseaux-de-personnages-de-fondation-session-2/prelude_a_fondation/chapter_1.txt.preprocessed", "r") as file:
    texte = file.read()

# Analyser le texte avec Stanza
doc = nlp(texte)

# Créer une liste pour stocker les phrases
sentences = [sentence.text for sentence in doc.sentences]

# Charger le modèle pour générer les embeddings
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

# Encoder les phrases
embeddings = model.encode(sentences)

# Calculer la similarité cosinus entre toutes les paires de phrases
similarities = cosine_similarity(embeddings)

# Afficher les paires de phrases avec une similarité supérieure à 0.60
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        similarity = similarities[i][j]
        if similarity > 0.60:  # Seules les similarités supérieures à 0.60 sont affichées
            print(f"Similarité entre '{sentences[i]}' et '{sentences[j]}': {similarity:.4f}")
            print('\n')